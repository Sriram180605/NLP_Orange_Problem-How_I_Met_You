# The Orange Problem - ChartQA Multimodal Fine-Tuning

**Team Name:** How I Met You  
**Team Members:**  
- Sai Amarnath G - PES1UG23AM255  
- Sai Abhinav K - PES1UG23AM254  
- Shreyas S - PES1UG23AM295  
- S Sriram - PES1UG23AM248  

Fine-tuning Qwen2-VL-2B-Instruct on [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) using QLoRA via Unsloth. The model answers questions about charts with a number or short phrase.

**Model on HuggingFace:** [Sriram1806/NLP_Orange_Problem-How_I_Met_You](https://huggingface.co/Sriram1806/NLP_Orange_Problem-How_I_Met_You)  
**GitHub:** [Sriram180605/NLP_Orange_Problem-How_I_Met_You](https://github.com/Sriram180605/NLP_Orange_Problem-How_I_Met_You)  
**Hardware:** Kaggle T4 x 2 | **Runtime:** ~155 min | **Framework:** Unsloth + TRL SFTTrainer

---

## Results (300 test samples)

| Metric | Base Model | Fine-Tuned | Delta |
|---|---|---|---|
| Relaxed Accuracy (primary) | 54.33% | 59.67% | +5.33% |
| Exact Match | 50.00% | 54.67% | +4.67% |
| ROUGE-L | 54.74% | 59.17% | +4.42% |

Relaxed Accuracy is the primary ChartQA metric: correct if exact string match OR numeric agreement within 5%.

---

## Approach, Assumptions and Observations

### Approach
We fine-tuned Qwen2-VL-2B-Instruct on ChartQA using QLoRA via Unsloth. The model is trained with a Chain-of-Thought prompt format that includes an `<ANSWER>:` marker, encouraging the model to briefly reason before giving a final answer. We used SFTTrainer with UnslothVisionDataCollator for correct multimodal batching and label masking. LoRA adapters are applied to all language attention and MLP layers. After training, the adapters are merged into the base model using `merge_and_unload()` to produce a standalone model that requires no special loading.

### Assumptions
- 6,000 training samples is sufficient for meaningful improvement on ChartQA within the T4 time budget
- The base model's frozen ViT encoder already extracts sufficient visual features; fine-tuning only the language layers is enough to improve chart reasoning
- fp16 precision is required due to a bitsandbytes/Unsloth compatibility issue with 4-bit quantisation on ViT MLP blocks
- A Chain-of-Thought style prompt with a structured answer marker improves answer extraction reliability compared to a plain prompt

### Observations
- Fine-tuning improved Relaxed Accuracy by +5.33% (54.33% to 59.67%) over the base model
- Training loss decreased steadily from ~0.215 to ~0.160 over 750 steps with no signs of overfitting
- The model handles direct value lookups well but struggles with arithmetic questions such as computing differences between bars
- The base model is already surprisingly strong on ChartQA due to Qwen2-VL's pre-training on visual data, making further gains harder to achieve
- Using a plain neutral prompt for the base model evaluation (without the CoT format) gives a fairer comparison than using the fine-tuned model's prompt

---

## Quickstart - Run Inference

```bash
pip install transformers qwen-vl-utils torch
```

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Sriram1806/NLP_Orange_Problem-How_I_Met_You",
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Sriram1806/NLP_Orange_Problem-How_I_Met_You")

def ask_chart(image, question):
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": (
            "You are a chart analysis assistant. "
            "When given a chart and a question, briefly identify the relevant data, "
            "then output your final answer after '<ANSWER>:'. "
            "The answer must be a number, percentage, or short phrase only."
        )}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                "Look at the chart carefully.\n"
                f"Question: {question}\n\n"
                "Briefly identify the relevant data point, then give your answer as:\n"
                "<ANSWER>: [value]"
            )},
        ]},
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False,
                             temperature=None, top_p=None)
    gen = out[0][inputs["input_ids"].shape[-1]:]
    raw = processor.decode(gen, skip_special_tokens=True).strip()
    return raw.split("<ANSWER>:")[-1].strip() if "<ANSWER>:" in raw else raw

image = Image.open("your_chart.png")
print(ask_chart(image, "What is the value in 2020?"))
```

---

## Repository Structure

```
orange-finished.ipynb        # Full training + evaluation notebook
upload-to-huggingface.ipynb  # Merges LoRA adapters and uploads full model to HuggingFace
inference.ipynb              # Loads model from HuggingFace and runs inference
README.md
```

---

## How to Reproduce Training

1. Open orange-finished.ipynb in Kaggle
2. Set Accelerator to GPU T4 x 2, Internet ON
3. Add your HuggingFace token as a Kaggle secret named HF_TOKEN
4. Run All

---

## Pushing the Model to HuggingFace

To merge the LoRA adapters and push the full model to HuggingFace, run upload-to-huggingface.ipynb in Kaggle:

1. Add the training notebook output as input data
2. Set Accelerator to GPU T4, Internet ON
3. Replace hf_YOUR_TOKEN_HERE with your HuggingFace write token
4. Run All

This loads the LoRA adapters, merges them into the base model with merge_and_unload(), and pushes the full 4.4 GB merged model to HuggingFace.

---

## Running Inference from HuggingFace

Open inference.ipynb in Kaggle with Internet ON and run all cells. It will:
1. Install required packages
2. Load the merged model directly from HuggingFace
3. Load the ChartQA test set
4. Run inference on the first test sample and print the result

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2-VL-2B-Instruct |
| Dataset | HuggingFaceM4/ChartQA |
| Training samples | 6,000 |
| Epochs | 2 |
| Steps completed | 750 |
| Train time | 155.1 min |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| LoRA dropout | 0.0 |
| LoRA targets | Language attention + MLP layers |
| Learning rate | 2e-4 |
| LR schedule | Cosine |
| Effective batch size | 16 (2 per device x 8 grad accum) |
| Precision | float16 |
| Max sequence length | 640 tokens |
| Optimizer | AdamW 8-bit |
| Warmup steps | 50 |

### Key Decisions

**fp16 instead of 4-bit:** Unsloth compiled VisionMlp_forward triggers a bitsandbytes Linear4bit recursion error when the ViT MLP blocks are quantised. Loading in fp16 avoids this. Two T4s (31 GB total) hold the 2B model comfortably.

**finetune_vision_layers=False:** The ViT MLP blocks cannot be safely LoRA-wrapped alongside Unsloth gradient checkpointing when quantised. Language layers are fine-tuned; the frozen ViT encoder still provides full visual feature extraction.

**CoT prompt with ANSWER: marker:** Training the model to emit a brief reasoning step before a structured answer tag improves extraction reliability at inference time and nudges it toward the correct format expected by ChartQA evaluation.

**normalise_answer before EM scoring:** Strips commas and % signs before comparison so 44% matches 44 and 1,000 matches 1000.

**Relaxed Accuracy as primary metric:** ChartQA's official metric allows numeric predictions within 5% of the gold answer to be counted as correct, which is more meaningful than strict exact match for chart question answering.