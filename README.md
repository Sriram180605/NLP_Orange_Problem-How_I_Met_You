# The Orange Problem - ChartQA Multimodal Fine-Tuning

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

## Quickstart

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
orange-finished.ipynb   # Full training + evaluation notebook
README.md
```

---

## How to Reproduce

1. Open orange-finished.ipynb in Kaggle
2. Set Accelerator to GPU T4 x 2, Internet ON
3. Add your HuggingFace token as a Kaggle secret named HF_TOKEN
4. Run All

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

**CoT prompt with ANSWER: marker:** Training the model to emit a brief reasoning step before a structured answer tag improves extraction reliability at inference time.

**normalise_answer before EM scoring:** Strips commas and % signs before comparison so 44% matches 44 and 1,000 matches 1000.