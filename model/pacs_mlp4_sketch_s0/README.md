---
base_model: google/vit-base-patch16-224-in21k
library_name: peft
tags:
- lora
- vision-transformer
- domain-generalization
- pacs
---

# LoRA adapter: ViT-B/16 on PACS, sketch held out

- **Base model:** `google/vit-base-patch16-224-in21k`, frozen.
- **Adapter:** LoRA rank 4 (alpha 8, dropout 0.1) on the MLP layers (`fc1`, `fc2`) of all 12 blocks, plus a trained
  7-class linear head. 374,023 trainable parameters.
- **Training data:** PACS photo, art painting and cartoon (90% of each; 10% used for validation and epoch selection).
- **Held-out domain:** sketch, never seen during training. Accuracy on all 3,929 sketch images: **76.6%**
  (macro-F1 78.4%).
- **Training:** 4 epochs, AdamW, learning rate 3e-3, batch 64, seed 0, fp16 autocast.
- **Classes:** dog, elephant, giraffe, guitar, horse, house, person.

Use `predict.py` in the repository root, or:

```python
from peft import PeftModel
from transformers import AutoModelForImageClassification
base = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=7)
model = PeftModel.from_pretrained(base, "model/pacs_mlp4_sketch_s0").eval()
```
