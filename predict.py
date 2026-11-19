"""
predict.py - classify images with the trained LoRA adapter (no training needed).

The adapter in model/pacs_mlp4_sketch_s0 was trained on PACS photo, art painting and cartoon only;
sketches were never seen during training, so sketch images show the domain-shift setting of the project.

  python predict.py                              # all images in demo_images/
  python predict.py my_image.jpg other.png       # your own images

The frozen ViT-B/16 backbone (~350 MB) is downloaded from the Hugging Face Hub on first use.
Runs on CPU or GPU.
"""
import argparse
import glob
import json
import os

import torch
from PIL import Image
from torchvision.transforms import v2

ADAPTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "pacs_mlp4_sketch_s0")


def load_model(adapter_dir=ADAPTER, device="cpu"):
    from peft import PeftModel
    from transformers import AutoModelForImageClassification
    info = json.load(open(os.path.join(adapter_dir, "demo_info.json")))
    base = AutoModelForImageClassification.from_pretrained(
        info["base_model"], num_labels=len(info["classes"]), ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
    return model, info


def preprocess(info):
    return v2.Compose([v2.ToImage(), v2.Resize(256, antialias=True), v2.CenterCrop(224),
                       v2.ToDtype(torch.float32, scale=True), v2.Normalize(info["image_mean"], info["image_std"])])


@torch.no_grad()
def predict(model, info, paths, device="cpu"):
    tf = preprocess(info)
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(device)
    probs = model(pixel_values=x).logits.softmax(-1).cpu()
    return [(p, info["classes"][int(pr.argmax())], float(pr.max()), pr) for p, pr in zip(paths, probs)]


def true_label(path, classes):
    name = os.path.basename(path).lower()
    return next((c for c in classes if f"_{c}" in name), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="image files (default: demo_images/*.png)")
    ap.add_argument("--adapter", default=ADAPTER)
    args = ap.parse_args()
    paths = args.images or sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_images", "*.png")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = load_model(args.adapter, device)
    print(f"Adapter: {info['config']} LoRA, trained without the '{info['held_out_domain']}' domain "
          f"(test accuracy on all {info['held_out_domain']} images: {info['test_acc'] * 100:.1f}%)\n")
    right = total = 0
    for p, pred, conf, _ in predict(model, info, paths, device):
        truth = true_label(p, info["classes"])
        mark = "" if truth is None else ("  correct" if truth == pred else f"  WRONG (true: {truth})")
        right += truth == pred
        total += truth is not None
        print(f"{os.path.basename(p):28s} -> {pred:9s} {conf * 100:5.1f}%{mark}")
    if total:
        print(f"\n{right}/{total} correct")


if __name__ == "__main__":
    main()
