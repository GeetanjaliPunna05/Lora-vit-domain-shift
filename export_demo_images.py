"""export_demo_images.py - writes a few PACS images to demo_images/ (sketch = unseen target domain, photo = source)."""
import argparse
import os

import numpy as np
from PIL import Image

from vit_dg import DATASET_CLASSES, DATASETS, load_cached

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache")
    args = ap.parse_args()
    x, y, d = load_cached("pacs", args.cache_dir)
    domains, classes = DATASETS["pacs"][1], DATASET_CLASSES["pacs"]
    rng = np.random.RandomState(7)
    os.makedirs("demo_images", exist_ok=True)
    for c, name in enumerate(classes):
        sk = rng.choice(np.where((d == domains.index("sketch")) & (y == c))[0], 2, replace=False)
        for k, i in enumerate(sk):
            Image.fromarray(np.array(x[i])).save(f"demo_images/sketch_{name}_{k + 1}.png")
        ph = rng.choice(np.where((d == domains.index("photo")) & (y == c))[0])
        Image.fromarray(np.array(x[ph])).save(f"demo_images/photo_{name}.png")
    print(len(os.listdir("demo_images")), "images written to demo_images/")
