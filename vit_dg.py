"""
vit_dg.py - LoRA placement / layer-wise rank / QLoRA study on a frozen ViT under domain shift.

Protocol: leave-one-domain-out (LODO). Three source domains are split 90/10 into train/val
(fixed split, independent of the training seed); the held-out target domain is test only.
Model selection uses source-domain validation accuracy only (best epoch); the target domain
is evaluated once, with the selected checkpoint.

Examples
  # tiny CPU smoke test
  python vit_dg.py --dataset pacs --model deit_s --grid smoke --targets sketch --seeds 0 --epochs 1 --limit 64
  # full PACS grid, split across 2 GPUs (run one process per GPU)
  CUDA_VISIBLE_DEVICES=0 python vit_dg.py --dataset pacs --model vit_b --grid pacs --shard 0/2
  CUDA_VISIBLE_DEVICES=1 python vit_dg.py --dataset pacs --model vit_b --grid pacs --shard 1/2
  # print trainable-parameter budgets of every config without training
  python vit_dg.py --model vit_b --grid pacs --dry_run
"""
import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

DATASETS = {
    "pacs": ("flwrlabs/pacs", ["art_painting", "cartoon", "photo", "sketch"]),
    "officehome": ("flwrlabs/office-home", ["Art", "Clipart", "Product", "Real World"]),
}
DATASET_CLASSES = {
    "pacs": ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
}
MODELS = {
    "deit_s": "facebook/deit-small-patch16-224",
    "vit_b": "google/vit-base-patch16-224-in21k",
}

# ---------------------------------------------------------------------------------------------
# Experiment configurations. Ranks are given per third of the network (early, middle, late);
# for 12-block models that is blocks 0-3, 4-7, 8-11. "attn" = LoRA on attention projections
# listed in attn_mods (q,k,v,o); "mlp" = LoRA on both MLP linears (fc1, fc2).
# Per-block parameter cost for ViT-B (d=768, mlp=3072): q or v -> 1536*r, fc1 or fc2 -> 3840*r,
# so attn(qv) r=10 == mlp r=4 == attn r=5 + mlp r=2 (30,720 params/block): budget-matched.
# early/middle/late r=12 on 4 blocks == all blocks r=4 (147,456 params): budget-matched.
# inc/dec/midheavy sum to the same total rank as uniform r=4: budget-matched.
# ---------------------------------------------------------------------------------------------
def _lora(attn=(0, 0, 0), mlp=(0, 0, 0), attn_mods="qv", quant=False):
    return {"method": "lora", "attn": attn, "mlp": mlp, "attn_mods": attn_mods, "quant": quant}


CONFIGS = {
    "linear": {"method": "linear"},
    "bitfit": {"method": "bitfit"},
    # RQ2 rank sweep (all blocks, attention q/v)
    "r1": _lora(attn=(1, 1, 1)),
    "r2": _lora(attn=(2, 2, 2)),
    "r4": _lora(attn=(4, 4, 4)),
    "r8": _lora(attn=(8, 8, 8)),
    "r16": _lora(attn=(16, 16, 16)),
    # RQ1 placement, same budget as r4
    "early": _lora(attn=(12, 0, 0)),
    "middle": _lora(attn=(0, 12, 0)),
    "late": _lora(attn=(0, 0, 12)),
    # RQ2 layer-wise rank allocation, same budget as r4
    "inc": _lora(attn=(1, 4, 7)),
    "dec": _lora(attn=(7, 4, 1)),
    "midheavy": _lora(attn=(2, 8, 2)),
    # RQ3 attention vs MLP, budget-matched
    "attn10": _lora(attn=(10, 10, 10)),
    "mlp4": _lora(mlp=(4, 4, 4)),
    "both": _lora(attn=(5, 5, 5), mlp=(2, 2, 2)),
    # RQ4 4-bit NF4 backbone (pairs with r4 and both)
    "qlora_r4": _lora(attn=(4, 4, 4), quant=True),
    "qlora_both": _lora(attn=(5, 5, 5), mlp=(2, 2, 2), quant=True),
}
GRIDS = {
    "smoke": ["linear", "bitfit", "r4", "early"],
    "smoke_q": ["r4", "qlora_r4"],
    # LR tuning (run with --lrs ... --seeds 0); pick per method by source-validation accuracy,
    # then set DEFAULT_LR. LoRA LR is shared by all LoRA/QLoRA configs.
    "tune": ["linear", "bitfit", "r4"],
    "pacs": list(CONFIGS),
    "officehome": ["linear", "bitfit", "r1", "r4", "r16", "early", "middle", "late",
                   "inc", "dec", "attn10", "mlp4", "qlora_r4"],
    # reduced grids for the time-boxed study (every RQ still covered)
    "pacs_core": ["linear", "bitfit", "r1", "r4", "r16", "early", "middle", "late", "inc", "dec",
                  "attn10", "mlp4", "qlora_r4"],
    "officehome_core": ["linear", "bitfit", "r4", "early", "late", "attn10", "mlp4", "qlora_r4"],
}
TUNE_REP = {"linear": "linear", "bitfit": "bitfit", "lora": "r4"}  # config whose tuned LR each method uses
DEFAULT_LR = {"linear": 1e-3, "bitfit": 1e-3, "lora": 1e-3}

# module names relative to a block: transformers v5 naming first, v4 naming second
ATTN_SUFFIX = {"q": ("attention.q_proj", "attention.attention.query"),
               "k": ("attention.k_proj", "attention.attention.key"),
               "v": ("attention.v_proj", "attention.attention.value"),
               "o": ("attention.o_proj", "attention.output.dense")}
MLP_SUFFIX = ("mlp.fc1", "mlp.fc2", "intermediate.dense", "output.dense")


# ---------------------------------------------------------------------------------------------
# Data: decode every image once to a 256x256 uint8 memmap (shorter side resize + center crop),
# so training does cheap tensor augmentations instead of JPEG decoding.
# ---------------------------------------------------------------------------------------------
def _prep_batch(batch, res):
    from PIL import Image
    out = []
    for im in batch["image"]:
        im = im.convert("RGB")
        w, h = im.size
        s = res / min(w, h)
        im = im.resize((max(res, round(w * s)), max(res, round(h * s))), Image.BICUBIC)
        w, h = im.size
        left, top = (w - res) // 2, (h - res) // 2
        out.append(np.asarray(im.crop((left, top, left + res, top + res)), dtype=np.uint8).tobytes())
    return {"px": out}


def load_cached(name, cache_dir, res=256):
    repo, domains = DATASETS[name]
    os.makedirs(cache_dir, exist_ok=True)
    fx, fm = (os.path.join(cache_dir, f"{name}_{res}{s}") for s in ("_x.npy", "_meta.npz"))
    if not (os.path.exists(fx) and os.path.exists(fm)):
        from datasets import load_dataset
        ds = load_dataset(repo, split="train")
        print(f"[data] caching {name}: {len(ds)} images -> {fx}", flush=True)
        y = np.asarray(ds["label"], dtype=np.int64)
        d = np.asarray([domains.index(v) for v in ds["domain"]], dtype=np.int64)
        ds = ds.map(_prep_batch, batched=True, batch_size=64, remove_columns=["image"],
                    num_proc=max(1, min(8, os.cpu_count() or 1)), fn_kwargs={"res": res})
        tmp = fx + ".tmp.npy"
        x = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(ds), res, res, 3))
        i = 0
        for b in ds.iter(batch_size=512):
            for px in b["px"]:
                x[i] = np.frombuffer(px, dtype=np.uint8).reshape(res, res, 3)
                i += 1
        x.flush()
        del x
        os.replace(tmp, fx)
        np.savez(fm, y=y, d=d)
    meta = np.load(fm)
    return np.load(fx, mmap_mode="r"), meta["y"], meta["d"]


def lodo_split(y, d, target, val_frac=0.1, limit=None, split_seed=0):
    rng = np.random.RandomState(split_seed)
    tr, va = [], []
    for dom in np.unique(d):
        idx = np.where(d == dom)[0]
        if dom == target:
            continue
        idx = rng.permutation(idx)
        if limit:
            idx = idx[:limit]
        n_val = max(1, int(round(len(idx) * val_frac)))
        va.append(idx[:n_val])
        tr.append(idx[n_val:])
    te = np.where(d == target)[0]
    if limit:
        te = rng.permutation(te)[:limit]
    return np.concatenate(tr), np.concatenate(va), te


class ImgDS(Dataset):
    def __init__(self, x, y, idx, train, mean, std, size=224):
        # keep only the memmap's path: on Windows (spawn) a pickled memmap would be copied whole to every worker
        self.x_path, self.x, self.y, self.idx = x.filename, None, y, idx
        ops = ([v2.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
                v2.RandomHorizontalFlip(), v2.ColorJitter(0.3, 0.3, 0.3)] if train
               else [v2.CenterCrop(size)])
        self.tf = v2.Compose(ops + [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.x is None:
            self.x = np.load(self.x_path, mmap_mode="r")
        j = self.idx[i]
        img = torch.from_numpy(np.array(self.x[j])).permute(2, 0, 1)
        return self.tf(img), int(self.y[j])


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------
def lora_targets(model, cfg, depth):
    """Explicit module names and per-module ranks for a LoRA config."""
    thirds = [min(2, i * 3 // depth) for i in range(depth)]
    attn_sfx = [s for c in cfg["attn_mods"] for s in ATTN_SUFFIX[c]]
    ranks = {}
    for name, mod in model.named_modules():
        m = re.search(r"(?:encoder\.layer|layers)\.(\d+)\.(.+)$", name)
        if not m or not isinstance(mod, torch.nn.Linear):
            continue
        blk, rel = int(m.group(1)), m.group(2)
        r = cfg["attn"][thirds[blk]] if rel in attn_sfx else cfg["mlp"][thirds[blk]] if rel in MLP_SUFFIX else 0
        if r > 0:
            ranks[name] = r
    return ranks


def build_model(cfg, model_key, num_labels, device, half_dtype, lora_dropout):
    from transformers import AutoModelForImageClassification
    kw = {"num_labels": num_labels, "ignore_mismatched_sizes": True, "attn_implementation": "sdpa"}
    if device.type == "cuda":
        kw["dtype"] = half_dtype  # frozen backbone kept in 16-bit for every method
    if cfg.get("quant"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=half_dtype, llm_int8_skip_modules=["classifier"])
        kw["device_map"] = {"": device.index or 0}
    model = AutoModelForImageClassification.from_pretrained(MODELS[model_key], **kw)
    for p in model.parameters():
        p.requires_grad = False
    depth = model.config.num_hidden_layers

    if cfg["method"] == "linear":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif cfg["method"] == "bitfit":
        for n, p in model.named_parameters():
            if n.endswith("bias") or n.startswith("classifier"):
                p.requires_grad = True
    elif cfg["method"] == "lora":
        from peft import LoraConfig, get_peft_model
        ranks = lora_targets(model, cfg, depth)
        assert ranks, "LoRA config selected no modules"
        lcfg = LoraConfig(
            target_modules=list(ranks), r=max(ranks.values()), lora_alpha=2 * max(ranks.values()),
            rank_pattern={re.escape(k): v for k, v in ranks.items()},
            alpha_pattern={re.escape(k): 2 * v for k, v in ranks.items()},
            lora_dropout=lora_dropout, bias="none", modules_to_save=["classifier"])
        model = get_peft_model(model, lcfg)
    # trainable params live in fp32; frozen weights stay 16-bit / 4-bit
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if not cfg.get("quant"):
        model.to(device)
    return model


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "classifier" in n)
    return trainable, trainable - head


# ---------------------------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, num_classes):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(pixel_values=x).logits
        probs.append(logits.float().softmax(-1).cpu())
        ys.append(y)
    p, y = torch.cat(probs), torch.cat(ys)
    conf, pred = p.max(-1)
    correct = (pred == y).float()
    # 15-bin expected calibration error
    bins = torch.clamp((conf * 15).long(), max=14)
    ece = sum((bins == b).float().mean() * (correct[bins == b].mean() - conf[bins == b].mean()).abs()
              for b in range(15) if (bins == b).any())
    return {"acc": correct.mean().item(),
            "f1": f1_score(y, pred, average="macro", labels=list(range(num_classes)), zero_division=0),
            "nll": F.nll_loss(p.clamp_min(1e-12).log(), y).item(),
            "ece": float(ece)}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_one(job, data, args, device):
    cfg = CONFIGS[job["config"]]
    x, y, d = data
    _, domains = DATASETS[job["dataset"]]
    target = domains.index(job["target"])
    num_classes = int(y.max()) + 1
    tr, va, te = lodo_split(y, d, target, limit=args.limit)

    is_cuda = device.type == "cuda"
    if is_cuda:
        bf16 = torch.cuda.get_device_capability(device)[0] >= 8
        amp_dtype = torch.bfloat16 if bf16 else torch.float16
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    else:
        amp_dtype = None

    set_seed(job["seed"])
    model = build_model(cfg, job["model"], num_classes, device, amp_dtype or torch.float32, args.lora_dropout)
    n_train, n_adapter = count_params(model)
    weights_mem = torch.cuda.memory_allocated(device) if is_cuda else 0

    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(MODELS[job["model"]])
    mean, std = proc.image_mean, proc.image_std
    g = torch.Generator().manual_seed(job["seed"])
    lkw = {"num_workers": args.workers, "pin_memory": is_cuda, "persistent_workers": args.workers > 0}
    train_dl = DataLoader(ImgDS(x, y, tr, True, mean, std), batch_size=args.batch_size, shuffle=True,
                          drop_last=len(tr) > args.batch_size, generator=g, **lkw)
    val_dl = DataLoader(ImgDS(x, y, va, False, mean, std), batch_size=args.batch_size * 2, **lkw)
    test_dl = DataLoader(ImgDS(x, y, te, False, mean, std), batch_size=args.batch_size * 2, **lkw)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=job["lr"], weight_decay=0.0 if cfg["method"] == "bitfit" else args.wd)
    total = args.epochs * len(train_dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    best, best_state, train_time, history = -1.0, None, 0.0, []
    for ep in range(args.epochs):
        model.train()
        if is_cuda:
            torch.cuda.synchronize(device)
        t0, loss_sum, n = time.time(), 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = F.cross_entropy(model(pixel_values=xb).logits.float(), yb, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(yb)
            n += len(yb)
        if is_cuda:
            torch.cuda.synchronize(device)
        train_time += time.time() - t0
        v = evaluate(model, val_dl, device, amp_dtype, num_classes)
        history.append({"epoch": ep, "loss": loss_sum / n, "val_acc": v["acc"]})
        print(f"  ep{ep} loss {loss_sum / n:.4f} val_acc {v['acc']:.4f}", flush=True)
        if v["acc"] > best:
            best = v["acc"]
            best_state = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}

    named = dict(model.named_parameters())
    for k, val in best_state.items():
        named[k].data.copy_(val.to(named[k].device))
    test = evaluate(model, test_dl, device, amp_dtype, num_classes)
    if args.save_dir and cfg["method"] == "lora":
        # LoRA weights + trained head only (a few MB); the frozen backbone is reloaded from the Hub
        out = os.path.join(args.save_dir, f"{job['dataset']}_{job['config']}_{job['target'].replace(' ', '_')}_s{job['seed']}")
        model.save_pretrained(out)
        with open(os.path.join(out, "demo_info.json"), "w") as f:
            json.dump({"base_model": MODELS[job["model"]], "dataset": job["dataset"], "config": job["config"],
                       "held_out_domain": job["target"], "seed": job["seed"], "test_acc": test["acc"],
                       "classes": DATASET_CLASSES.get(job["dataset"]), "image_mean": list(mean),
                       "image_std": list(std)}, f, indent=1)
        print(f"  saved adapter to {out}", flush=True)
    peak = torch.cuda.max_memory_allocated(device) if is_cuda else 0
    del model, opt
    return {**job, "method": cfg["method"], "test_acc": test["acc"], "test_f1": test["f1"],
            "test_ece": test["ece"], "test_nll": test["nll"], "val_acc": best,
            "trainable_params": n_train, "adapter_params": n_adapter,
            "weights_mem_mb": weights_mem / 2**20, "peak_mem_mb": peak / 2**20,
            "train_time_s": train_time, "img_per_s": args.epochs * len(tr) / max(train_time, 1e-9),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te), "epochs": args.epochs,
            "batch_size": args.batch_size, "amp": str(amp_dtype), "history": history,
            "device": torch.cuda.get_device_name(device) if is_cuda else "cpu"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pacs", choices=list(DATASETS))
    ap.add_argument("--model", default="vit_b", choices=list(MODELS))
    ap.add_argument("--grid", default=None, choices=list(GRIDS))
    ap.add_argument("--configs", nargs="*", default=None, help="explicit config names (overrides --grid)")
    ap.add_argument("--targets", nargs="*", default=None, help="held-out domains (default: all)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--lrs", nargs="*", type=float, default=None, help="LR sweep (default: per-method)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None, help="cap images per domain (smoke tests)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out", default="results/results.jsonl")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--dry_run", action="store_true", help="print parameter budgets and exit")
    ap.add_argument("--save_dir", default=None, help="save trained LoRA adapter + head (LoRA configs only)")
    ap.add_argument("--tuned_from", default=None, help="dataset whose tuning runs (in --out) set the LRs")
    ap.add_argument("--hours", type=float, default=None,
                    help="time budget; stop starting new runs when the next one may not finish in time")
    args = ap.parse_args()
    start = time.time()

    configs = args.configs or GRIDS[args.grid or "smoke"]
    if args.dry_run:
        for c in configs:
            cfg = CONFIGS[c]
            m = build_model({**cfg, "quant": False}, args.model, 7, torch.device("cpu"), torch.float32, 0.0)
            print(f"{c:12s} trainable={count_params(m)[0]:>9,d}  excl. head={count_params(m)[1]:>9,d}")
        return

    _, domains = DATASETS[args.dataset]
    targets = args.targets or domains
    lr_for = dict(DEFAULT_LR)
    if args.tuned_from and os.path.exists(args.out):
        # per-method LR = best mean source-validation accuracy among tuning runs (never target accuracy)
        rows = [json.loads(l) for l in open(args.out) if l.strip()]
        for method, rep in TUNE_REP.items():
            acc = {}
            for r in rows:
                if (r["dataset"], r["model"], r["config"], r["epochs"]) == (args.tuned_from, args.model, rep, args.epochs):
                    acc.setdefault(r["lr"], []).append(r["val_acc"])
            if acc:
                lr_for[method] = max(acc, key=lambda k: np.mean(acc[k]))
        print(f"[lr] tuned on {args.tuned_from}: {lr_for}", flush=True)
    # seed-major order: if time runs out, every config still has its first seed(s) complete
    jobs = [{"dataset": args.dataset, "model": args.model, "config": c, "target": t, "seed": s, "lr": lr}
            for s in args.seeds for c in configs
            for lr in (args.lrs or [lr_for[CONFIGS[c]["method"]]]) for t in targets]
    tag = f"e{args.epochs}" + (f"_lim{args.limit}" if args.limit else "")
    for j in jobs:
        j["run_id"] = f"{j['dataset']}|{j['model']}|{j['config']}|{j['target']}|s{j['seed']}|lr{j['lr']:g}|{tag}"
    i, n = map(int, args.shard.split("/"))
    jobs = jobs[i::n]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["run_id"] for l in f if l.strip()}
    todo = [j for j in jobs if j["run_id"] not in done]
    print(f"[jobs] {len(jobs)} in shard, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    data = load_cached(args.dataset, args.cache_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    longest = 0.0
    for k, job in enumerate(todo):
        if args.hours and time.time() - start + 1.3 * longest > args.hours * 3600:
            print(f"[budget] stopping: {len(todo) - k} runs left for the next session", flush=True)
            break
        print(f"[{k + 1}/{len(todo)}] {job['run_id']}", flush=True)
        t0 = time.time()
        res = run_one(job, data, args, device)
        res["wall_time_s"] = time.time() - t0
        longest = max(longest, res["wall_time_s"])
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"  -> test_acc {res['test_acc']:.4f} f1 {res['test_f1']:.4f} params {res['trainable_params']:,} "
              f"peak {res['peak_mem_mb']:.0f}MB time {res['train_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
