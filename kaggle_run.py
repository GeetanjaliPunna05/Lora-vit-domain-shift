"""
kaggle_run.py - run vit_dg.py experiments on Kaggle GPUs from this machine.

  python kaggle_run.py push --dataset pacs --model vit_b --grid pacs     # start a session
  python kaggle_run.py status                                            # running / complete / error
  python kaggle_run.py pull                                              # merge results into results/results.jsonl

Each push embeds vit_dg.py plus every result already in results/results.jsonl, so a new
session skips finished runs: push -> wait -> pull -> push again until the grid is done.
With 2 GPUs (T4 x2) the job list is split into two shards that run in parallel.
Requires a Kaggle API token (Kaggle -> Settings -> API -> Create New Token) and a
phone-verified Kaggle account (needed for GPU + internet access).
"""
import argparse
import base64
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "kaggle_build")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
SLUG = "vit-dg-runner-v2"

KERNEL_TEMPLATE = r'''
import base64, os, subprocess, sys, torch
# Kaggle's preinstalled torchao is too old for current peft, which then refuses plain LoRA; we don't use it
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", {pip}], check=True)
os.makedirs("/kaggle/working/results", exist_ok=True)
open("/kaggle/working/vit_dg.py", "wb").write(base64.b64decode({code!r}))
open("/kaggle/working/results/results.jsonl", "wb").write(base64.b64decode({prev!r}))
os.chdir("/kaggle/working")
import json, time
n = max(1, torch.cuda.device_count())
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
start, budget = time.time(), {hours}

def merge():
    seen, lines = set(), []
    for f in ["results/results.jsonl"] + [f"results/shard{{i}}.jsonl" for i in range(n)]:
        if os.path.exists(f):
            for l in open(f):
                if l.strip() and (rid := json.loads(l)["run_id"]) not in seen:
                    seen.add(rid); lines.append(l if l.endswith("\n") else l + "\n")
    open("results/results.jsonl", "w").writelines(lines)
    return len(lines)

# phases run in order; each one gets whatever time budget is left
for k, args in enumerate({phases}):
    left = budget - (time.time() - start) / 3600
    if left < 0.15:
        print(f"[phase {{k}}] skipped, out of time", flush=True); continue
    ds = args[args.index("--dataset") + 1]
    print(f"[phase {{k}}] {{' '.join(args)}} | {{left:.2f}} h left", flush=True)
    procs = []
    for i in range(n):   # each shard starts from all results so far, so finished runs are skipped
        open(f"results/shard{{i}}.jsonl", "wb").write(open("results/results.jsonl", "rb").read())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_PROGRESS_BARS="1")
        cmd = [sys.executable, "vit_dg.py", *args, "--hours", f"{{left:.3f}}", "--shard", f"{{i}}/{{n}}",
               "--cache_dir", "/tmp/cache", "--out", f"results/shard{{i}}.jsonl",
               "--workers", str(max(1, (os.cpu_count() or 2) // n))]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(f"log_p{{k}}_gpu{{i}}.txt", "w"), stderr=subprocess.STDOUT))
        if i == 0 and n > 1:   # let shard 0 build this dataset's cache first so shards don't race on it
            while procs[0].poll() is None and not os.path.exists(f"/tmp/cache/{{ds}}_256_meta.npz"):
                time.sleep(10)
    codes = [p.wait() for p in procs]
    total = merge()
    for i in range(n):
        print(f"== phase {{k}} gpu {{i}} (tail) ==\n" + "".join(open(f"log_p{{k}}_gpu{{i}}.txt").readlines()[-12:]), flush=True)
    print(f"[phase {{k}}] exit codes {{codes}} | total results {{total}} | {{(time.time() - start) / 3600:.2f}} h elapsed", flush=True)
'''

# Time-boxed plan (~5.5 GPU-hours): short LR tuning on one PACS split, then the core grids in priority
# order. Within a phase jobs are seed-major, so a phase cut short still has whole seeds.
EPOCHS = ["--epochs", "4"]
PLANS = {
    "deadline": [
        ["--dataset", "pacs", "--model", "vit_b", "--grid", "tune", "--lrs", "1e-3", "3e-3", "1e-2",
         "--seeds", "0", "--targets", "art_painting", *EPOCHS],
        ["--dataset", "pacs", "--model", "vit_b", "--grid", "pacs_core", "--seeds", "0", "--tuned_from", "pacs", *EPOCHS],
        ["--dataset", "officehome", "--model", "vit_b", "--grid", "officehome_core", "--seeds", "0", "--tuned_from", "pacs", *EPOCHS],
        ["--dataset", "pacs", "--model", "vit_b", "--grid", "pacs_core", "--seeds", "1", "2", "--tuned_from", "pacs", *EPOCHS],
    ],
    # PACS only (Office-Home runs on the local GPU)
    "pacs": [
        ["--dataset", "pacs", "--model", "vit_b", "--grid", "tune", "--lrs", "1e-3", "3e-3", "1e-2",
         "--seeds", "0", "--targets", "art_painting", *EPOCHS],
        ["--dataset", "pacs", "--model", "vit_b", "--grid", "pacs_core", "--seeds", "0", "1", "2", "--tuned_from", "pacs", *EPOCHS],
    ],
}

PIP = ["transformers==5.23.0", "peft==0.21.3", "accelerate", "bitsandbytes", "datasets", "scikit-learn"]


def kaggle(*a, capture=False):
    # via python, not the kaggle.exe launcher (Smart App Control blocks that launcher on this PC)
    cmd = [sys.executable, "-c", "import sys, kaggle.cli as c; sys.argv[0] = 'kaggle'; c.main()", *a]
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def username():
    out = kaggle("config", "view", capture=True).stdout
    for line in out.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    sys.exit("Could not read Kaggle username; is the API token installed?")


def push(args, extra):
    user = username()
    code = base64.b64encode(open(os.path.join(ROOT, "vit_dg.py"), "rb").read()).decode()
    prev = base64.b64encode(open(RESULTS, "rb").read() if os.path.exists(RESULTS) else b"").decode()
    phases = PLANS[args.plan] if args.plan else [
        ["--dataset", args.dataset, "--model", args.model, "--grid", args.grid, *extra]]
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, "run.py"), "w") as f:
        f.write(KERNEL_TEMPLATE.format(pip=", ".join(map(repr, PIP)), code=code, prev=prev,
                                       phases=phases, hours=args.hours))
    run_args = phases
    meta = {"id": f"{user}/{SLUG}", "title": SLUG, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": args.accelerator, "dataset_sources": [], "kernel_sources": [],
            "competition_sources": []}
    with open(os.path.join(BUILD, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    for k, p in enumerate(run_args):
        print(f"phase {k}:", " ".join(p))
    kaggle("kernels", "push", "-p", BUILD)


def pull():
    user = username()
    out = os.path.join(BUILD, "output")
    os.makedirs(out, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{SLUG}", "-p", out, "--force")
    new = os.path.join(out, "results", "results.jsonl")
    if not os.path.exists(new):
        sys.exit("No results.jsonl in the kernel output yet.")
    seen, lines = set(), []
    for f in [RESULTS, new]:
        if os.path.exists(f):
            for l in open(f):
                if l.strip() and (rid := json.loads(l)["run_id"]) not in seen:
                    seen.add(rid)
                    lines.append(l if l.endswith("\n") else l + "\n")
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        f.writelines(lines)
    print(f"{len(lines)} results in {RESULTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status", "pull"])
    ap.add_argument("--dataset", default="pacs")
    ap.add_argument("--model", default="vit_b")
    ap.add_argument("--grid", default="pacs")
    ap.add_argument("--hours", type=float, default=11.0, help="stay under Kaggle's 12h session limit")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    ap.add_argument("--plan", default=None, choices=list(PLANS), help="run a multi-phase plan instead")
    args, extra = ap.parse_known_args()   # unknown args are forwarded to vit_dg.py
    if args.cmd == "push":
        push(args, extra)
    elif args.cmd == "status":
        kaggle("kernels", "status", f"{username()}/{SLUG}")
    else:
        pull()


if __name__ == "__main__":
    main()
