#!/bin/sh
# Office-Home on the local RX 9070 GRE: LR tuning on one split, then the core grid (3 seeds).
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
PY=./.venv-rocm/Scripts/python.exe
$PY vit_dg.py --dataset officehome --model vit_b --grid tune --lrs 1e-3 3e-3 1e-2 --seeds 0 --targets Art \
    --epochs 4 --workers 6 --out results/local.jsonl
$PY vit_dg.py --dataset officehome --model vit_b --grid officehome_core --seeds 0 1 2 --tuned_from officehome \
    --epochs 4 --workers 6 --out results/local.jsonl
echo "ALL DONE"
