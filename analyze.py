"""
analyze.py - tables and figures from results/results.jsonl.

  python analyze.py --dataset pacs --model vit_b --epochs 10

Writes results/<dataset>_<model>_summary.csv / .md and figures/<dataset>_*.png.
Per config: mean over seeds of the per-target accuracy, then average / worst over targets;
the +- is the std over seeds of the target-averaged accuracy. If a config was run with
several learning rates, the LR with the best mean *source-validation* accuracy is kept
(target accuracy is never used for selection).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ORDER = ["linear", "bitfit", "r1", "r2", "r4", "r8", "r16", "early", "middle", "late", "inc", "dec",
         "midheavy", "attn10", "mlp4", "both", "qlora_r4", "qlora_both"]
LABEL = {"linear": "Linear probe", "bitfit": "BitFit", "r1": "All blocks r1", "r2": "All blocks r2",
         "r4": "All blocks r4", "r8": "All blocks r8", "r16": "All blocks r16", "early": "Early (0-3) r12",
         "middle": "Middle (4-7) r12", "late": "Late (8-11) r12", "inc": "Increasing 1/4/7",
         "dec": "Decreasing 7/4/1", "midheavy": "Middle-heavy 2/8/2", "attn10": "Attention q,v",
         "mlp4": "MLP fc1,fc2", "both": "Attention + MLP", "qlora_r4": "QLoRA r4", "qlora_both": "QLoRA attn+MLP"}
# reference palette (dataviz skill): categorical slots 1-2, ink, muted, grid, sequential blue ramp
BLUE, ORANGE, INK, INK2, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load(paths, dataset, model, epochs):
    rows = [json.loads(l) for p in paths if os.path.exists(p) for l in open(p) if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates("run_id")
    df = df[(df.dataset == dataset) & (df.model == model) & (df.epochs == epochs)]
    df = df[~df.run_id.str.contains("_lim")]
    # The LR of each config was fixed by source-validation tuning before the main grid ran; the
    # main-grid LR is the one with the most runs. Tuning runs at other LRs are dropped here.
    n = df.groupby(["config", "lr"]).size().reset_index(name="n")
    best = n.loc[n.groupby("config").n.idxmax(), ["config", "lr"]]
    return df.merge(best, on=["config", "lr"])


def summarize(df):
    out = []
    main_device = df.device.mode().iloc[0] if "device" in df else None
    for c, g in df.groupby("config"):
        # memory / time only from the dataset's main GPU, so they stay comparable across configs
        hw = g[g.device == main_device] if "device" in g else g
        per_t = g.groupby("target").test_acc.mean()
        per_seed = g.groupby("seed").test_acc.mean()          # target-average per seed
        r = {"config": c, "label": LABEL.get(c, c), "lr": g.lr.iloc[0],
             "avg_acc": per_t.mean() * 100, "std_seeds": per_seed.std(ddof=1) * 100 if len(per_seed) > 1 else np.nan,
             "worst_acc": per_t.min() * 100, "worst_domain": per_t.idxmin(),
             "avg_f1": g.groupby("target").test_f1.mean().mean() * 100,
             "avg_ece": g.groupby("target").test_ece.mean().mean() * 100,
             "trainable_params": int(g.trainable_params.iloc[0]), "adapter_params": int(g.adapter_params.iloc[0]),
             "peak_mem_mb": hw.peak_mem_mb.mean(), "weights_mem_mb": hw.weights_mem_mb.mean(),
             "train_time_s": hw.train_time_s.mean(), "n_runs": len(g), "complete": g.seed.nunique() * g.target.nunique() == len(g)}
        for t, v in per_t.items():
            r[f"acc_{t}"] = v * 100
        out.append(r)
    s = pd.DataFrame(out)
    s["_o"] = s.config.map({c: i for i, c in enumerate(ORDER)}).fillna(99)
    return s.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def dot_panel(ax, s, configs, title, ref=None):
    sub = s.set_index("config").reindex([c for c in configs if c in set(s.config)])
    y = np.arange(len(sub))[::-1]
    ax.errorbar(sub.avg_acc, y, xerr=sub.std_seeds.fillna(0), fmt="o", color=BLUE, ecolor=BLUE,
                elinewidth=2, capsize=0, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for yi, v in zip(y, sub.avg_acc):
        ax.annotate(f"{v:.1f}", (v, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    if ref is not None and ref in set(s.config):
        v = s.set_index("config").avg_acc[ref]
        ax.axvline(v, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        title += f"   - - {LABEL[ref]} = {v:.1f}"
    ax.set_yticks(y, [LABEL.get(c, c) for c in sub.index])
    ax.set_ylim(-0.6, len(sub) - 0.2)
    ax.margins(x=0.08)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("Avg. target-domain accuracy (%)", fontsize=9, color=INK2)
    style(ax)


def figures(s, df, prefix, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE})
    have = set(s.config)

    # 1. rank sweep
    ranks = [(int(c[1:]), c) for c in ["r1", "r2", "r4", "r8", "r16"] if c in have]
    if len(ranks) > 1:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        sub = s.set_index("config")
        x = [r for r, _ in ranks]
        ax.errorbar(x, [sub.avg_acc[c] for _, c in ranks], yerr=[sub.std_seeds.fillna(0)[c] for _, c in ranks],
                    color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                    elinewidth=1.5, capsize=0, label="LoRA, all blocks, q,v")
        for ref in ("bitfit",):   # linear probe is far below; it is in the table instead
            if ref in have:
                ax.axhline(sub.avg_acc[ref], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
                ax.annotate(f"{LABEL[ref]} {sub.avg_acc[ref]:.1f}", (0.01, sub.avg_acc[ref]), xytext=(0, 3),
                            xycoords=("axes fraction", "data"), textcoords="offset points", ha="left",
                            fontsize=8, color=MUTED, backgroundcolor=SURFACE)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(r) for r in x])
        ax.set_xlabel("LoRA rank r", fontsize=9, color=INK2)
        ax.set_ylabel("Avg. target-domain accuracy (%)", fontsize=9, color=INK2)
        ax.set_title(f"{dataset}: accuracy vs. LoRA rank", loc="left", fontsize=10, color=INK)
        style(ax)
        fig.tight_layout()
        fig.savefig(f"{prefix}_rank.png", dpi=200)
        plt.close(fig)

    # 2. placement / layer-wise rank / module type, all budget-matched within a panel
    panels = [(["early", "middle", "late"], "RQ1 placement (147k params)", "r4"),
              (["inc", "dec", "midheavy"], "RQ2 layer-wise rank (147k params)", "r4"),
              (["attn10", "mlp4", "both"], "RQ3 module type (369k params)", None)]
    panels = [p for p in panels if have & set(p[0])]
    if panels:
        fig, axes = plt.subplots(len(panels), 1, figsize=(5.6, 1.9 * len(panels) + 0.4), squeeze=False)
        for ax, (cfgs, title, ref) in zip(axes[:, 0], panels):
            dot_panel(ax, s, cfgs, title, ref)
        fig.tight_layout()
        fig.savefig(f"{prefix}_placement_rank_module.png", dpi=200)
        plt.close(fig)

    # 3. QLoRA trade-off: accuracy (left) and memory from zero (right), 16-bit vs 4-bit backbone
    if {"r4", "qlora_r4"} <= have:
        sub = s.set_index("config")
        cs, cols, names = ["r4", "qlora_r4"], [BLUE, ORANGE], ["16-bit backbone", "4-bit NF4 backbone"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1, 1.5]})
        for i, c in enumerate(cs):
            a1.errorbar(i, sub.avg_acc[c], yerr=sub.std_seeds.fillna(0)[c], fmt="o", color=cols[i], elinewidth=2,
                        capsize=0, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
            a1.annotate(f"{sub.avg_acc[c]:.1f}", (i, sub.avg_acc[c]), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        lo = min(sub.avg_acc[c] - sub.std_seeds.fillna(0)[c] for c in cs)
        hi = max(sub.avg_acc[c] + sub.std_seeds.fillna(0)[c] for c in cs)
        a1.set_ylim(lo - 1.5, hi + 1.5)
        a1.set_xlim(-0.6, 1.6)
        a1.set_xticks([0, 1], ["16-bit", "4-bit NF4"])
        a1.set_ylabel("Avg. target-domain acc. (%), +-1 sd", fontsize=8.5, color=INK2)
        a1.set_title("Accuracy (LoRA r4)", loc="left", fontsize=10, color=INK)
        style(a1)
        y = np.arange(2)
        for i, c in enumerate(cs):
            a2.barh(y[0] + (0.18 if i == 0 else -0.18), sub.weights_mem_mb[c], height=0.34, color=cols[i],
                    edgecolor=SURFACE, linewidth=2, label=names[i])
            a2.barh(y[1] + (0.18 if i == 0 else -0.18), sub.peak_mem_mb[c], height=0.34, color=cols[i],
                    edgecolor=SURFACE, linewidth=2)
            for yy, v in [(y[0], sub.weights_mem_mb[c]), (y[1], sub.peak_mem_mb[c])]:
                a2.annotate(f"{v:.0f} MB", (v, yy + (0.18 if i == 0 else -0.18)), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        a2.set_yticks(y, ["Model weights\non GPU", "Peak during\ntraining"])
        a2.invert_yaxis()
        a2.margins(x=0.22)
        a2.set_xlabel("GPU memory (MB)", fontsize=9, color=INK2)
        a2.set_title("Memory", loc="left", fontsize=10, color=INK)
        a2.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
        style(a2)
        a2.grid(axis="y", visible=False)
        fig.suptitle(f"{dataset}: 4-bit backbone (QLoRA) vs 16-bit", x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout()
        fig.savefig(f"{prefix}_qlora.png", dpi=200)
        plt.close(fig)

    # 4. per-domain heatmap (configs x held-out domain)
    tcols = [c for c in s.columns if c.startswith("acc_")]
    if tcols:
        from matplotlib.colors import LinearSegmentedColormap
        M = s[tcols].to_numpy()
        # shade within each column (domains differ a lot in difficulty); print the raw accuracy
        lo, hi = np.nanmin(M, axis=0), np.nanmax(M, axis=0)
        Z = (M - lo) / np.maximum(hi - lo, 1e-9)
        fig, ax = plt.subplots(figsize=(1.3 * len(tcols) + 2.6, 0.32 * len(s) + 1.2))
        ax.imshow(Z, cmap=LinearSegmentedColormap.from_list("seq", SEQ), aspect="auto", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    dark = Z[i, j] > 0.55
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if dark else INK)
        ax.set_xticks(range(len(tcols)), [c[4:] for c in tcols], fontsize=8, color=INK2)
        ax.set_yticks(range(len(s)), s.label, fontsize=8, color=INK2)
        ax.set_title(f"{dataset}: accuracy per held-out domain (%)\nshade = worst to best within each domain",
                     loc="left", fontsize=10, color=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{prefix}_per_domain.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results/results.jsonl", "results/local.jsonl"])
    ap.add_argument("--dataset", default="pacs")
    ap.add_argument("--model", default="vit_b")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no_figs", action="store_true")
    ap.add_argument("--tune", action="store_true", help="print mean source-val accuracy per config x LR")
    args = ap.parse_args()
    if args.tune:
        rows = pd.DataFrame([json.loads(l) for p in args.results if os.path.exists(p) for l in open(p) if l.strip()])
        rows = rows[(rows.dataset == args.dataset) & (rows.model == args.model) & (rows.epochs == args.epochs)]
        print((rows.pivot_table(index="config", columns="lr", values="val_acc", aggfunc="mean") * 100).round(2))
        return
    df = load(args.results, args.dataset, args.model, args.epochs)
    if df.empty:
        raise SystemExit("no matching results")
    s = summarize(df)
    base = f"{args.dataset}_{args.model}_e{args.epochs}"
    os.makedirs("results", exist_ok=True)
    s.to_csv(f"results/{base}_summary.csv", index=False)
    cols = ["label", "avg_acc", "std_seeds", "worst_acc", "worst_domain", "avg_f1", "adapter_params",
            "peak_mem_mb", "train_time_s", "n_runs"]
    md = s[cols].to_markdown(index=False, floatfmt=".1f") if _has_tabulate() else s[cols].to_string(index=False)
    open(f"results/{base}_summary.md", "w").write(md + "\n")
    print(md)
    comps = paired_tests(df)
    comps.to_csv(f"results/{base}_paired_tests.csv", index=False)
    print("\nPaired comparisons over (target, seed):")
    print(comps.to_string(index=False))
    if not args.no_figs:
        os.makedirs("figures", exist_ok=True)
        figures(s, df, f"figures/{base}", {"pacs": "PACS", "officehome": "Office-Home"}.get(args.dataset, args.dataset))
        print("figures written to figures/")


COMPARISONS = [("early", "r4"), ("middle", "r4"), ("late", "r4"), ("early", "late"), ("middle", "late"),
               ("early", "middle"), ("r1", "r4"), ("r16", "r4"), ("inc", "r4"), ("dec", "r4"), ("inc", "dec"),
               ("mlp4", "attn10"), ("qlora_r4", "r4"), ("r4", "bitfit"), ("r4", "linear"), ("bitfit", "linear")]


def paired_tests(df):
    """A - B accuracy difference, paired over identical (target, seed) runs; two-sided paired t-test."""
    from scipy.stats import ttest_rel
    acc = df.pivot_table(index=["target", "seed"], columns="config", values="test_acc")
    rows = []
    for a, b in COMPARISONS:
        if a in acc and b in acc:
            pair = acc[[a, b]].dropna()
            d = (pair[a] - pair[b]) * 100
            p = ttest_rel(pair[a], pair[b]).pvalue if len(pair) > 1 else np.nan
            rows.append({"A": a, "B": b, "mean_diff_pts": round(d.mean(), 2), "n_pairs": len(pair),
                         "A_wins": int((d > 0).sum()), "p_value": round(p, 4)})
    return pd.DataFrame(rows)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
