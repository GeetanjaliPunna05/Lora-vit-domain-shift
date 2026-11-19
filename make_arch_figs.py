"""make_arch_figs.py - draws fig1_base_paper_architecture.png and fig2_project_architecture.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
FROZEN, FROZEN_EDGE = "#ecebe7", "#b9b7b0"            # neutral = frozen
TRAIN, TRAIN_EDGE = "#cde2fb", "#2a78d6"               # blue = trainable (LoRA / head)
QUANT, QUANT_EDGE = "#fbe0d4", "#eb6834"               # orange = 4-bit storage
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE})


def box(ax, x, y, w, h, text, fc=FROZEN, ec=FROZEN_EDGE, fs=9, bold=False, color=INK, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11, color=color, lw=lw))


def fig1():
    """Base papers: ViT encoder (Dosovitskiy et al.) + LoRA update W0 + BA (Hu et al.) + QLoRA NF4 storage."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 10.4)
    ax.axis("off")

    # (a) ViT
    ax.text(0.2, 9.9, "(a) ViT-B/16", fontsize=11, color=INK, fontweight="bold")
    ax.text(0.2, 9.45, "(Dosovitskiy et al., 2021)", fontsize=8.5, color=INK2)
    for i in range(4):
        box(ax, 0.3 + i * 0.75, 0.6, 0.6, 0.6, "", fc="#d8d6d0", ec=FROZEN_EDGE)
    ax.text(1.8, 0.2, "image -> 16x16 patches", ha="center", fontsize=8, color=INK2)
    box(ax, 0.3, 1.8, 3.0, 0.8, "Linear patch embedding\n+ [CLS] + position emb.", fs=8)
    arrow(ax, 1.8, 1.25, 1.8, 1.8)
    box(ax, 0.3, 3.2, 3.0, 4.3, "", fc=SURFACE, ec=FROZEN_EDGE, ls="--")
    # pre-norm block, read bottom to top (residual connections omitted)
    box(ax, 0.55, 3.4, 2.5, 0.55, "LayerNorm", fs=8)
    box(ax, 0.55, 4.15, 2.5, 0.85, "Multi-head self-attention\n(q, k, v, o projections)", fs=8)
    box(ax, 0.55, 5.2, 2.5, 0.55, "LayerNorm", fs=8)
    box(ax, 0.55, 5.95, 2.5, 0.85, "MLP (fc1 768->3072,\nGELU, fc2 3072->768)", fs=8)
    ax.text(1.8, 7.1, "+ residual connections", ha="center", fontsize=7, color=INK2)
    ax.text(3.45, 5.3, "x 12\nblocks", fontsize=9, color=INK2, va="center")
    arrow(ax, 1.8, 2.6, 1.8, 3.2)
    box(ax, 0.3, 8.0, 3.0, 0.75, "MLP head -> class", fs=8.5)
    arrow(ax, 1.8, 7.5, 1.8, 8.0)

    # (b) LoRA
    ax.text(5.6, 9.9, "(b) LoRA (Hu et al., 2021)", fontsize=11, color=INK, fontweight="bold")
    box(ax, 6.0, 4.0, 2.6, 2.6, "Pretrained\nweight\nW0 (d x k)\nfrozen", fs=9)
    box(ax, 9.4, 5.6, 2.4, 0.9, "A (r x k)\ninit Gaussian", fc=TRAIN, ec=TRAIN_EDGE, fs=8.5)
    box(ax, 9.4, 4.1, 2.4, 0.9, "B (d x r)\ninit 0", fc=TRAIN, ec=TRAIN_EDGE, fs=8.5)
    ax.text(10.6, 5.27, "rank r << min(d, k)", ha="center", fontsize=7.5, color=TRAIN_EDGE)
    box(ax, 7.9, 1.2, 2.2, 0.8, "input x", fc=SURFACE, fs=9)
    arrow(ax, 8.4, 2.0, 7.3, 4.0)
    arrow(ax, 9.6, 2.0, 10.6, 4.1)
    ax.add_patch(plt.Circle((9.0, 8.0), 0.35, fc=SURFACE, ec=INK2, lw=1.2))
    ax.text(9.0, 8.0, "+", ha="center", va="center", fontsize=13, color=INK)
    arrow(ax, 7.3, 6.6, 8.75, 7.8)
    arrow(ax, 10.6, 6.5, 9.25, 7.8)
    ax.text(9.0, 9.0, "h = W0 x + (alpha / r) B A x", ha="center", fontsize=10, color=INK)
    ax.text(9.0, 0.35, "only A and B are trained", ha="center", fontsize=8.5, color=TRAIN_EDGE)

    # (c) QLoRA
    ax.text(13.7, 9.9, "(c) QLoRA (Dettmers et al., 2023)", fontsize=11, color=INK, fontweight="bold")
    box(ax, 14.0, 6.2, 3.4, 1.5, "W0 stored in 4-bit NF4\n(+ double-quantized\nscales)", fc=QUANT, ec=QUANT_EDGE, fs=8.5)
    box(ax, 14.0, 3.9, 3.4, 1.2, "dequantize to 16-bit\nat compute time", fs=8.5)
    arrow(ax, 15.7, 6.2, 15.7, 5.1)
    box(ax, 18.2, 3.9, 3.4, 1.2, "LoRA A, B in 16/32-bit\n(trainable)", fc=TRAIN, ec=TRAIN_EDGE, fs=8.5)
    box(ax, 15.0, 1.6, 5.6, 1.0, "h = deq(W0_NF4) x + (alpha / r) B A x", fc=SURFACE, fs=8.5)
    arrow(ax, 15.7, 3.9, 16.8, 2.6)
    arrow(ax, 19.9, 3.9, 18.8, 2.6)
    ax.text(17.8, 8.6, "4x smaller frozen weights;\ngradients flow through deq(W0)", ha="center", fontsize=8.5, color=INK2)

    for x in (5.2, 13.2):
        ax.plot([x, x], [0.3, 10.1], color="#e4e3df", lw=1)
    leg = [(FROZEN, FROZEN_EDGE, "frozen"), (TRAIN, TRAIN_EDGE, "trainable"), (QUANT, QUANT_EDGE, "4-bit storage")]
    for i, (fc, ec, t) in enumerate(leg):
        box(ax, 14.0 + i * 2.6, 0.1, 0.45, 0.35, "", fc=fc, ec=ec)
        ax.text(14.55 + i * 2.6, 0.27, t, fontsize=8, color=INK2, va="center")
    fig.tight_layout()
    fig.savefig("figures/fig1_base_paper_architecture.png", dpi=220)
    plt.close(fig)


def fig2():
    """This project: frozen ViT-B with LoRA placement / rank / module / 4-bit study under LODO."""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12.4)
    ax.axis("off")
    ax.text(0.2, 11.9, "Project: where does a frozen ViT need adaptation capacity under domain shift?",
            fontsize=12, color=INK, fontweight="bold")

    # LODO protocol
    ax.text(0.3, 10.9, "Leave-one-domain-out", fontsize=10, color=INK, fontweight="bold")
    doms = ["Source domain 1", "Source domain 2", "Source domain 3", "Target domain"]
    for i, d in enumerate(doms):
        tgt = i == 3
        box(ax, 0.3, 9.4 - i * 1.15, 3.6, 0.85, d + ("\n(held-out target: test only)" if tgt else "\n(source: 90% train / 10% val)"),
            fc=QUANT if tgt else FROZEN, ec=QUANT_EDGE if tgt else FROZEN_EDGE, fs=7.5)
    ax.text(2.1, 4.6, "PACS: Photo, Art, Cartoon, Sketch\nOffice-Home: Art, Clipart, Product,\nReal World. Each domain is the target\n"
            "once, 3 seeds; LR and epoch picked\non source validation only", ha="center", va="center",
            fontsize=7.2, color=INK2, linespacing=1.35)
    arrow(ax, 4.0, 8.2, 5.2, 8.2)

    # ViT backbone with three groups
    ax.text(5.4, 10.9, "Frozen ViT-B/16 (ImageNet-21k): 16-bit or 4-bit NF4 weights", fontsize=10, color=INK, fontweight="bold")
    groups = [("Early\nblocks 0-3", 5.4), ("Middle\nblocks 4-7", 9.4), ("Late\nblocks 8-11", 13.4)]
    for name, x0 in groups:
        box(ax, x0, 5.6, 3.7, 4.9, "", fc=SURFACE, ec=FROZEN_EDGE, ls="--")
        ax.text(x0 + 1.85, 10.05, name, ha="center", va="center", fontsize=8.5, color=INK2)
        box(ax, x0 + 0.25, 7.9, 3.2, 1.3, "Self-attention\nq  k  v  o", fs=8.5)
        box(ax, x0 + 0.25, 6.0, 3.2, 1.3, "MLP\nfc1   fc2", fs=8.5)
        for xx in (x0 + 0.55, x0 + 1.85):        # LoRA on q, v
            box(ax, xx - 0.35, 7.45, 0.7, 0.38, "LoRA", fc=TRAIN, ec=TRAIN_EDGE, fs=6.5)
        for xx in (x0 + 0.95, x0 + 2.75):        # LoRA on fc1, fc2
            box(ax, xx - 0.35, 5.62, 0.7, 0.36, "LoRA", fc=TRAIN, ec=TRAIN_EDGE, fs=6.5, ls="--")
    arrow(ax, 9.1, 8.2, 9.4, 8.2)
    arrow(ax, 13.1, 8.2, 13.4, 8.2)
    box(ax, 17.6, 7.6, 2.4, 1.2, "[CLS] ->\nlinear head\n(trainable)", fc=TRAIN, ec=TRAIN_EDGE, fs=8)
    arrow(ax, 17.1, 8.2, 17.6, 8.2)
    box(ax, 20.5, 7.6, 3.2, 1.2, "accuracy, macro-F1,\nworst domain, ECE,\nparams, peak mem, time", fc=SURFACE, fs=7.5)
    arrow(ax, 20.0, 8.2, 20.5, 8.2)

    # research questions
    rq = [("RQ1 placement", "same budget (147k):\nr12 in early / middle / late\nvs r4 in all blocks"),
          ("RQ2 rank", "uniform r = 1, 4, 16;\nlayer-wise increasing 1/4/7\nvs decreasing 7/4/1"),
          ("RQ3 module", "same budget (369k):\nattention q,v r10 vs\nMLP r4 (dashed LoRA)"),
          ("RQ4 4-bit", "QLoRA: NF4 vs 16-bit\nbackbone; accuracy vs\nmemory and time"),
          ("RQ5 transfer", "repeat core grid on\nOffice-Home (65 classes)")]
    for i, (h, t) in enumerate(rq):
        x0 = 5.4 + i * 3.7
        box(ax, x0, 1.0, 3.5, 2.6, "", fc=SURFACE, ec=FROZEN_EDGE)
        ax.text(x0 + 1.75, 3.15, h, ha="center", fontsize=8.5, color=TRAIN_EDGE, fontweight="bold")
        ax.text(x0 + 1.75, 2.0, t, ha="center", va="center", fontsize=7.2, color=INK2, linespacing=1.35)
    box(ax, 0.3, 1.0, 4.7, 2.6, "Baselines\nlinear probe (head only)\nBitFit (biases + head)", fc=FROZEN, fs=8)

    leg = [(FROZEN, FROZEN_EDGE, "frozen"), (TRAIN, TRAIN_EDGE, "trainable"), (QUANT, QUANT_EDGE, "held-out target")]
    for i, (fc, ec, t) in enumerate(leg):
        box(ax, 0.3 + i * 2.8, 0.15, 0.45, 0.35, "", fc=fc, ec=ec)
        ax.text(0.85 + i * 2.8, 0.32, t, fontsize=8, color=INK2, va="center")
    fig.tight_layout()
    fig.savefig("figures/fig2_project_architecture.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    fig1()
    fig2()
    print("wrote figures/fig1_base_paper_architecture.png, figures/fig2_project_architecture.png")
