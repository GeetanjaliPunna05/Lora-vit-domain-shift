"""make_tables.py - LaTeX results tables from results/*_summary.csv (written by analyze.py) -> paper/tables.tex"""
import pandas as pd

ROWS = ["linear", "bitfit", "r1", "r4", "r16", "early", "middle", "late", "inc", "dec", "attn10", "mlp4", "qlora_r4"]
NAME = {"linear": "Linear probe", "bitfit": "BitFit", "r1": "LoRA all, $r$=1", "r4": "LoRA all, $r$=4",
        "r16": "LoRA all, $r$=16", "early": "Early (0--3), $r$=12", "middle": "Middle (4--7), $r$=12",
        "late": "Late (8--11), $r$=12", "inc": "Increasing 1/4/7", "dec": "Decreasing 7/4/1",
        "attn10": "Attention $q,v$, $r$=10", "mlp4": "MLP $fc_1,fc_2$, $r$=4", "qlora_r4": "QLoRA $r$=4 (NF4)"}
GROUP_BREAK = {"r1", "early", "inc", "attn10", "qlora_r4"}


def k(n):
    return "0" if n == 0 else f"{n / 1000:.0f}k" if n >= 1000 else str(n)


def main_table(name, title, label, domains):
    s = pd.read_csv(f"results/{name}_summary.csv").set_index("config")
    best = s.avg_acc.max()
    cols = "l" + "r" * (len(domains) + 6)
    head = " & ".join(["Method"] + [d for _, d in domains] + ["Avg", "Worst", "F1", "Params", "Mem", "Time"])
    out = [r"\begin{table*}[t]", r"\centering\small", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
           head + r" \\", r"\midrule"]
    for c in ROWS:
        if c not in s.index:
            continue
        r = s.loc[c]
        if c in GROUP_BREAK:
            out.append(r"\midrule")
        avg = f"{r.avg_acc:.1f}" + (f"$\\pm${r.std_seeds:.1f}" if pd.notna(r.std_seeds) else "")
        if abs(r.avg_acc - best) < 1e-9:
            avg = rf"\textbf{{{avg}}}"
        cells = [NAME[c]] + [f"{r[f'acc_{d}']:.1f}" for d, _ in domains] + [
            avg, f"{r.worst_acc:.1f}", f"{r.avg_f1:.1f}", k(int(r.adapter_params)),
            f"{r.peak_mem_mb / 1024:.2f}", f"{r.train_time_s:.0f}"]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            rf"\caption{{{title} Target-domain accuracy (\%) per held-out domain, mean over 3 seeds; Avg $\pm$ sd over "
            r"seeds of the domain-averaged accuracy; Worst = lowest held-out domain; F1 = macro-F1 averaged over domains; "
            r"Params = trainable parameters excluding the head; Mem = peak training memory (GB); Time = training "
            rf"seconds per run.}}", rf"\label{{{label}}}", r"\end{table*}", ""]
    return "\n".join(out)


def tests_table():
    out = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Comparison (A vs B) & \multicolumn{2}{c}{PACS} & \multicolumn{2}{c}{Office-Home} \\",
           r" & $\Delta$ & $p$ & $\Delta$ & $p$ \\", r"\midrule"]
    p = pd.read_csv("results/pacs_vit_b_e4_paired_tests.csv")
    o = pd.read_csv("results/officehome_vit_b_e4_paired_tests.csv")
    m = p.merge(o, on=["A", "B"], how="left", suffixes=("_p", "_o"))
    for _, r in m.iterrows():
        def f(d, pv):
            return ("--", "--") if pd.isna(d) else (f"{d:+.1f}", f"{pv:.3f}")
        dp, pp = f(r.mean_diff_pts_p, r.p_value_p)
        do, po = f(r.mean_diff_pts_o, r.p_value_o)
        name = f"{NAME[r.A]} vs {NAME[r.B]}"
        out.append(f"{name} & {dp} & {pp} & {do} & {po} " + r"\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Paired comparisons: mean accuracy difference A$-$B in points over the 12 matched (target domain, "
            r"seed) runs, and two-sided paired $t$-test $p$-value.}", r"\label{tab:tests}", r"\end{table}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    pacs = [("art_painting", "Art"), ("cartoon", "Cart."), ("photo", "Photo"), ("sketch", "Sketch")]
    oh = [("Art", "Art"), ("Clipart", "Clip."), ("Product", "Prod."), ("Real World", "Real")]
    tex = main_table("pacs_vit_b_e4", "PACS, ViT-B/16, Tesla T4.", "tab:pacs", pacs)
    tex += main_table("officehome_vit_b_e4", "Office-Home, ViT-B/16, RX 9070 GRE.", "tab:oh", oh)
    tex += tests_table()
    open("paper/tables.tex", "w").write(tex)
    print("wrote paper/tables.tex")
