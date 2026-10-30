"""Static figures for the manuscript (PNG, 300 dpi, sized for an IEEE column).

usage: python scripts/figures.py --results results --runs_dir results/runs
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.io import load_predictions  # noqa: E402
from rexerr.labels import NATIVE, NATIVE_DISPLAY  # noqa: E402
from rexerr.metrics import any_error_score, prevalence_analysis  # noqa: E402

# Reference categorical palette (light mode), fixed order; neutrals for text and context.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#b8b7b0", "#e6e5e0"
COL_W = 3.5  # inches, one IEEE column

plt.rcParams.update({
    "font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
    "savefig.bbox": "tight", "text.color": INK, "axes.axisbelow": True,
})


def label_agreement(audit, out):
    per = audit["labels"]["test"]["per_type"]
    rows = [(NATIVE_DISPLAY[k], v["sampled_confirmed_by_sentence"] or 0.0) for k, v in per.items()]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.55, color=SERIES[0])
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", color=INK_2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Sampled errors confirmed by sentence-level annotation (test split)")
    ax.grid(axis="y", visible=False)
    fig.savefig(out)
    plt.close(fig)


def per_type_f1(summary, out, other_key=None):
    runs = summary["runs"]
    main = summary["main_run"]
    keys = [main] + ([other_key] if other_key else [])
    per = {k: {c["class"]: c for c in runs[k]["report"]["native_per_class"] if c["support"] > 0} for k in keys}
    classes = sorted(per[main], key=lambda c: per[main][c]["auprc"])
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    y = np.arange(len(classes))
    for s, k in enumerate(keys):
        vals = [per[k][c]["auprc"] for c in classes]
        ax.scatter(vals, y, s=22, color=SERIES[s], zorder=3, label=runs[k]["name"], edgecolor="white", linewidth=1)
    if other_key:
        for i, c in enumerate(classes):
            ax.plot([per[other_key][c]["auprc"], per[main][c]["auprc"]], [i, i], color=MUTED, linewidth=1, zorder=2)
    ax.set_yticks(y, [NATIVE_DISPLAY[c] for c in classes])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("AUPRC per error type (official test split)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left")  # the left half is empty because AUPRC > 0.5 for every type
    fig.savefig(out)
    plt.close(fig)


def pr_curves(summary, runs_dir, out, keys):
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    for s, k in enumerate(keys):
        d = load_predictions(os.path.join(runs_dir, k), "test")
        y, sc = d["y"].max(1), any_error_score(d["p"])
        p, r, _ = precision_recall_curve(y, sc)
        auprc = summary["runs"][k]["report"]["any_error"]["at_0.5"]["auprc"]
        ax.plot(r, p, color=SERIES[s], linewidth=2, label=f"{summary['runs'][k]['name']} (AUPRC {auprc:.3f})")
    ax.set_xlabel("Recall (reports with any error)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, 1.01)
    ax.legend(loc="lower left")
    fig.savefig(out)
    plt.close(fig)


def ppv_vs_prevalence(summary, out):
    r = summary["runs"][summary["main_run"]]["report"]["any_error"]
    prev = np.linspace(0.01, 0.30, 60)
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for s, (key, target) in enumerate((("val_sens_0.90", "0.90"), ("val_sens_0.95", "0.95"))):
        b = r[key]
        ppv = [row["ppv"] for row in prevalence_analysis(b["sensitivity"], b["specificity"], prev)]
        ax.plot(prev, ppv, color=SERIES[s], linewidth=2,
                label=f"Target Se {target} (test Se {b['sensitivity']:.3f}, Sp {b['specificity']:.3f})")
    ax.axvspan(0.03, 0.05, color=GRID, zorder=0)
    ax.text(0.04, 0.03, "reported\nerror rate", ha="center", va="bottom", color=INK_2, fontsize=6.5)
    ax.set_xlabel("True prevalence of reports with errors")
    ax.set_ylabel("Positive predictive value")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=6.5, title="Validation operating point",
              title_fontsize=6.5)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs_dir", default="results/runs")
    ap.add_argument("--compare", default=None, help="run key of the comparison model for the per-type figure")
    args = ap.parse_args()
    fig_dir = os.path.join(args.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    with open(os.path.join(args.results, "audit.json")) as f:
        label_agreement(json.load(f), os.path.join(fig_dir, "fig_label_agreement.png"))
    path = os.path.join(args.results, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            summary = json.load(f)
        if summary.get("main_run"):
            per_type_f1(summary, os.path.join(fig_dir, "fig_per_type_auprc.png"), args.compare)
            keys = [summary["main_run"]] + ([args.compare] if args.compare else [])
            pr_curves(summary, args.runs_dir, os.path.join(fig_dir, "fig_pr_any_error.png"), keys)
            ppv_vs_prevalence(summary, os.path.join(fig_dir, "fig_ppv_prevalence.png"))
    print("figures in", fig_dir, sorted(os.listdir(fig_dir)))


if __name__ == "__main__":
    main()
