#!/usr/bin/env python
"""
Summarise and plot the adaptive-efficacy job array.

Reads the per-replicate CSVs written by fit_efficacy.py, aggregates over
replicates, writes a tidy summary table, and renders a figure in the style of
Figure 4 of the manuscript (Okabe-Ito palette, four metric columns, SHD on a
log axis, 1 SD error bars, single frameless legend row above the panels).

    python summarize_efficacy.py \
        --resdir result/simu_eff/scGRAPH/NB \
        --out-csv efficacy_summary.csv --out-fig efficacy_figure.pdf

Input schema (long format, one file per replicate):
    method,measure,value
    hard,mcc,0.87        <- arm in {hard, soft, adaptive}
    truth,e_center,0.4   <- nominal efficacy centre
    truth,e_true_mean,.. <- REALISED mean efficacy for this replicate
    settings,pen1,1e-4   <- run settings, checked for consistency

The x axis uses the REALISED mean efficacy, not the nominal centre: with a
clipped uniform of half-width 0.1, the centres at 0 and 1 put half the genes
exactly on the boundary, so the realised means are about 0.025 and 0.975.
"""
import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

# --- house style, measured off Figure 4 of the preprint -----------------------
OKABE = {
    "orange":  "#E69F00",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
}

# Arms mapped onto the manuscript's visual grammar: the adaptive fit takes the
# vermillion circle/solid combination used for RICE throughout the paper.
ARM_STYLE = {
    "adaptive": dict(label=r"RICE (adaptive $e_j$)", color=OKABE["vermil"],
                     marker="o", ls="-",  mfc=OKABE["vermil"]),
    "soft":     dict(label=r"RICE (fixed $e_j=0$, soft)", color=OKABE["sky"],
                     marker="s", ls="--", mfc=OKABE["sky"]),
    "hard":     dict(label=r"RICE (fixed $e_j=1$, hard)", color=OKABE["green"],
                     marker="D", ls=":",  mfc=OKABE["green"]),
}
ARM_ORDER = ["adaptive", "soft", "hard"]

# measure name in the CSV -> panel title. Aliases cover both the measurement()
# keys and the PPV/TPR naming used in the figures.
METRICS = [
    (("precision", "ppv"), "PPV",  False),
    (("recall", "tpr"),    "TPR",  False),
    (("mcc",),             "MCC",  False),
    (("shd",),             "SHD",  True),   # log y axis
]

STEM_RE = re.compile(
    r"n(?P<n>\d+)k_p(?P<p>\d+)_s(?P<s0>\d+)_(?P<graph>\w+?)_xic(?P<xic>\d+)_repl(?P<repl>\d+)"
)

SETTING_KEYS = ["pen1", "w_threshold", "max_iter", "epochs", "lr",
                "xi_scope_is_parent", "backend_is_cupy"]


def load_results(resdir, pattern):
    """Read every per-replicate CSV into one long DataFrame."""
    paths = sorted(glob.glob(os.path.join(resdir, pattern)))
    if not paths:
        sys.exit(f"no files matched {os.path.join(resdir, pattern)}")
    frames = []
    for path in paths:
        stem = os.path.basename(path)[:-4]
        try:
            df = pd.read_csv(path)
        except Exception as exc:                      # truncated / empty file
            warnings.warn(f"skipping unreadable {stem}: {exc}")
            continue
        if not {"method", "measure", "value"} <= set(df.columns):
            warnings.warn(f"skipping {stem}: unexpected columns {list(df.columns)}")
            continue
        m = STEM_RE.search(stem)
        df["file"] = stem
        df["repl"] = int(m.group("repl")) if m else np.nan
        df["xic_tag"] = int(m.group("xic")) if m else np.nan
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"read {len(frames)} files, {len(out)} rows, from {resdir}")
    return out


def check_settings(long):
    """Fail loudly if the arms were not all fit under the same settings."""
    s = long[long.method == "settings"]
    if s.empty:
        warnings.warn("no settings rows found; cannot verify run consistency")
        return
    tab = s.pivot_table(index="measure", values="value",
                        aggfunc=lambda v: sorted(set(np.round(v, 12))))
    mixed = {k: v for k, v in tab["value"].items() if len(v) > 1}
    print("\nrun settings:")
    for k, v in tab["value"].items():
        print(f"  {k:<20} {v[0] if len(v) == 1 else v}")
    if mixed:
        print("\n!! settings differ across files -- the arms are NOT comparable:")
        for k, v in mixed.items():
            print(f"     {k}: {v}")
        print("   Re-run the affected centres with one set of settings, or split "
              "the summary by setting, before using this figure.")


def per_replicate_truth(long):
    """One row per file carrying the nominal and realised efficacy."""
    t = long[long.method == "truth"]
    wide = t.pivot_table(index=["file", "repl", "xic_tag"], columns="measure",
                         values="value").reset_index()
    if "e_center" not in wide.columns:
        # older CSVs carry xi units only; e = 1 - xi
        if "xi_center" not in wide.columns:
            sys.exit("CSVs carry neither e_center nor xi_center")
        wide["e_center"] = 1.0 - wide["xi_center"]
        print("note: e_center absent, derived as 1 - xi_center")
    if "e_true_mean" not in wide.columns:
        if "xi_true_mean" in wide.columns:
            wide["e_true_mean"] = 1.0 - wide["xi_true_mean"]
            print("note: e_true_mean absent, derived as 1 - xi_true_mean")
        else:
            wide["e_true_mean"] = wide["e_center"]
            warnings.warn("no realised efficacy in CSVs; falling back to the "
                          "nominal centre, which is biased at the boundaries")
    return wide


def aggregate(long, truth, min_reps):
    """Mean, SD and n per (efficacy centre, arm, measure)."""
    arms = long[long.method.isin(ARM_ORDER)].copy()
    arms = arms.merge(truth[["file", "e_center", "e_true_mean"]], on="file",
                      how="left")
    g = arms.groupby(["e_center", "method", "measure"], dropna=False)
    summ = g["value"].agg(mean="mean", sd="std", n="count").reset_index()
    # realised efficacy is a property of the centre, averaged over replicates
    x = (truth.groupby("e_center")["e_true_mean"]
               .agg(e_realised="mean", e_realised_sd="std").reset_index())
    summ = summ.merge(x, on="e_center", how="left")
    summ = summ.rename(columns={"method": "arm"}).sort_values(
        ["measure", "arm", "e_center"]).reset_index(drop=True)

    counts = (summ[summ.measure == summ.measure.iloc[0]]
              .groupby("e_center")["n"].max())
    print("\nreplicates per efficacy centre:")
    for c, k in counts.items():
        flag = "  <-- incomplete" if k < min_reps else ""
        print(f"  e_center={c:<5} n={int(k)}{flag}")
    print("\nmeasures available:", ", ".join(sorted(summ.measure.unique())))
    return summ


def _pick(summ, aliases):
    for a in aliases:
        if (summ.measure == a).any():
            return a
    return None


def plot_figure(summ, out_fig, title=None):
    """Figure 4 style: metric columns (row A) + efficacy recovery (row B)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.bbox": "tight",
    })
    present = [(al, lab, lg) for al, lab, lg in METRICS if _pick(summ, al)]
    rowA, rowB = [], []
    if not present:
        sys.exit("none of the expected metrics are present in the summary")

    has_rec = (summ.measure == "e_mean_hat").any()
    nrow = 2 if has_rec else 1
    fig = plt.figure(figsize=(2.55 * len(present), 2.65 * nrow + 0.6))
    gs = fig.add_gridspec(nrow, len(present), hspace=0.62, wspace=0.33,
                          top=0.86 if nrow == 1 else 0.90)

    # ---- row A: graph-recovery metrics ----
    for j, (aliases, label, logy) in enumerate(present):
        ax = fig.add_subplot(gs[0, j])
        meas = _pick(summ, aliases)
        for arm in ARM_ORDER:
            d = summ[(summ.arm == arm) & (summ.measure == meas)]
            if d.empty:
                continue
            st = ARM_STYLE[arm]
            ax.errorbar(d.e_realised, d["mean"], yerr=d["sd"],
                        color=st["color"], marker=st["marker"], ls=st["ls"],
                        mfc=st["mfc"], mec=st["color"], ms=4.5, lw=1.2,
                        elinewidth=0.9, capsize=2.2, zorder=3)
        ax.set_title(label)
        if logy:
            ax.set_yscale("log")
            # powers of ten only, as in Figure 4 -- matplotlib otherwise labels
            # minor ticks (6x10^2 and the like) on short log ranges
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.yaxis.set_minor_formatter(NullFormatter())
        if j == 0:
            ax.set_ylabel("Graph recovery")
        ax.set_xlim(-0.05, 1.05)
        rowA.append(ax)

    # ---- row B: efficacy recovery, adaptive arm only ----
    if has_rec:
        ax = fig.add_subplot(gs[1, 0])
        d = summ[(summ.arm == "adaptive") & (summ.measure == "e_mean_hat")]
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=(0, (4, 3)),
                zorder=1, label="identity")
        st = ARM_STYLE["adaptive"]
        ax.errorbar(d.e_realised, d["mean"], yerr=d["sd"], color=st["color"],
                    marker=st["marker"], ls=st["ls"], mfc=st["mfc"],
                    mec=st["color"], ms=4.5, lw=1.2, elinewidth=0.9,
                    capsize=2.2, zorder=3)
        ax.set_title(r"$\hat{e}$ vs. truth")
        ax.set_ylabel(r"Estimated $\hat{\bar{e}}$")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        rowB.append(ax)

        # RMSE, split by whether the gene has parents (identifiability)
        panels = [("e_rmse_parents", "with parents"),
                  ("e_rmse_parentless", "no parents")]
        avail = [(m, lb) for m, lb in panels if (summ.measure == m).any()]
        if avail:
            ax = fig.add_subplot(gs[1, 1])
            for (m, lb), c, mk, ls in zip(avail,
                                          [OKABE["vermil"], OKABE["purple"]],
                                          ["o", "^"], ["-", "-."]):
                d = summ[(summ.arm == "adaptive") & (summ.measure == m)]
                ax.errorbar(d.e_realised, d["mean"], yerr=d["sd"], color=c,
                            marker=mk, ls=ls, mfc=c if mk == "o" else "none",
                            mec=c, ms=4.5, lw=1.2, elinewidth=0.9, capsize=2.2,
                            label=lb)
            ax.set_title(r"$\hat{e}$ error by identifiability")
            ax.set_ylabel(r"RMSE of $\hat{e}_j$")
            ax.set_xlim(-0.05, 1.05)
            ax.legend(frameon=False, fontsize=8)
            rowB.append(ax)

        if (summ.measure == "auc").any():
            ax = fig.add_subplot(gs[1, 2])
            for arm in ARM_ORDER:
                d = summ[(summ.arm == arm) & (summ.measure == "auc")]
                if d.empty:
                    continue
                st = ARM_STYLE[arm]
                ax.errorbar(d.e_realised, d["mean"], yerr=d["sd"],
                            color=st["color"], marker=st["marker"], ls=st["ls"],
                            mfc=st["mfc"], mec=st["color"], ms=4.5, lw=1.2,
                            elinewidth=0.9, capsize=2.2)
            ax.set_title("AUC")
            ax.set_xlim(-0.05, 1.05)
            rowB.append(ax)

        if (summ.measure == "time").any():
            ax = fig.add_subplot(gs[1, 3])
            for arm in ARM_ORDER:
                d = summ[(summ.arm == arm) & (summ.measure == "time")]
                if d.empty:
                    continue
                st = ARM_STYLE[arm]
                ax.errorbar(d.e_realised, d["mean"] / 60, yerr=d["sd"] / 60,
                            color=st["color"], marker=st["marker"], ls=st["ls"],
                            mfc=st["mfc"], mec=st["color"], ms=4.5, lw=1.2,
                            elinewidth=0.9, capsize=2.2)
            ax.set_title("Runtime")
            ax.set_ylabel("Minutes per fit")
            ax.set_xlim(-0.05, 1.05)
            rowB.append(ax)

    # shared x label under each row block, and bold panel letters
    XLAB = r"Perturbation efficacy $\bar{e}$"
    for letter, row in (("A", rowA), ("B", rowB)):
        if not row:
            continue
        boxes = [ax.get_position() for ax in row]
        y0 = min(b.y0 for b in boxes)
        fig.text(0.5, y0 - 0.062, XLAB, ha="center", va="top")
        if nrow > 1:
            fig.text(min(b.x0 for b in boxes) - 0.055,
                     max(b.y1 for b in boxes) + 0.045, letter,
                     ha="left", va="bottom", fontsize=13, fontweight="bold")

    handles = [Line2D([], [], color=ARM_STYLE[a]["color"],
                      marker=ARM_STYLE[a]["marker"], ls=ARM_STYLE[a]["ls"],
                      mfc=ARM_STYLE[a]["mfc"], mec=ARM_STYLE[a]["color"],
                      ms=5, lw=1.2, label=ARM_STYLE[a]["label"])
               for a in ARM_ORDER if (summ.arm == a).any()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.text(0.5, 1.045, title, ha="center", fontsize=9, color="0.3")
    fig.savefig(out_fig)
    print(f"\nwrote {out_fig}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resdir", required=True,
                    help="directory of per-replicate CSVs")
    ap.add_argument("--pattern", default="*_xic*_repl*.csv")
    ap.add_argument("--out-csv", default="efficacy_summary.csv")
    ap.add_argument("--out-fig", default="efficacy_figure.pdf")
    ap.add_argument("--min-reps", type=int, default=20,
                    help="warn when a centre has fewer replicates than this")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()

    long = load_results(a.resdir, a.pattern)
    check_settings(long)
    truth = per_replicate_truth(long)
    summ = aggregate(long, truth, a.min_reps)
    summ.to_csv(a.out_csv, index=False)
    print(f"\nwrote {a.out_csv}  {summ.shape}")

    key = _pick(summ, ("mcc",))
    if key:
        piv = (summ[summ.measure == key]
               .pivot_table(index="e_realised", columns="arm", values="mean"))
        print(f"\n{key.upper()} by arm:\n{piv.round(3).to_string()}")

    if not a.no_fig:
        plot_figure(summ, a.out_fig)


if __name__ == "__main__":
    main()
