#!/usr/bin/env python
"""
Control-function comparison across exclusion-restriction violation regimes.

Reads the per-replicate CSVs written by fit_cf_comparison.py, runs the paired
reduced-vs-full tests, and renders the four-metric figure (PPV / TPR above, MCC / SHD
below) in the manuscript's house style.

    python plot_control_function_violation.py --resdir result/simu_cfcompare/scGRAPH/NB

Notes on three deliberate choices:

* The tests are PAIRED within replicate and ONE-SIDED with the alternative "the reduced
  control function is better". The direction flips per metric: greater for PPV/TPR/MCC,
  less for SHD. This is why the brackets are drawn here rather than with
  `statannotations`, whose defaults are independent two-sided tests.
* Only comparisons with p <= 0.05 are annotated, and the caption says so, so that a
  missing bracket does not read as a missing test.
* Three tests are computed and all are written to the tests CSV: the paired t-test, the
  Wilcoxon signed-rank test (DEFAULT for the brackets -- the paired nonparametric
  analogue), and the Mann-Whitney U test, which is reported for reference only because
  it treats the arms as independent samples and so discards the pairing. `--test`
  selects which one drives the brackets; the chosen p-value lands in the `p` column and
  `stars` is derived from it.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from scipy import stats

# --- naming -----------------------------------------------------------------------
MEASURES = {"precision": "PPV", "recall": "TPR", "mcc": "MCC", "shd": "SHD"}
METRIC_ORDER = ["PPV", "TPR", "MCC", "SHD"]
LOWER_BETTER = {"SHD"}

VIOLATIONS = {"none": "None",
              "direct_s0": "Direct\ndeterministic",
              "direct_s05": "Direct\nstochastic",
              "mediator": "Hidden\nmediator"}
VIOL_ORDER = [VIOLATIONS[k] for k in ["none", "direct_s0", "direct_s05", "mediator"]]

ARMS = {"rcf": "Reduced control function",
        "fcf": "Full control function",
        "nocf": "No control function"}
# reduced left, full centre, none right -- matches cfm.pdf
ARM_ORDER = ["Reduced control function", "Full control function", "No control function"]
PALETTE = {"Reduced control function": "#D55E00",
           "Full control function": "grey",
           "No control function": "#56B4E9"}
# dodge offsets for three hues at seaborn's default box width of 0.8
OFFSET = {"Reduced control function": -0.267,
          "Full control function": 0.0,
          "No control function": 0.267}


def load_results(resdir):
    """Concatenate the per-replicate CSVs, tagging each with its replicate index."""
    files = sorted(glob.glob(os.path.join(resdir, "*_cfcompare_*.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(resdir, "*.csv")))
    if not files:
        raise SystemExit(f"no CSVs found under {resdir}")
    frames = []
    for f in files:
        m = re.search(r"repl(\d+)\.csv$", f)
        if m is None:
            continue
        frames.append(pd.read_csv(f).assign(rep=int(m.group(1))))
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(frames)} replicate files from {resdir}")
    return df


def check_settings(df):
    """Refuse to pool replicates that were fit under different settings."""
    st = (df[df.method == "settings"]
          .pivot_table(index="rep", columns="measure", values="value"))
    varying = {c: sorted(st[c].unique()) for c in st.columns
               if c != "replicate" and st[c].nunique() > 1}
    if varying:
        raise SystemExit(f"replicates differ in fit settings, refusing to pool: {varying}")
    fixed = {c: st[c].iloc[0] for c in ["pen1", "w_threshold", "d", "max_iter", "epochs",
                                        "lr"] if c in st}
    print("settings (uniform across replicates):", fixed)
    return fixed


def to_long(df):
    """Per-replicate rows of (rep, violation, arm, metric, value)."""
    long = (df[df.method.isin(ARMS) & df.measure.isin(MEASURES)]
            .assign(measure=lambda d: d.measure.map(MEASURES),
                    violation=lambda d: d.config.map(VIOLATIONS),
                    method=lambda d: d.method.map(ARMS))
            [["rep", "violation", "method", "measure", "value"]])
    counts = long.groupby(["measure", "violation", "method"]).size()
    if counts.nunique() != 1:
        raise SystemExit(f"unbalanced design, cells range {counts.min()}-{counts.max()}; "
                         "an array element probably failed")
    return long, int(counts.iloc[0])


TEST_NAME = {"wilcoxon": "Wilcoxon signed-rank (paired)",
             "ttest": "Student t (paired)",
             "mannwhitney": "Mann-Whitney U (UNPAIRED -- discards the pairing)"}


def stars(p):
    """statannotations' star convention."""
    return ("****" if p <= 1e-4 else "***" if p <= 1e-3 else
            "**" if p <= 1e-2 else "*" if p <= 5e-2 else "ns")


def paired_tests(long, test="wilcoxon"):
    """
    One-sided tests, reduced vs full, per metric and violation regime.

    Every test is computed and written to the CSV; `test` selects which one drives the
    brackets. Default is the Wilcoxon signed-rank test, the paired nonparametric
    analogue. "mannwhitney" is available but treats the arms as independent samples,
    which discards the pairing.
    """
    rows = []
    for metric in METRIC_ORDER:
        for viol in VIOL_ORDER:
            w = (long[(long.measure == metric) & (long.violation == viol)]
                 .pivot(index="rep", columns="method", values="value"))
            r = w["Reduced control function"].values
            f = w["Full control function"].values
            alt = "less" if metric in LOWER_BETTER else "greater"
            t, p_t = stats.ttest_rel(r, f, alternative=alt)
            _, p_w = stats.wilcoxon(r, f, alternative=alt)
            # Mann-Whitney U is reported for reference only: it treats the two arms as
            # INDEPENDENT samples and so discards the pairing that the shared graph and
            # seed provide. Wilcoxon signed-rank is its paired analogue and is the
            # nonparametric test that matches this design.
            _, p_u = stats.mannwhitneyu(r, f, alternative=alt)
            d = r - f
            rows.append(dict(measure=metric, violation=viol.replace("\n", " "), n=len(r),
                             rcf_mean=r.mean(), fcf_mean=f.mean(), mean_diff=d.mean(),
                             se=d.std(ddof=1) / np.sqrt(len(d)),
                             alternative=f"rcf {alt} fcf", t=t, p_ttest=p_t,
                             p_wilcoxon=p_w, p_mannwhitney_unpaired=p_u,
                             p=None, stars=None))
    out = pd.DataFrame(rows)
    out["p"] = out[{"ttest": "p_ttest", "wilcoxon": "p_wilcoxon",
                    "mannwhitney": "p_mannwhitney_unpaired"}[test]]
    out["stars"] = out["p"].map(stars)
    # name the test in the table itself: with three p-value columns present it is
    # otherwise impossible to tell from the CSV which one the brackets used
    out["test_used"] = TEST_NAME[test]
    print(f"[test] brackets use: {TEST_NAME[test]}, one-sided, "
          f"alternative = reduced control function is better")
    return out


def plot_figure(long, tests, nrep, out_png, alpha=0.05, test=None):
    """
    Render the four-metric figure. Brackets mark the reduced-vs-full comparison.

    `test` is accepted only so main() can pass it; it is unused, because the caption
    naming the test now lives in the LaTeX caption rather than in the figure. Which
    test drives the brackets is decided in paired_tests(), which writes the chosen
    p-value to the `p` column -- read that here, NOT `p_ttest`, or the brackets and the
    printed stars can disagree whenever --test is not "ttest".
    """
    missing = {"p", "test_used"} - set(tests.columns)
    if missing:
        raise SystemExit(
            f"`tests` is missing {sorted(missing)} -- it was produced by an older "
            "paired_tests(). Re-run the cell that DEFINES the functions, then rebuild "
            "`tests` with paired_tests(long, test=...). Brackets must not be drawn from "
            "a p_ttest column when another test was selected.")
    # do not trust a pre-computed stars column: derive the labels from the selected
    # p-value here, so a stale frame cannot label brackets with the wrong test
    tests = tests.assign(stars=tests["p"].map(stars))
    print(f"[plot] brackets: {tests.test_used.iloc[0]}, alpha={alpha}")

    g = sns.FacetGrid(long, col="measure", col_wrap=2, sharey=False, height=3, aspect=2,
                      col_order=METRIC_ORDER)
    g.map_dataframe(sns.boxplot, x="violation", y="value", hue="method", palette=PALETTE,
                    hue_order=ARM_ORDER, order=VIOL_ORDER, showfliers=False,
                    boxprops={"alpha": 0.4})
    g.map_dataframe(sns.stripplot, x="violation", y="value", hue="method",
                    palette=PALETTE, hue_order=ARM_ORDER, order=VIOL_ORDER, jitter=True,
                    size=3, dodge=True)

    ax_shd = g.axes_dict["SHD"]
    ax_shd.set_yscale("log")
    ax_shd.yaxis.set_major_locator(LogLocator(base=10))
    ax_shd.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3)))
    ax_shd.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax_shd.yaxis.set_minor_formatter(LogFormatterSciNotation(labelOnlyBase=False))

    g.set_axis_labels("", "")
    g.set_titles(col_template="{col_name}")

    n_marked = 0
    for metric in METRIC_ORDER:
        ax = g.axes_dict[metric]
        logy = metric in LOWER_BETTER
        sub = long[long.measure == metric]
        lo, hi = sub.value.min(), sub.value.max()
        span = (np.log10(hi) - np.log10(lo)) if logy else (hi - lo)
        step, bar = 0.10 * span, 0.030 * span
        tops = [hi]
        for gi, viol in enumerate(VIOL_ORDER):
            row = tests[(tests.measure == metric) &
                        (tests.violation == viol.replace("\n", " "))].iloc[0]
            if row.p > alpha:                           # significant comparisons only
                continue
            pair = sub[(sub.violation == viol) &
                       sub.method.isin(["Reduced control function",
                                        "Full control function"])]
            ytop = pair.value.max()
            y = 10 ** (np.log10(ytop) + step) if logy else ytop + step
            y2 = 10 ** (np.log10(y) + bar) if logy else y + bar
            x1 = gi + OFFSET["Reduced control function"]
            x2 = gi + OFFSET["Full control function"]
            ax.plot([x1, x1, x2, x2], [y, y2, y2, y], lw=1.0, c="0.25",
                    solid_joinstyle="miter", clip_on=False, zorder=5)
            ax.text((x1 + x2) / 2, y2, row.stars, ha="center", va="bottom", fontsize=12,
                    color="0.25", zorder=5)
            tops.append(y2)
            n_marked += 1
        head = max(tops)
        ax.set_ylim(top=10 ** (np.log10(head) + 0.16 * span) if logy
                    else head + 0.16 * span)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_title(ax.get_title(), fontsize=16)

    g.figure.legend(handles=[Patch(facecolor=PALETTE[a], label=a, alpha=0.6)
                             for a in ARM_ORDER],
                    loc="upper right", bbox_to_anchor=(1, 1.07), ncol=3, fontsize=14,
                    frameon=False, handlelength=1, handleheight=1, handletextpad=0.5)
    # SHD runs the opposite way to the other three panels, and the in-figure direction
    # cues are gone -- the LaTeX caption states it instead.
    g.fig.text(0.53, -0.01, "Violation type", ha="center", fontsize=16)
    plt.tight_layout()
    g.figure.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"wrote {out_png}  ({n_marked} of {len(tests)} comparisons marked)")
    return g.figure


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resdir", required=True,
                    help="directory of per-replicate *_cfcompare_*_repl*.csv files")
    ap.add_argument("--out-fig", default="cfm_violation.png")
    ap.add_argument("--out-tests", default="cfm_violation_tests.csv")
    ap.add_argument("--out-means", default="cfm_violation_means.csv")
    ap.add_argument("--test", choices=["wilcoxon", "ttest", "mannwhitney"],
                    default="wilcoxon",
                    help="which test drives the brackets; all three are written to the "
                         "tests CSV. wilcoxon (signed-rank) is the paired nonparametric "
                         "analogue; mannwhitney discards the pairing.")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="annotate only comparisons at or below this p-value")
    a = ap.parse_args()

    df = load_results(a.resdir)
    check_settings(df)
    long, nrep = to_long(df)

    tests = paired_tests(long, test=a.test)
    # "None" round-trips through CSV as NaN, so name the reference explicitly on disk;
    # the figure keeps the short axis label
    CSV_LABEL = {"None": "No violation"}
    tests.assign(violation=tests.violation.replace(CSV_LABEL)).to_csv(a.out_tests,
                                                                     index=False)
    print(tests[["measure", "violation", "rcf_mean", "fcf_mean", "mean_diff", "p",
                 "stars", "p_ttest", "p_wilcoxon", "p_mannwhitney_unpaired"]]
          .round(4).to_string(index=False))

    means = (long.pivot_table(index=["measure", "violation"], columns="method",
                              values="value")
             .loc[[(m, v) for m in METRIC_ORDER for v in VIOL_ORDER]][ARM_ORDER])
    means.index = pd.MultiIndex.from_tuples([(m, v.replace("\n", " "))
                                             for m, v in means.index],
                                            names=["measure", "violation"])
    means.columns = ["rcf", "fcf", "nocf"]
    means = means.rename(index=CSV_LABEL, level="violation")
    means.round(4).to_csv(a.out_means)

    plot_figure(long, tests, nrep, a.out_fig, alpha=a.alpha, test=a.test)


if __name__ == "__main__":
    main()
