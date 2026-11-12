#!/usr/bin/env python
"""
Proportion of significant perturbation effects among the top-k edges, for several
network-inference methods at matched edge counts.

This generalizes the single-method version of the figure to a method comparison, which
is what answers the reviewer's objection: because RICE returns fewer edges than some
competitors, enrichment and annotation comparisons at each method's own threshold favor
whichever method is most conservative. Ranking every method's edges by |weight| and
comparing the top k of each removes that confound.

    from topk_significance import summarize, pairwise_tests, plot_topk
    summary = summarize(edges, ks=(100, 200, 400, 700, 1000))
    tests = pairwise_tests(summary, reference="RICE")
    plot_topk(summary, tests, out_png="RPE1_significant_proportion.pdf")

`edges` is {method: (w, p_direct, p_total)} with the three arrays aligned edge-by-edge,
exactly the layout of the existing npz files -- see `load_npz_triplets`.

TWO THINGS THE SINGLE-METHOD VERSION DID NOT HAVE TO HANDLE
-----------------------------------------------------------
1. A method may hold fewer than k edges. Silently taking `comb[:k]` then dividing by k
   understates that method's proportion; dividing by len(comb[:k]) compares it at a
   different edge count than the others, which is the very confound the figure exists to
   remove. `summarize` records `n_available` and `truncated` per cell, `plot_topk` draws
   truncated points as open markers, and `max_matched_k` reports the largest k at which
   ALL methods are genuinely matched. Report that k in the text.

2. With one method, a binomial test against 0.5 answers "is this better than chance".
   With four, the question is whether the reference method beats each competitor at the
   SAME k, which is a two-proportion comparison -- Fisher's exact test on the 2x2 table
   of (significant, not significant) x (reference, competitor). `pairwise_tests` returns
   those; the chance-level test is kept as `p_vs_chance` for continuity with the earlier
   figure.

Wilson intervals are used for the error bars, as before: they keep coverage near the
boundary, where a proportion of 0.95 on 100 edges otherwise produces an interval that
runs past 1.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, fisher_exact
from statsmodels.stats.proportion import proportion_confint

# RICE in vermillion, matching the other manuscript figures; the rest Okabe-Ito
PALETTE = {"RICE": "#D55E00", "DOTEARS": "#0072B2", "IBCD": "#009E73",
           "inspre": "#CC79A7", "GIES": "#E69F00", "NOTEARS": "#56B4E9"}
MARKERS = {"RICE": "D", "DOTEARS": "o", "IBCD": "s", "inspre": "^",
           "GIES": "v", "NOTEARS": "P"}
EFFECTS = {"total": "Total", "direct": "Direct"}
ALPHA = 0.05


def load_npz_triplets(ce_path, pval_direct_path, pval_total_path, weight_index=0):
    """
    Read the existing npz layout into {method: (w, p_direct, p_total)}.

    `ce_path` holds one entry per method whose [weight_index] element is the edge-weight
    vector; the two p-value files are keyed the same way and aligned edge-by-edge.
    """
    CE = np.load(ce_path, allow_pickle=True)
    PD = np.load(pval_direct_path, allow_pickle=True)
    PT = np.load(pval_total_path, allow_pickle=True)
    out = {}
    for key in CE.files:
        w = np.asarray(CE[key][weight_index], dtype=float).ravel()
        pd_, pt_ = np.asarray(PD[key]).ravel(), np.asarray(PT[key]).ravel()
        if not (len(w) == len(pd_) == len(pt_)):
            raise ValueError(f"{key}: lengths differ -- w {len(w)}, p_direct {len(pd_)}, "
                             f"p_total {len(pt_)}; the three arrays must be aligned "
                             "edge-by-edge")
        out[key] = (w, pd_, pt_)
    return out


def summarize(edges, ks=(100, 200, 400, 700, 1000), alpha=ALPHA):
    """
    Rank each method's edges by |weight| and count significant effects in the top k.

    Returns a tidy frame with one row per (method, effect, k): success, total,
    proportion, Wilson bounds, n_available and truncated.
    """
    rows = []
    for method, (w, p_direct, p_total) in edges.items():
        order = np.argsort(-np.abs(np.asarray(w, dtype=float)))
        n_avail = len(order)
        for effect, p in (("direct", np.asarray(p_direct)),
                          ("total", np.asarray(p_total))):
            p_sorted = p[order]
            for k in ks:
                take = min(k, n_avail)
                sig = int(np.sum(p_sorted[:take] < alpha))
                lo, hi = proportion_confint(sig, take, alpha=0.05, method="wilson")
                rows.append(dict(method=method, effect=effect, k=k, success=sig,
                                 total=take, proportion=sig / take,
                                 lower=lo, upper=hi,
                                 n_available=n_avail, truncated=take < k))
    df = pd.DataFrame(rows)
    df["yerr_lower"] = df.proportion - df.lower
    df["yerr_upper"] = df.upper - df.proportion
    return df


def max_matched_k(summary):
    """Largest k at which every method still has k edges -- the honest comparison range."""
    per_method = summary.groupby("method").n_available.first()
    return int(per_method.min()), per_method.to_dict()


def pairwise_tests(summary, reference="RICE", alternative="greater"):
    """
    Reference vs each competitor at matched k (Fisher's exact), plus the chance-level
    binomial test kept from the single-method figure.
    """
    rows = []
    for (effect, k), grp in summary.groupby(["effect", "k"]):
        g = grp.set_index("method")
        if reference not in g.index:
            raise KeyError(f"reference method {reference!r} not in {list(g.index)}")
        ref = g.loc[reference]
        for method, row in g.iterrows():
            p_chance = binomtest(int(row.success), int(row.total), p=0.5,
                                 alternative="greater").pvalue
            if method == reference:
                p_vs_ref = np.nan
            else:
                table = [[int(ref.success), int(ref.total - ref.success)],
                         [int(row.success), int(row.total - row.success)]]
                p_vs_ref = fisher_exact(table, alternative=alternative)[1]
            rows.append(dict(effect=effect, k=k, method=method, reference=reference,
                             success=int(row.success), total=int(row.total),
                             proportion=row.proportion, p_vs_chance=p_chance,
                             p_vs_reference=p_vs_ref,
                             matched=bool(not row.truncated and not ref.truncated)))
    out = pd.DataFrame(rows)
    out["stars_vs_reference"] = out.p_vs_reference.map(p_to_stars)
    out["stars_vs_chance"] = out.p_vs_chance.map(p_to_stars)
    return out


def bootstrap_greater(edges, edge_ids, ks=(100, 200, 400, 700, 1000), reference="RICE",
                      clusters=None, n_boot=2000, alpha=ALPHA, seed=0):
    """
    One-sided p-value for "the reference method's top-k edges are significant more often
    than the competitor's", by a PAIRED CLUSTER BOOTSTRAP.

    Why not just Fisher's exact (which `pairwise_tests` reports): Fisher treats the two
    top-k sets as independent samples. They are not.

      * The sets OVERLAP. A gene pair in both methods' top k contributes the same
        p-value to both proportions, so the two counts are positively correlated and an
        independent-samples test understates the standard error of their difference.
      * Edges sharing a TARGET gene are not independent of each other either: their
        effect estimates come from one regression on the same held-out cells.

    This resamples CLUSTERS (target genes by default) with replacement, and recomputes
    both methods' proportions on the same resample, so the overlap is preserved within
    each bootstrap replicate and the comparison stays paired. The reported p is the
    percentile-bootstrap one-sided value, (1 + #{diff* <= 0}) / (B + 1); it is an
    approximation, not an exact test, and is labeled as such in the output.

    edge_ids : {method: array of hashable edge identifiers, aligned with the weights}
               e.g. (source, target) tuples, so an edge can be recognized across methods
    clusters : {method: array of cluster labels}; defaults to the TARGET taken as the
               second element of each edge id, falling back to the edge id itself
    """
    rng = np.random.default_rng(seed)

    def _clusters_for(method):
        if clusters is not None:
            return np.asarray(clusters[method], dtype=object)
        ids = np.asarray(edge_ids[method], dtype=object)
        try:
            return np.array([e[1] for e in ids], dtype=object)
        except (TypeError, IndexError):
            return ids

    rows = []
    for effect, idx in (("direct", 1), ("total", 2)):
        for method in edges:
            if method == reference:
                continue
            out = {}
            for name in (reference, method):
                w = np.asarray(edges[name][0], dtype=float)
                p = np.asarray(edges[name][idx], dtype=float)
                order = np.argsort(-np.abs(w))
                out[name] = (order, p, _clusters_for(name))
            for k in ks:
                sel = {}
                for name in (reference, method):
                    order, p, cl = out[name]
                    take = order[:min(k, len(order))]
                    sel[name] = (p[take] < alpha, cl[take])
                pool = np.array(sorted({c for _, cl in sel.values() for c in cl},
                                       key=str), dtype=object)
                obs = {n: s.mean() for n, (s, _) in sel.items()}
                diff = obs[reference] - obs[method]

                boot = np.empty(n_boot)
                members = {n: {c: s[cl == c] for c in pool} for n, (s, cl) in sel.items()}
                for b in range(n_boot):
                    draw = pool[rng.integers(0, len(pool), len(pool))]
                    vals = {}
                    for n in (reference, method):
                        hits = [members[n][c] for c in draw if members[n][c].size]
                        vals[n] = (np.concatenate(hits).mean() if hits else np.nan)
                    boot[b] = vals[reference] - vals[method]
                boot = boot[~np.isnan(boot)]
                p_boot = (1 + int(np.sum(boot <= 0))) / (len(boot) + 1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                rows.append(dict(effect=effect, k=k, reference=reference, method=method,
                                 prop_reference=obs[reference], prop_method=obs[method],
                                 diff=diff, ci_lower=lo, ci_upper=hi,
                                 p_bootstrap_greater=p_boot, n_boot=len(boot),
                                 n_clusters=len(pool),
                                 test="paired cluster bootstrap (percentile, one-sided)"))
    df = pd.DataFrame(rows)
    df["stars"] = df.p_bootstrap_greater.map(p_to_stars)
    return df


def p_to_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return ("****" if p < 1e-4 else "***" if p < 1e-3 else
            "**" if p < 1e-2 else "*" if p < 0.05 else "")


def plot_topk(summary, tests=None, out_png="topk_significance.pdf", reference="RICE",
              method_order=None, annotate="vs_all", figsize=(3.6, 6.3)):
    """
    Two stacked panels (Total above, Direct below), methods as series across k.

    annotate:
      "vs_all"    star above the reference series at each k where it beats EVERY
                  competitor (Fisher, p<0.05). This is the claim the comparison makes;
                  per-method stars collide once there are four series.
      "vs_chance" the original single-method annotation, reference series only.
      None        no stars.
    """
    methods = method_order or ([reference] + sorted(m for m in summary.method.unique()
                                                    if m != reference))
    ks = sorted(summary.k.unique())
    xpos = {k: i for i, k in enumerate(ks)}

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, layout="constrained")
    fig.get_layout_engine().set(hspace=0.06)

    for ax, effect in zip(axes, ["total", "direct"]):
        sub = summary[summary.effect == effect]
        for method in methods:
            g = sub[sub.method == method].sort_values("k")
            x = [xpos[k] for k in g.k]
            color = PALETTE.get(method, "0.3")
            ax.errorbar(x, g.proportion,
                        yerr=[g.yerr_lower.values, g.yerr_upper.values],
                        fmt="none", ecolor=color, elinewidth=1, capsize=4, zorder=5)
            ax.plot(x, g.proportion, linestyle=":", linewidth=1, color=color, zorder=6)
            # truncated points are open: that method ran out of edges before k
            for xi, yi, trunc in zip(x, g.proportion, g.truncated):
                ax.plot(xi, yi, marker=MARKERS.get(method, "o"), markersize=5,
                        color=color, markerfacecolor="white" if trunc else color,
                        markeredgecolor=color, linestyle="none", zorder=7,
                        label=method if (xi == x[0] and effect == "total") else None)
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.set_title(EFFECTS[effect], fontsize=16, y=1.05)
        ax.set_ylabel("")
        ax.set_xlabel("")

        if annotate and tests is not None:
            t = tests[(tests.effect == effect)]
            for k in ks:
                tk = t[t.k == k]
                ref_row = tk[tk.method == reference]
                if ref_row.empty:
                    continue
                others = tk[tk.method != reference]
                if annotate == "vs_all":
                    if others.empty or not (others.p_vs_reference < 0.05).all():
                        continue
                    star = p_to_stars(others.p_vs_reference.max())
                else:
                    star = ref_row.stars_vs_chance.iloc[0]
                r = summary[(summary.effect == effect) & (summary.k == k) &
                            (summary.method == reference)].iloc[0]
                ax.text(xpos[k], r.proportion + r.yerr_upper + 0.01, star,
                        ha="center", va="bottom", fontsize=14,
                        color=PALETTE.get(reference, "0.2"))

    axes[0].set_xlim(-0.5, len(ks) - 0.5)
    axes[1].set_xticks(range(len(ks)))
    axes[1].set_xticklabels([str(k) for k in ks], rotation=45, ha="center")
    axes[1].set_xlabel(r"Top $k$ edges", fontsize=16)
    fig.supylabel("Proportion of significant effects", fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False,
               fontsize=11, handletextpad=0.4, columnspacing=1.2)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    return fig
