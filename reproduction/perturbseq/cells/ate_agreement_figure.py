
r"""Compare the ATE-agreement metrics across the four methods, both cell lines.
Reads {ds}_ate_method_agreement.csv for each dataset.

Metrics are grouped by SCALE into six groups, because a correlation, a proportion, an
odds ratio and a count cannot share an axis. Each group gets a COLUMN OF TWO PANELS,
one cell line above the other on a shared y scale, so a method's bar can be compared
vertically between cell lines:

  (A) direct agreement      precision / recall / MCC, method edge vs significant ATE
  (B) reachable agreement   the same three on the descendant relation
  (C) enrichment            odds ratios, log axis (infinite = no false positive)
  (D) rank agreement        Spearman of |W| against |ATE|, two ways
  (E) orientation           fraction of one-way-significant pairs oriented the same way

Dropped from the figure at the author's request, still written to the CSV by the
agreement cell: median_spearman_per_source, direct_n_pred / reach_n_pred (set sizes),
and precision_at_matched_k / auc_edges_vs_ate_t / auc_reach_vs_ate_t.

Read (A) with the mediation caveat: low direct recall is expected, because a mediated
pair is a genuine total effect with no direct edge. Precision is the column that speaks
to whether a method's edges are supported. Read (B) together with (F): a method can
reach high recall on total effects simply by connecting everything.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt

DATASETS = globals().get('DATASETS') or ['RPE1', 'K562']   # top row first
PALETTE  = {'RICE': '#D55E00', 'DOTEARS': '#0072B2', 'IBCD': '#009E73', 'inspre': '#CC79A7'}
ORDER    = ['RICE', 'DOTEARS', 'IBCD', 'inspre']
OUT      = 'plot/ate_agreement.pdf'

PANELS = [
    ('Direct edge vs significant ATE',
     [('direct_precision', 'Precision'), ('direct_recall', 'Recall'),
      ('direct_mcc', 'MCC')], 'linear', (0, None)),
    ('Reachable (descendant) vs significant ATE',
     [('reach_precision', 'Precision'), ('reach_recall', 'Recall'),
      ('reach_mcc', 'MCC')], 'linear', (0, None)),
    ('Enrichment (odds ratio)',
     [('direct_odds_ratio', 'Direct'), ('reach_odds_ratio', 'Reachable')], 'log', None),
    ('Rank agreement with $|$ATE$|$ (Spearman)',
     [('spearman_on_edges', "On the\nmethod's edges"),
      ('spearman_all_pairs', 'All\npairs')], 'linear', None),
    ('Orientation of one-way-significant pairs',
     [('orient_frac_agree', 'Fraction agreeing')], 'linear', (0, 1)),
]

TAB = {ds: pd.read_csv(f'{ds}_ate_method_agreement.csv').set_index('method')
       for ds in DATASETS}
methods = [m for m in ORDER if m in TAB[DATASETS[0]].index]
nD, nM = len(DATASETS), len(methods)
NCOL = 3
nrow = nD * int(np.ceil(len(PANELS) / NCOL))

fig, axs = plt.subplots(nrow, NCOL, figsize=(4.9 * NCOL, 2.9 * nrow), layout='constrained')
fig.get_layout_engine().set(hspace=0.03)
axs = np.atleast_2d(axs)

for gi, (title, cols, scale, ylim) in enumerate(PANELS):
    block, col = divmod(gi, NCOL)
    # y range shared by the two cell lines of this group, so bars compare vertically
    allv = np.array([[float(TAB[ds].loc[m, c]) if c in TAB[ds].columns else np.nan
                      for m in methods for c, _ in cols] for ds in DATASETS]).ravel()
    finite = allv[np.isfinite(allv) & (allv > 0)]
    cap = (finite.max() * 3 if len(finite) else 10) if scale == 'log' else None
    if scale == 'linear':
        # a None bound is taken from the data across BOTH cell lines, so the two panels
        # of a group still share one scale and compare vertically
        lo_d, hi_d = np.nanmin(allv), np.nanmax(allv)
        pad = 0.08 * max(hi_d - lo_d, 1e-9)
        lo, hi = (ylim or (None, None))
        ylim = (min(0, lo_d - pad) if lo is None else lo,
                hi_d + pad if hi is None else hi)

    for si, ds in enumerate(DATASETS):
        ax = axs[block * nD + si, col]
        w = 0.8 / nM
        for ti, (c, lab) in enumerate(cols):
            for mi, m in enumerate(methods):
                v = float(TAB[ds].loc[m, c]) if c in TAB[ds].columns else np.nan
                x = ti + (mi - (nM - 1) / 2) * w
                plotted = cap if (scale == 'log' and np.isinf(v)) else v
                ax.bar(x, plotted, width=w * 0.9, color=PALETTE.get(m, '0.5'),
                       edgecolor='white', linewidth=0.5,
                       label=m if (gi == 0 and si == 0 and ti == 0) else None)
                if np.isinf(v):
                    ax.text(x, plotted, r'$\infty$', ha='center', va='bottom', fontsize=8)
                elif np.isnan(v):
                    ax.text(x, 0, 'n/a', ha='center', va='bottom', fontsize=7, color='0.5')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([lab for _, lab in cols] if si == nD - 1 else [''] * len(cols),
                           fontsize=10)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_yscale(scale)
        if ylim and scale == 'linear':
            ax.set_ylim(*ylim)
        if scale == 'linear' and ylim and ylim[0] < 0:
            ax.axhline(0, lw=0.8, color='0.6')
        if title.startswith('Enrichment'):
            ax.axhline(1, ls='--', lw=1, color='0.45')      # 1 = no enrichment
        if si == 0:
            ax.set_title(title, fontsize=12, y=1.03)
        if col == 0:
            ax.set_ylabel(ds, fontsize=13)
        else:
            ax.set_ylabel('')
        ax.spines.top.set_visible(False); ax.spines.right.set_visible(False)
        ax.tick_params(axis='y', labelsize=9)

used = {(gi // NCOL * nD + si, gi % NCOL)
        for gi in range(len(PANELS)) for si in range(nD)}
for r in range(axs.shape[0]):
    for c in range(axs.shape[1]):
        if (r, c) not in used:
            axs[r, c].set_visible(False)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=nM, frameon=False, fontsize=12,
           handletextpad=0.5, columnspacing=1.8)
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print('wrote', OUT, '| grid:', axs.shape)
for ds in DATASETS:
    print(f'\n== {ds} ==')
    print(TAB[ds].loc[methods, ['direct_precision', 'direct_recall', 'direct_mcc',
                                'reach_precision', 'reach_recall', 'reach_mcc',
                                'spearman_on_edges', 'orient_frac_agree']]
          .round(3).to_string())
