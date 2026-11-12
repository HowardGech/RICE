
r"""How well does each method's edge RANKING align with the measured ATE?

For each method, take its top k edges by |W| (IBCD by PIP if IBCD_pip is set) and plot
the fraction whose ATE is significant at FDR. Sweeping k rather than reporting one
number is the point: methods differ in how many edges they report, so a single
thresholded graph confounds ranking quality with sparsity.

The dashed line is the base rate -- the fraction of ALL estimable pairs with a
significant ATE -- so a curve above it means the ranking carries information about
total effects, and the vertical gap is the enrichment.

Filled markers: k is within the method's edge count. Open marker: the method's full
edge count, past which the curve cannot be extended.

Needs {ctype}_ATE.npz for each dataset in DATASETS and the four weight matrices for
each. Provide them as WEIGHTS = {dataset: {method: W}}; if a single dataset is loaded,
the fallback below picks up W_cpd / dotears_w / IBCD_W / inspre_W for `ctype`.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt

FDR      = 0.05
KS       = [50, 100, 200, 400, 700, 1000, 2000, 4000]
OUT      = 'plot/ate_alignment.pdf'
PALETTE  = {'RICE': '#D55E00', 'DOTEARS': '#0072B2', 'IBCD': '#009E73', 'inspre': '#CC79A7'}
MARKERS  = {'RICE': 'D', 'DOTEARS': 'o', 'IBCD': 's', 'inspre': '^'}
IBCD_pip = globals().get('IBCD_pip', None)

WEIGHTS = globals().get('WEIGHTS') or {ctype: {
    'RICE': W_cpd, 'DOTEARS': dotears_w, 'IBCD': IBCD_W, 'inspre': inspre_W}}
DATASETS = list(WEIGHTS)

rows, base = [], {}
for ds in DATASETS:
    Z = np.load(f'{ds}_ATE.npz', allow_pickle=True)
    ate, q = Z['ate'], Z['q']
    msr = np.isfinite(ate)
    sig = msr & (q < FDR)
    base[ds] = float(sig.sum() / msr.sum())
    for m, W in WEIGHTS[ds].items():
        Wm = np.asarray(W, dtype=float)
        E = (Wm != 0) & msr
        np.fill_diagonal(E, False)
        ii, jj = np.where(E)
        if m == 'IBCD' and IBCD_pip is not None:
            key = np.lexsort((np.abs(Wm[ii, jj]), np.asarray(IBCD_pip)[ii, jj]))[::-1]
        else:
            key = np.argsort(-np.abs(Wm[ii, jj]))
        ii, jj = ii[key], jj[key]
        hit = sig[ii, jj]
        n_edges = len(ii)
        for k in KS:
            take = min(k, n_edges)
            if take == 0:
                continue
            rows.append(dict(dataset=ds, method=m, k=k, n_used=take,
                             truncated=take < k, n_edges=n_edges,
                             precision=float(hit[:take].mean()),
                             recall=float(hit[:take].sum() / sig.sum()),
                             enrichment=float(hit[:take].mean() / base[ds])))
align = pd.DataFrame(rows).drop_duplicates(['dataset', 'method', 'n_used'])
align.to_csv('ate_alignment.csv', index=False)

fig, axs = plt.subplots(1, len(DATASETS), figsize=(5.4 * len(DATASETS), 4.2),
                        sharey=True, layout='constrained')
axs = np.atleast_1d(axs)
for ax, ds in zip(axs, DATASETS):
    for m in ['RICE', 'DOTEARS', 'IBCD', 'inspre']:
        g = align[(align.dataset == ds) & (align.method == m)].sort_values('n_used')
        if not len(g):
            continue
        c = PALETTE[m]
        ax.plot(g.n_used, g.precision, ls=':', lw=1.1, color=c, zorder=5)
        for x, y, tr in zip(g.n_used, g.precision, g.truncated):
            ax.plot(x, y, marker=MARKERS[m], ms=6, ls='none', color=c, mew=1.2,
                    mfc='white' if tr else c, mec=c, zorder=6,
                    label=m if (x == g.n_used.iloc[0] and ds == DATASETS[0]) else None)
    ax.axhline(base[ds], ls='--', lw=1, color='0.45', zorder=3)
    ax.text(0.98, base[ds], ' base rate', transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', fontsize=9, color='0.45')
    ax.set_xscale('log')
    ax.set_title(ds, fontsize=15, y=1.02)
    ax.spines.top.set_visible(False); ax.spines.right.set_visible(False)
    ax.tick_params(axis='both', labelsize=12)
axs[0].set_ylim(0, 1)
fig.supxlabel('Top $k$ edges of each method', fontsize=14)
fig.supylabel('Fraction with a significant ATE', fontsize=14)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=4, frameon=False, fontsize=11,
           handletextpad=0.4, columnspacing=1.6)
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print('base rate (fraction of estimable pairs with a significant ATE):',
      {d: round(v, 4) for d, v in base.items()})
print(align.pivot_table(index=['dataset', 'k'], columns='method',
                        values='precision').round(3).to_string())
print('wrote', OUT)
