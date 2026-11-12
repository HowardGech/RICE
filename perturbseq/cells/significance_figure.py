
r"""Top-k significance figure, both datasets, read from the per-method CSVs written by
write_significance_csvs.py: {ctype}_{method}_significance_{te,de}.csv (success, total).

Those CSVs carry COUNTS only, which is enough for the proportions, the Wilson
intervals and Fisher's exact test. The paired cluster bootstrap needs edge-level
data, so if the edge-list CSVs and p-value npz files are also present this cell reads
them and uses the bootstrap instead -- Fisher treats the two top-k sets as
independent samples, and they overlap and share target genes, so it overstates
significance. The printout says which test the brackets came from.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import fisher_exact, binomtest
from statsmodels.stats.proportion import proportion_confint

DATASETS = ['RPE1', 'K562']           # the {ctype} values used when the CSVs were written
KS       = [100, 200, 400, 700, 1000] # must match `thresholds` in write_significance_csvs.py
REF      = 'scRISE'                   # every bracket is REF vs one competitor
OUT      = 'plot/significant_proportion_topk.pdf'
SHOW_NS  = False
DODGE    = 0.72
N_BOOT   = 2000
ALPHA    = 0.01                       # only used by the bootstrap; must equal SIG there

# file stem -> label on the plot
METHODS = {'scRISE': 'RICE', 'dotears': 'DOTEARS', 'IBCD': 'IBCD', 'inspre': 'inspre'}
# edge-list CSVs (cell 31) and npz keys (cell 67), for the bootstrap only
EDGE_CSV = {'scRISE': '{ds}_scRISE_all_nonzero_gene_pairs_normalize.csv',
            'IBCD':   '{ds}_IBCD_all_nonzero_gene_pairs.csv',
            'inspre': '{ds}_inspre_all_nonzero_gene_pairs.csv',
            'dotears': '{ds}_dotears_all_nonzero_gene_pairs.csv'}
NPZ_KEY  = {'scRISE': 'RICE', 'IBCD': 'IBCD', 'inspre': 'inspre', 'dotears': 'dotears'}

PALETTE = {'RICE': '#D55E00', 'DOTEARS': '#0072B2', 'IBCD': '#009E73', 'inspre': '#CC79A7'}
MARKERS = {'RICE': 'D', 'DOTEARS': 'o', 'IBCD': 's', 'inspre': '^'}
TITLES  = {'Total': 'Total effects', 'Direct': 'Direct effects (est.)'}
ref_lab = METHODS[REF]

# --- 1. proportions, straight from the significance CSVs ---------------------------
rows = []
for ds in DATASETS:
    for stem, lab in METHODS.items():
        for effect, tag in (('Total', 'te'), ('Direct', 'de')):
            f = f'{ds}_{stem}_significance_{tag}.csv'
            if not os.path.exists(f):
                raise FileNotFoundError(f'{f} -- run write_significance_csvs.py for {ds}')
            d = pd.read_csv(f)
            assert len(d) == len(KS), f'{f}: {len(d)} rows, expected {len(KS)} for KS={KS}'
            for k, (_, r) in zip(KS, d.iterrows()):
                lo, hi = proportion_confint(int(r.success), int(r.total), alpha=0.05,
                                            method='wilson')
                rows.append(dict(dataset=ds, method=lab, effect=effect, k=k,
                                 success=int(r.success), total=int(r.total),
                                 proportion=r.success / r.total, lower=lo, upper=hi,
                                 truncated=int(r.total) < k))
summary = pd.DataFrame(rows)
summary['yerr_lower'] = summary.proportion - summary.lower
summary['yerr_upper'] = summary.upper - summary.proportion
for ds in DATASETS:
    s = summary[summary.dataset == ds]
    n_at_max = s[s.k == max(KS)].set_index('method').total
    print(f'{ds}: edges available at k={max(KS)} {n_at_max.to_dict()}; '
          f'matched up to k = {int(n_at_max.min())}')

# --- 2. edge-level data, only if present, for the paired cluster bootstrap ----------
edge = {}
for ds in DATASETS:
    try:
        PT = np.load(f'{ds}_pvalues_te.npz', allow_pickle=True)
        PD = np.load(f'{ds}_pvalues_de.npz', allow_pickle=True)
        for stem, lab in METHODS.items():
            e = pd.read_csv(EDGE_CSV[stem].format(ds=ds))
            ij = list(zip(e.source_idx.astype(int), e.target_idx.astype(int)))
            for effect, PV in (('Total', PT), ('Direct', PD)):
                p = np.asarray(PV[NPZ_KEY[stem]], dtype=float).ravel()
                if len(p) != len(ij):
                    raise ValueError(f'{ds}/{stem}/{effect}: {len(p)} p-values vs '
                                     f'{len(ij)} edges')
                order = np.argsort(-np.abs(e.weight.values))
                edge[(ds, lab, effect)] = (p[order] < ALPHA,
                                           np.array([ij[t][1] for t in order]))  # target gene
    except (FileNotFoundError, ValueError, KeyError) as err:
        print(f'[warn] {ds}: edge-level data unavailable ({err}); falling back to Fisher')
        edge = {}
        break
use_boot = bool(edge)

# --- 3. REF vs each competitor at matched k ----------------------------------------
rng = np.random.default_rng(0)
tests = []
for (ds, effect, k), g in summary.groupby(['dataset', 'effect', 'k']):
    g = g.set_index('method'); ref = g.loc[ref_lab]
    for m, r in g.iterrows():
        if m == ref_lab:
            continue
        p_f = fisher_exact([[int(ref.success), int(ref.total - ref.success)],
                            [int(r.success),   int(r.total - r.success)]],
                           alternative='greater')[1]
        p_b, lo_b, hi_b = np.nan, np.nan, np.nan
        if use_boot:
            sel = {}
            for n_ in (ref_lab, m):
                hits, tgt = edge[(ds, n_, effect)]
                take = min(k, len(hits))
                sel[n_] = (hits[:take], tgt[:take])
            pool = np.unique(np.concatenate([t for _, t in sel.values()]))
            members = {n_: {c: h[t == c] for c in pool} for n_, (h, t) in sel.items()}
            boot = np.empty(N_BOOT)
            for b in range(N_BOOT):
                draw = pool[rng.integers(0, len(pool), len(pool))]
                vals = {}
                for n_ in (ref_lab, m):
                    hh = [members[n_][c] for c in draw if members[n_][c].size]
                    vals[n_] = np.concatenate(hh).mean() if hh else np.nan
                boot[b] = vals[ref_lab] - vals[m]
            boot = boot[~np.isnan(boot)]
            p_b = (1 + int((boot <= 0).sum())) / (len(boot) + 1)
            lo_b, hi_b = np.percentile(boot, [2.5, 97.5])
        tests.append(dict(dataset=ds, effect=effect, k=k, method=m,
                          proportion=r.proportion, diff=ref.proportion - r.proportion,
                          p_fisher=p_f, p_bootstrap=p_b, ci_lower=lo_b, ci_upper=hi_b,
                          p_ref_greater=(p_b if use_boot else p_f),
                          p_vs_chance=binomtest(int(r.success), int(r.total), 0.5,
                                                alternative='greater').pvalue,
                          matched=bool(not r.truncated and not ref.truncated)))
tests = pd.DataFrame(tests)
stars = lambda p: ('ns' if not np.isfinite(p) or p >= .05 else '****' if p < 1e-4 else
                   '***' if p < 1e-3 else '**' if p < 1e-2 else '*')
tests['stars'] = tests.p_ref_greater.map(stars)
tests['test_used'] = ('paired cluster bootstrap over target genes (one-sided)'
                      if use_boot else 'Fisher exact, one-sided (ASSUMES INDEPENDENCE)')
print(f'[test] brackets: {tests.test_used.iloc[0]}; alternative = {ref_lab} greater')
if not use_boot:
    print('[warn] Fisher treats the two top-k sets as independent samples; they overlap '
          'and share target genes, so these p-values OVERSTATE significance.')

# --- 4. figure: rows = effect, columns = dataset -----------------------------------
order_m = [ref_lab] + [METHODS[s] for s in METHODS if METHODS[s] != ref_lab]
off = {m: (i - (len(order_m) - 1) / 2) * (DODGE / (len(order_m) - 1))
       for i, m in enumerate(order_m)}
xpos = {k: i for i, k in enumerate(KS)}

fig, axs = plt.subplots(2, len(DATASETS), figsize=(6.6 * len(DATASETS), 7.8),
                        sharex=True, sharey='row', layout='constrained')
axs = np.atleast_2d(axs)
fig.get_layout_engine().set(hspace=0.07, wspace=0.04)

for i, effect in enumerate(['Total', 'Direct']):
    row = summary[summary.effect == effect]
    span = row.upper.max() - row.lower.min()
    step, bar = 0.085 * span, 0.018 * span
    row_tops = [row.upper.max()]
    for j, ds in enumerate(DATASETS):
        ax = axs[i, j]
        sub = row[row.dataset == ds]
        for m in order_m:
            g = sub[sub.method == m].sort_values('k')
            x, c = [xpos[k] + off[m] for k in g.k], PALETTE.get(m, '0.3')
            ax.errorbar(x, g.proportion, yerr=[g.yerr_lower, g.yerr_upper], fmt='none',
                        ecolor=c, elinewidth=1.2, capsize=4, zorder=5)
            # dotted connector: the trend across k for this method, drawn through its
            # dodged positions so it stays inside its own series
            ax.plot(x, g.proportion, ls=':', lw=1.1, color=c, zorder=6)
            for xi, yi, tr in zip(x, g.proportion, g.truncated):
                ax.plot(xi, yi, marker=MARKERS.get(m, 'o'), ms=6, ls='none', color=c,
                        mfc='white' if tr else c, mec=c, mew=1.2, zorder=7,
                        label=m if (xi == x[0] and i == 0 and j == 0) else None)
        t = tests[(tests.effect == effect) & (tests.dataset == ds)]
        for k in KS:
            base = sub[sub.k == k].upper.max()
            level = 0
            for m in order_m[1:]:
                r_ = t[(t.k == k) & (t.method == m)].iloc[0]
                if r_.stars == 'ns' and not SHOW_NS:
                    continue
                y = base + step * (level + 1)
                x1, x2 = xpos[k] + off[ref_lab], xpos[k] + off[m]
                ax.plot([x1, x1, x2, x2], [y, y + bar, y + bar, y], lw=1.0, c='0.25',
                        solid_joinstyle='miter', clip_on=False, zorder=8)
                ax.text((x1 + x2) / 2, y + bar, r_.stars, ha='center', va='bottom',
                        fontsize=11, color='0.25', zorder=8)
                row_tops.append(y + bar + 0.045 * span)
                level += 1
        ax.spines.top.set_visible(False); ax.spines.right.set_visible(False)
        ax.set_xlabel(''); ax.tick_params(axis='both', labelsize=13)
        if i == 0:
            ax.set_title(ds, fontsize=17, y=1.03)
        ax.set_ylabel(TITLES[effect] if j == 0 else '', fontsize=15)
    axs[i, 0].set_ylim(top=max(row_tops))
    ylo, yhi = axs[i, 0].get_ylim()
    axs[i, 0].set_yticks([t_ for t_ in axs[i, 0].get_yticks() if ylo <= t_ <= 1.0 + 1e-9])
    axs[i, 0].set_ylim(ylo, yhi)

axs[0, 0].set_xlim(-0.5, len(KS) - 0.5)
for ax in axs[-1]:
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([str(k) for k in KS], fontsize=14)
fig.supxlabel(r'Top $k$ edges', fontsize=16)
fig.supylabel('Proportion of significant effects', fontsize=16)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=len(order_m), frameon=False,
           fontsize=13, handletextpad=0.4, columnspacing=1.6)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
summary.to_csv('topk_summary.csv', index=False)
tests.to_csv('topk_tests.csv', index=False)
print('wrote', OUT)
