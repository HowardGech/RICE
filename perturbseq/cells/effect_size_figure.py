
r"""Effect sizes at matched top-k, both cell lines, drawn to match
significant_proportion_topk.pdf: columns are cell lines, rows are effect types,
shared y within a row, brackets above. NO connector lines between k values -- the
reference PDF still has them, but they imply a continuous trend across a discrete,
nested axis (the top 200 edges contain the top 100), so the points stand alone.
Reads {ctype}_effect_sizes.csv -- no session objects, no W matrices.

Companion figure to the significance one: same panels, same k grid, same palette.
That figure asks how OFTEN a top-k edge is supported; this asks how LARGE the
supporting effect is.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt

DATASETS = ['RPE1', 'K562']      # same order as the significance figure
KS       = [100, 200, 400, 700, 1000]
REF      = 'RICE'                # every bracket is REF vs one competitor
SHOW_NS  = False
DODGE    = 0.72
N_BOOT   = 2000
QUOTE_K  = 400
PALETTE  = {'RICE': '#D55E00', 'DOTEARS': '#0072B2', 'IBCD': '#009E73', 'inspre': '#CC79A7'}
MARKERS  = {'RICE': 'D', 'DOTEARS': 'o', 'IBCD': 's', 'inspre': '^'}
LABELS   = {'RICE': 'RICE', 'dotears': 'DOTEARS', 'IBCD': 'IBCD', 'inspre': 'inspre'}
TITLES   = {'te': 'Total effects', 'de': 'Direct effects (est.)'}
OUT      = 'plot/effect_size_topk.pdf'
os.makedirs('plot', exist_ok=True)

def weighted_median(v_sorted, counts_per_value):
    """Median of values repeated by integer weights, vectorised over bootstrap draws."""
    cw = np.cumsum(counts_per_value, axis=1)
    idx = (cw >= cw[:, -1:] / 2).argmax(axis=1)
    return v_sorted[idx]

def paired_cluster_bootstrap(vals_a, clus_a, vals_b, clus_b, n_boot=N_BOOT, seed=0):
    """One-sided P(median |effect| of A exceeds that of B), resampling TARGET GENES.

    The two top-k sets overlap and several edges share a target gene, so an unpaired
    rank test would treat correlated observations as independent. Resampling whole
    target genes, with the same draw applied to both methods, respects both.
    """
    pool = np.unique(np.concatenate([clus_a, clus_b]))
    pos = {c: i for i, c in enumerate(pool)}
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(pool), np.full(len(pool), 1 / len(pool)), size=n_boot)
    out = []
    for vals, clus in ((vals_a, clus_a), (vals_b, clus_b)):
        o = np.argsort(vals)
        v, c = vals[o], np.array([pos[x] for x in clus[o]])
        out.append(weighted_median(v, counts[:, c]))
    diff = out[0] - out[1]
    return float(np.median(vals_a) - np.median(vals_b)), float(np.mean(diff <= 0))

def stars(p):
    return ('****' if p < 1e-4 else '***' if p < 1e-3 else
            '**' if p < 1e-2 else '*' if p < 0.05 else 'ns')

# --- load, rank, summarise -------------------------------------------------------------
summary, tests, loaded, kmatch, topk = [], [], [], {}, {}
for ds in DATASETS:
    fn = f'{ds}_effect_sizes.csv'
    if not os.path.exists(fn):
        print(f'{fn} not found -- {ds} omitted'); continue
    t = pd.read_csv(fn)
    t['abs_effect'] = t.coef.abs()
    t['label'] = t.method.map(lambda m: LABELS.get(m, m))
    loaded.append(ds)
    n_edges = t[t.effect == 'te'].groupby('label').size()
    kmatch[ds] = min(int(n_edges.min()), max(KS))
    print(f'{ds}: {n_edges.to_dict()} | like-for-like up to k = {kmatch[ds]}')
    for effect in ('te', 'de'):
        for m in n_edges.index:
            sub = t[(t.effect == effect) & (t.label == m)]
            sub = sub.iloc[np.argsort(-sub.weight.abs().values)]
            for k in KS:
                if k > len(sub):
                    continue
                v = sub.abs_effect.values[:k]
                topk[(ds, effect, m, k)] = (v, sub.j.values[:k])
                rng = np.random.default_rng(0)
                b = np.median(rng.choice(v, size=(N_BOOT, len(v)), replace=True), axis=1)
                summary.append(dict(dataset=ds, effect=effect, method=m, k=k,
                                    n_edges=len(sub), median=float(np.median(v)),
                                    lower=float(np.percentile(b, 2.5)),
                                    upper=float(np.percentile(b, 97.5)),
                                    truncated=k > kmatch[ds]))
summary = pd.DataFrame(summary)
assert len(summary), 'no {ctype}_effect_sizes.csv found -- run the effect-size cell first'
summary['yerr_lower'] = summary['median'] - summary.lower
summary['yerr_upper'] = summary.upper - summary['median']

for (ds, effect, k), _ in summary.groupby(['dataset', 'effect', 'k']):
    if (ds, effect, REF, k) not in topk:
        continue
    va, ca = topk[(ds, effect, REF, k)]
    for m in [x for x in summary.method.unique() if x != REF]:
        if (ds, effect, m, k) not in topk:
            continue
        vb, cb = topk[(ds, effect, m, k)]
        d, p = paired_cluster_bootstrap(va, ca, vb, cb)
        tests.append(dict(dataset=ds, effect=effect, k=k, method=m, diff_median=d,
                          p_bootstrap_greater=p, stars=stars(p), n_boot=N_BOOT))
tests = pd.DataFrame(tests)
print(f'[test] brackets: paired cluster bootstrap over target genes, one-sided '
      f'({REF} greater); resolution {1/N_BOOT:g}')

# --- figure ------------------------------------------------------------------------------
order_m = [REF] + [m for m in summary.method.unique() if m != REF]
off = {m: (i - (len(order_m) - 1) / 2) * (DODGE / max(len(order_m) - 1, 1))
       for i, m in enumerate(order_m)}
xpos = {k: i for i, k in enumerate(KS)}

fig, axs = plt.subplots(2, len(loaded), figsize=(6.6 * len(loaded), 7.8),
                        sharex=True, sharey='row', layout='constrained')
axs = np.atleast_2d(axs).reshape(2, len(loaded))
fig.get_layout_engine().set(hspace=0.07, wspace=0.04)

for i, effect in enumerate(['te', 'de']):
    row = summary[summary.effect == effect]
    span = row.upper.max() - row.lower.min()
    step, bar = 0.085 * span, 0.018 * span
    row_tops = [row.upper.max()]
    for j, ds in enumerate(loaded):
        ax = axs[i, j]
        sub = row[row.dataset == ds]
        for m in order_m:
            g = sub[sub.method == m].sort_values('k')
            if g.empty:
                continue
            x, c = [xpos[k] + off[m] for k in g.k], PALETTE.get(m, '0.3')
            ax.errorbar(x, g['median'], yerr=[g.yerr_lower, g.yerr_upper], fmt='none',
                        ecolor=c, elinewidth=1.2, capsize=4, zorder=5)
            for xi, yi, tr in zip(x, g['median'], g.truncated):
                ax.plot(xi, yi, marker=MARKERS.get(m, 'o'), ms=6, ls='none', color=c,
                        mfc='white' if tr else c, mec=c, mew=1.2, zorder=7,
                        label=m if (xi == x[0] and i == 0 and j == 0) else None)
        tt = tests[(tests.effect == effect) & (tests.dataset == ds)] if len(tests) else tests
        for k in KS:
            if not (sub.k == k).any():
                continue
            base, level = sub[sub.k == k].upper.max(), 0
            for m in order_m[1:]:
                r_ = tt[(tt.k == k) & (tt.method == m)]
                if r_.empty or (r_.iloc[0].stars == 'ns' and not SHOW_NS):
                    continue
                y = base + step * (level + 1)
                x1, x2 = xpos[k] + off[REF], xpos[k] + off[m]
                ax.plot([x1, x1, x2, x2], [y, y + bar, y + bar, y], lw=1.0, c='0.25',
                        solid_joinstyle='miter', clip_on=False, zorder=8)
                ax.text((x1 + x2) / 2, y + bar, r_.iloc[0].stars, ha='center', va='bottom',
                        fontsize=11, color='0.25', zorder=8)
                row_tops.append(y + bar + 0.045 * span)
                level += 1
        ax.spines.top.set_visible(False); ax.spines.right.set_visible(False)
        ax.set_xlabel(''); ax.tick_params(axis='both', labelsize=13)
        if i == 0:
            ax.set_title(ds, fontsize=17, y=1.03)
        ax.set_ylabel(TITLES[effect] if j == 0 else '', fontsize=15)
    axs[i, 0].set_ylim(top=max(row_tops))

axs[0, 0].set_xlim(-0.5, len(KS) - 0.5)
for ax in axs[-1]:
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([str(k) for k in KS], fontsize=14)
fig.supxlabel(r'Top $k$ edges', fontsize=16)
fig.supylabel('Median $|$effect size$|$ (SD of target gene)', fontsize=16)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=len(order_m), frameon=False,
           fontsize=13, handletextpad=0.4, columnspacing=1.6)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
summary.to_csv('effect_size_topk_summary.csv', index=False)
tests.to_csv('effect_size_topk_tests.csv', index=False)
print(f'\nmedian |effect size| at k = {QUOTE_K} (SD of target gene)')
print(summary[summary.k == QUOTE_K].pivot_table(index=['dataset', 'effect'],
                                                columns='method', values='median').round(3).to_string())
print('wrote', OUT)
