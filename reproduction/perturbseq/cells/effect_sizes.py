
r"""Effect sizes for the perturbation tests, and the matched-top-k figure.
Self-contained: recomputes the regressions rather than relying on `eff`, on the
*_all_nonzero tuple lists, or on mask_test, any of which may be absent or hold
something else. Needs only what the preceding cell needs, plus at least one W.

  total effect  : y_j ~ D_i + Z                                  (as in cell 50)
  direct effect : y_j ~ D_i + Z + x_{parents(j) \ i}             (as in cell 51)
The coefficient on D_i is the effect size belonging to that test, in standard
deviations of the target gene (data_X_cont_norm is column-standardized).
"""
import os
import numpy as np, pandas as pd, statsmodels.api as sm, matplotlib.pyplot as plt
from tqdm.auto import tqdm

KS        = [100, 200, 400, 700, 1000]
MAX_EDGES = max(KS)     # fit only the top-|w| edges needed by the figure.
                        # Set to None to fit EVERY reported edge -- do that for the
                        # deposited tables, where each p-value should carry its coefficient.
DODGE, N_BOOT = 0.72, 2000
PALETTE = {'RICE': '#D55E00', 'dotears': '#0072B2', 'IBCD': '#009E73', 'inspre': '#CC79A7'}
MARKERS = {'RICE': 'D', 'dotears': 'o', 'IBCD': 's', 'inspre': '^'}
LABELS  = {'RICE': 'RICE', 'dotears': 'DOTEARS', 'IBCD': 'IBCD', 'inspre': 'inspre'}
TITLES  = {'te': 'Total effects', 'de': 'Direct effects (est.)'}

# --- what must already exist ----------------------------------------------------------
# dir() is evaluated HERE, outside any comprehension, for the reason your previous cell notes.
_defined = set(dir()) | set(globals())
_req = {'data_X_cont_norm_test': 19, 'Z_test': 19, 'perturbed_rows': 19, 'unperturbed_rows': 19}
_missing = {k: v for k, v in _req.items() if k not in _defined}
if _missing:
    raise NameError('missing ' + ', '.join(_missing) + ' -- run notebook cell(s) ' +
                    ', '.join(str(c) for c in sorted(set(_missing.values()))) + ' first')

# Methods are whichever W matrices are actually in the session: the loaders for the
# competitors are commented out in some versions of this notebook, and a method that is
# absent should be skipped rather than crash the cell.
_WNAMES = {'RICE': 'W_cpd', 'inspre': 'inspre_W', 'IBCD': 'IBCD_W', 'dotears': 'dotears_w'}
W_BY_METHOD = {m: np.asarray(globals()[v], dtype=float) for m, v in _WNAMES.items()
               if isinstance(globals().get(v), np.ndarray)}
if not W_BY_METHOD:
    raise NameError('no weight matrix found (looked for ' + ', '.join(_WNAMES.values()) +
                    ') -- run the cell that builds W_cpd first')
print('methods found:', ', '.join(f'{m} ({v})' for m, v in _WNAMES.items() if m in W_BY_METHOD))

# --- inputs, normalized to one shape ----------------------------------------------------
Xn   = np.asarray(data_X_cont_norm_test, dtype=float)
Zn   = np.asarray(Z_test, dtype=float)
ctrl = np.asarray(unperturbed_rows, dtype=int)
# perturbed_rows is a dict keyed by gene index in this notebook, but a list elsewhere;
# both are accepted here so the cell does not depend on which.
PR = {int(k): np.asarray(v, dtype=int) for k, v in
      (perturbed_rows.items() if hasattr(perturbed_rows, 'items') else enumerate(perturbed_rows))}
if isinstance(globals().get('mask_test'), np.ndarray):
    D = np.asarray(mask_test, dtype=float)          # use the real indicator when present
else:                                               # otherwise rebuild it from perturbed_rows
    D = np.zeros(Xn.shape)
    for _j, _idx in PR.items():
        if _j < D.shape[1]:
            D[_idx, _j] = 1.0
    print('mask_test absent -- perturbation indicator rebuilt from perturbed_rows')

def fit_pair(i, j, W, conditional):
    idx = PR.get(i)
    if idx is None or len(idx) == 0:
        return None
    comb = np.concatenate((idx, ctrl))
    X = D[comb][:, i].reshape(-1, 1)
    if conditional:
        par = np.where(W[:, j] != 0)[0]
        par = par[par != i]
        X = np.hstack((X, Zn[comb], Xn[comb][:, par]))
    else:
        X = np.hstack((X, Zn[comb]))
    X = sm.add_constant(X)
    y = Xn[comb, j]
    if np.ptp(X[:, 1]) == 0 or len(comb) <= X.shape[1] + 1:
        return None                                  # no contrast, or not enough cells
    m = sm.OLS(y, X).fit(cov_type='HC3')
    return dict(coef=float(m.params[1]), se=float(m.bse[1]), p=float(m.pvalues[1]),
                target_sd=float(np.std(y, ddof=1)), n_perturbed=int(len(idx)),
                n_cells=int(len(comb)))

# --- fit ---------------------------------------------------------------------------------
rows, skipped = [], 0
for method, W in W_BY_METHOD.items():
    ii, jj = np.nonzero(W)
    order = np.argsort(-np.abs(W[ii, jj]))           # rank by |weight|, as the figure does
    if MAX_EDGES is not None:
        order = order[:MAX_EDGES]
    for effect, conditional in (('te', False), ('de', True)):
        for t in tqdm(order, desc=f'{method} {effect}', leave=False):
            i, j = int(ii[t]), int(jj[t])
            r = fit_pair(i, j, W, conditional)
            if r is None:
                skipped += 1
                continue
            rows.append(dict(method=method, effect=effect, i=i, j=j,
                             weight=float(W[i, j]), **r))
eff_tbl = pd.DataFrame(rows)
assert len(eff_tbl), 'no pair could be fitted -- check that perturbed_rows indexes rows of ' \
                     'data_X_cont_norm_test'
eff_tbl['abs_effect'] = eff_tbl.coef.abs()
eff_tbl.to_csv(f'{ctype}_effect_sizes.csv', index=False)
print(f'fitted {len(eff_tbl)} regressions, skipped {skipped}')

# --- median |effect| at matched k ----------------------------------------------------------
def median_ci(v, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

rows = []
for effect in ('te', 'de'):
    for method in W_BY_METHOD:
        sub = eff_tbl[(eff_tbl.effect == effect) & (eff_tbl.method == method)]
        sub = sub.iloc[np.argsort(-sub.weight.abs().values)]
        for k in KS:
            if k > len(sub):
                continue                              # fewer edges than k: no point at all
            med, lo, hi = median_ci(sub.abs_effect.values[:k])
            rows.append(dict(effect=effect, method=method, k=k, n_edges=len(sub),
                             median_abs_effect=med, ci_lower=lo, ci_upper=hi))
es = pd.DataFrame(rows)
n_edges   = eff_tbl[eff_tbl.effect == 'te'].groupby('method').size()
k_matched = min(int(n_edges.min()), max(KS))
print('edges per method:', n_edges.to_dict(), '| like-for-like up to k =', k_matched)

os.makedirs('plot', exist_ok=True)
OUT = f'plot/{ctype}_effect_size_topk.pdf'
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout='constrained')
for ax, effect in zip(axs, ('te', 'de')):
    present = [m for m in W_BY_METHOD if ((es.effect == effect) & (es.method == m)).any()]
    for s_, method in enumerate(present):
        g = es[(es.effect == effect) & (es.method == method)]
        x = np.array([KS.index(k) for k in g.k]) + (s_ - (len(present) - 1) / 2) * DODGE / max(len(present), 1)
        c = PALETTE[method]
        ax.errorbar(x, g.median_abs_effect,
                    yerr=[g.median_abs_effect - g.ci_lower, g.ci_upper - g.median_abs_effect],
                    fmt='none', ecolor=c, elinewidth=1.2, capsize=4, zorder=5)
        ins = (g.k <= k_matched).values
        ax.plot(x[ins], g.median_abs_effect[ins], marker=MARKERS[method], ms=6, ls='none',
                color=c, label=LABELS[method], zorder=6)
        ax.plot(x[~ins], g.median_abs_effect[~ins], marker=MARKERS[method], ms=6, ls='none',
                mfc='white', mec=c, mew=1.4, zorder=6)
    ax.set_xticks(range(len(KS))); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(-0.5, len(KS) - 0.5)
    ax.set_xlabel(r'Top $k$ edges by $|$weight$|$', fontsize=13)
    ax.set_title(TITLES[effect], fontsize=13)
    ax.spines[['top', 'right']].set_visible(False)
axs[0].set_ylabel('Median $|$effect size$|$ (SD of target gene)', fontsize=12)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=len(h), frameon=False, fontsize=11)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
es.to_csv(f'{ctype}_effect_size_topk.csv', index=False)
print(es.round(4).to_string(index=False))
print('wrote', OUT)
# coef is the RAW coefficient: knocking down an activator gives coef < 0. Figure 5A plots
# the NEGATIVE total effect against the weight, so use -coef to match that panel.
