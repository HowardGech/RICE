r"""Which way does the association run? (Follow-up to the balance diagnostics.)

The balance cell found that perturbation identity explains a large share of the
variance in mitopercent and UMI_size. Those covariates are measured AFTER the
perturbation has acted, so the association is equally consistent with
  (i)  the perturbation changing the covariate  -- harmless for instrument validity,
       expected in an essential-gene screen where knockdown impairs growth, or
  (ii) the covariate influencing assignment or uptake -- the reviewer's concern.

These make opposite predictions at the level of individual genes. Under (i) a gene
whose knockdown is STRONGER should show a LARGER covariate shift, because the shift is
its consequence; the association across genes should be ordered by knockdown strength.
Under (ii) there is no reason for the shift to track knockdown strength at all.

For each targeted gene this computes
  knockdown  : mean log expression of the target in its perturbed cells minus controls
               (negative; plotted as magnitude)
  shift      : standardized mean difference of the covariate, perturbed minus controls
and reports the correlation between them, with a bootstrap interval over genes.

CAVEAT, stated because it changes which panel is evidence. The knockdown is computed on
log(1 + X / Offset) and Offset IS UMI_size, so a gene that lowers total RNA inflates its
own apparent knockdown: the UMI_size panel is partly circular and should be read as
indicative only. mitopercent does not enter the normalisation, so that panel is clean,
and it is also the covariate with the large association. Judge the question on it.

Needs, from cell 1: adata_real_inc, mask, gene_name_targeting, data_X, Offset.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

MIN_CELLS = 30
CONT      = ['mitopercent', 'UMI_size']        # clean one first
N_BOOT    = 2000
OUT_CSV   = f'{ctype}_covariate_direction.csv'
OUT_FIG   = f'{ctype}_covariate_direction.pdf'

obs  = adata_real_inc.obs
gid  = obs['gene_id'].values.astype(str)
ctrl = gid == 'non-targeting'
logX = np.log1p(np.asarray(data_X) / np.asarray(Offset))
M    = np.asarray(mask)
rng  = np.random.default_rng(0)

recs = []
for j, gname in enumerate(gene_name_targeting):
    pj = M[:, j]
    if pj.sum() < MIN_CELLS:
        continue
    r = dict(gene=str(gname), n_cells=int(pj.sum()),
             knockdown=float(logX[pj, j].mean() - logX[ctrl, j].mean()))
    for cv in CONT:
        x = obs[cv].values.astype(float)
        xg, xc = x[pj], x[ctrl]
        sp = np.sqrt((xg.var(ddof=1) + xc.var(ddof=1)) / 2)
        r[f'{cv}_shift'] = (xg.mean() - xc.mean()) / sp if sp > 0 else np.nan
    recs.append(r)

tab = pd.DataFrame(recs)
tab['knockdown_strength'] = -tab.knockdown          # positive = stronger knockdown
tab.to_csv(OUT_CSV, index=False)
print(f'{ctype}: {len(tab)} genes with at least {MIN_CELLS} perturbed cells')
print(f'   knockdown strength: median {tab.knockdown_strength.median():.3f}, '
      f'range {tab.knockdown_strength.min():.3f} to {tab.knockdown_strength.max():.3f} (log)\n')

fig, axes = plt.subplots(1, len(CONT), figsize=(5.2 * len(CONT), 4.4))
axes = np.atleast_1d(axes)
for ax, cv in zip(axes, CONT):
    d = tab[['knockdown_strength', f'{cv}_shift']].dropna()
    xk, ys = d.knockdown_strength.values, d[f'{cv}_shift'].values
    rho, p_rho = stats.spearmanr(xk, ys)
    rp = np.corrcoef(xk, ys)[0, 1]
    boot = [stats.spearmanr(*np.array([xk, ys])[:, rng.integers(0, len(xk), len(xk))])[0]
            for _ in range(N_BOOT)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    sl, ic = np.polyfit(xk, ys, 1)
    note = ' (partly circular -- see header)' if cv == 'UMI_size' else ''
    print(f'{cv}{note}')
    print(f'   Spearman rho {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]   Pearson {rp:+.3f}')
    print(f'   slope {sl:+.3f} SD of {cv} per unit of log knockdown')
    print(f'   genes in the strongest knockdown quartile shift '
          f'{d[xk >= np.quantile(xk, 0.75)][f"{cv}_shift"].mean():+.3f} SD, '
          f'weakest quartile {d[xk <= np.quantile(xk, 0.25)][f"{cv}_shift"].mean():+.3f} SD\n')

    ax.scatter(xk, ys, s=14, alpha=0.5, color='#D55E00' if cv == 'mitopercent' else 'grey',
               edgecolor='none')
    xs = np.linspace(xk.min(), xk.max(), 50)
    ax.plot(xs, sl * xs + ic, color='black', lw=1.2)
    ax.axhline(0, color='0.6', lw=0.8, ls='--')
    ax.set_xlabel('Knockdown strength (log)', fontsize=12)
    ax.set_ylabel(f'{cv} shift vs control (SD)', fontsize=12)
    ax.set_title(f'{cv}   ' + r'$\rho$' + f' = {rho:+.2f} [{lo:+.2f}, {hi:+.2f}]',
                 fontsize=12)
    ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
print(f'wrote {OUT_CSV} and {OUT_FIG}')
print('Reading it: a clearly POSITIVE rho on mitopercent means the covariate shift is '
      'ordered by how strongly each gene was knocked down, i.e. the perturbation is '
      'moving the covariate -- the association does not indicate non-random assignment. '
      'A rho whose interval covers zero leaves the direction unresolved and the '
      'reverse-causation argument should not be made on this evidence.')
