
r"""Vanilla average treatment effects for every (perturbed gene i -> gene j) pair.

Same regression as the total-effect p-value loop (gene j on a perturbation indicator
for gene i plus Z, over the cells perturbed for i together with the unperturbed
cells), but run for ALL pairs and keeping the COEFFICIENT as well as the p-value:

    ATE(i -> j) = E[x_j | perturb i] - E[x_j | no perturbation]

a TOTAL effect -- every path from i to j contributes -- not the direct effect w_ij.

Uses the objects already in the notebook: data_X_cont_norm_test, Z_test,
perturbed_rows, unperturbed_rows, gene_name_targeting, ctype.
Run it once per ctype; the summary covers every *_ATE_long.csv present, so after the
second run it spans both cell types.
"""
import os, glob
import numpy as np, pandas as pd
from scipy import stats

MIN_CELLS = 10        # skip a perturbation with fewer cells than this
FDR       = 0.05
OUTDIR    = '.'       # where the ATE files are written

Y    = np.asarray(data_X_cont_norm_test, dtype=float)
Zc   = None if Z_test is None else np.asarray(Z_test, dtype=float)
ctrl = np.asarray(unperturbed_rows, dtype=int)
genes = np.asarray(gene_name_targeting, dtype=object)
n_cells, p = Y.shape

ate = np.full((p, p), np.nan); se = np.full((p, p), np.nan)
tst = np.full((p, p), np.nan); pv = np.full((p, p), np.nan)
n_pert = np.zeros(p, dtype=int)

# one least-squares solve per perturbed gene: the design is the same for every outcome
for i in range(p):
    idx_p = np.asarray(perturbed_rows.get(i, []), dtype=int)
    n_pert[i] = len(idx_p)
    if len(idx_p) < MIN_CELLS:
        continue
    rows = np.concatenate([idx_p, ctrl])
    d = np.concatenate([np.ones(len(idx_p)), np.zeros(len(ctrl))])
    X = np.column_stack([np.ones(len(rows)), d] + ([] if Zc is None else [Zc[rows]]))
    Yi = Y[rows]
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ (X.T @ Yi)
    resid = Yi - X @ B
    dof = len(rows) - np.linalg.matrix_rank(X)
    sigma2 = (resid ** 2).sum(axis=0) / dof
    ate[i] = B[1]
    se[i] = np.sqrt(XtX_inv[1, 1] * sigma2)
    with np.errstate(divide='ignore', invalid='ignore'):
        tst[i] = ate[i] / se[i]
    pv[i] = 2 * stats.t.sf(np.abs(tst[i]), dof)

for M in (ate, se, tst, pv):      # i -> i is the knockdown itself, not a gene-gene effect
    np.fill_diagonal(M, np.nan)

# Benjamini-Hochberg across all pairs within this dataset
q = np.full(pv.shape, np.nan)
ok = np.isfinite(pv)
pf = pv[ok]; order = np.argsort(pf); m = len(pf)
qf = np.empty(m)
qf[order] = np.minimum.accumulate((pf[order] * m / np.arange(1, m + 1))[::-1])[::-1]
q[ok] = np.clip(qf, 0, 1)

ii, jj = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
keep = np.isfinite(ate)
tab = pd.DataFrame({'dataset': ctype,
                    'source': genes[ii[keep]], 'target': genes[jj[keep]],
                    'source_idx': ii[keep], 'target_idx': jj[keep],
                    'ate': ate[keep], 'se': se[keep], 't': tst[keep],
                    'p': pv[keep], 'q': q[keep],
                    'n_perturbed': n_pert[ii[keep]], 'n_control': len(ctrl)})

np.savez(os.path.join(OUTDIR, f'{ctype}_ATE.npz'),
         ate=ate, se=se, t=tst, p=pv, q=q, n_perturbed=n_pert, genes=genes)
tab.to_csv(os.path.join(OUTDIR, f'{ctype}_ATE_long.csv'), index=False)
print(f'{ctype}: {len(tab):,} pairs from {int((n_pert >= MIN_CELLS).sum())} perturbations '
      f'({int((n_pert < MIN_CELLS).sum())} skipped, <{MIN_CELLS} cells), '
      f'{len(ctrl):,} control cells; {int((tab.q < FDR).sum()):,} significant at FDR {FDR}')

# summary over every dataset computed so far
rows = []
for f in sorted(glob.glob(os.path.join(OUTDIR, '*_ATE_long.csv'))):
    t = pd.read_csv(f)
    sig = t[t.q < FDR]
    rows.append(dict(dataset=t.dataset.iloc[0],
                     n_perturbations=t.source_idx.nunique(), n_genes=t.target_idx.nunique(),
                     n_pairs=len(t), n_control_cells=int(t.n_control.iloc[0]),
                     median_cells_per_perturbation=float(
                         t.groupby('source_idx').n_perturbed.first().median()),
                     n_significant=len(sig), frac_significant=len(sig) / len(t),
                     median_abs_ate=float(t.ate.abs().median()),
                     median_abs_ate_significant=float(sig.ate.abs().median()) if len(sig) else np.nan,
                     q95_abs_ate=float(t.ate.abs().quantile(0.95)),
                     frac_negative_among_significant=float((sig.ate < 0).mean()) if len(sig) else np.nan))
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUTDIR, 'ATE_summary.csv'), index=False)
print()
print(summary.to_string(index=False))
