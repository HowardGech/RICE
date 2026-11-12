#!/usr/bin/env python
"""
Vanilla average treatment effects for every (perturbed gene i -> gene j) pair.

This is the same regression real_differential_analysis.ipynb cell 60 already runs for
its total-effect p-values -- gene j regressed on a perturbation indicator for gene i plus
the technical covariates Z, over the cells perturbed for i together with the unperturbed
cells -- except that it is run for ALL pairs rather than only a method's non-zero edges,
and the COEFFICIENT is kept alongside the p-value.

    ATE(i -> j) = E[x_j | do(perturb i)] - E[x_j | no perturbation]

so it is a TOTAL effect: every path from i to j contributes, and it is not an estimate
of the direct effect w_ij that RICE reports.

Vectorized: for a given i the design matrix is the same for every outcome gene, so one
least-squares solve gives the whole row of ATEs. That is p solves rather than p^2.

Inputs, exactly the objects cell 19 builds:
    data_X_cont_norm_test   (n_cells, p)  normalized expression, test split
    Z_test                  (n_cells, d)  technical covariates
    perturbed_rows          {i: row indices perturbed for gene i}
    unperturbed_rows        row indices with no perturbation
    gene_name_targeting     (p,)          gene names

Usage inside the notebook, with ctype already set:
    exec(open('compute_ate.py').read())
or from the shell after saving those objects with save_ate_inputs() below.
"""
import os
import numpy as np, pandas as pd
from scipy import stats

MIN_CELLS = 10          # skip a perturbation with fewer than this many cells
FDR = 0.05


def ate_matrix(Y, Z, perturbed_rows, unperturbed_rows, min_cells=MIN_CELLS):
    """Return (ate, se, tstat, pval, n_pert) arrays of shape (p, p), NaN where skipped."""
    n, p = Y.shape
    ate = np.full((p, p), np.nan)
    se = np.full((p, p), np.nan)
    tst = np.full((p, p), np.nan)
    pv = np.full((p, p), np.nan)
    n_pert = np.zeros(p, dtype=int)
    ctrl = np.asarray(unperturbed_rows)
    Zc = None if Z is None else np.asarray(Z, dtype=float)

    for i in range(p):
        idx_p = np.asarray(perturbed_rows.get(i, []), dtype=int)
        n_pert[i] = len(idx_p)
        if len(idx_p) < min_cells:
            continue
        rows = np.concatenate([idx_p, ctrl])
        d = np.concatenate([np.ones(len(idx_p)), np.zeros(len(ctrl))])
        X = np.column_stack([np.ones(len(rows)), d] +
                            ([] if Zc is None else [Zc[rows]]))
        Yi = np.asarray(Y[rows], dtype=float)                 # (n_i, p) all outcomes

        XtX_inv = np.linalg.pinv(X.T @ X)
        B = XtX_inv @ (X.T @ Yi)                              # (q, p)
        resid = Yi - X @ B
        dof = len(rows) - np.linalg.matrix_rank(X)
        sigma2 = (resid ** 2).sum(axis=0) / dof               # per outcome gene
        var_d = XtX_inv[1, 1] * sigma2                        # variance of the D coef

        ate[i] = B[1]
        se[i] = np.sqrt(var_d)
        with np.errstate(divide='ignore', invalid='ignore'):
            tst[i] = ate[i] / se[i]
        pv[i] = 2 * stats.t.sf(np.abs(tst[i]), dof)

    np.fill_diagonal(ate, np.nan)                             # i -> i is the knockdown
    np.fill_diagonal(se, np.nan); np.fill_diagonal(tst, np.nan); np.fill_diagonal(pv, np.nan)
    return ate, se, tst, pv, n_pert


def bh(p):
    """Benjamini-Hochberg q-values over the finite entries of a flat array."""
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    qv = np.empty(m)
    qv[order] = np.minimum.accumulate((pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    q[ok] = np.clip(qv, 0, 1)
    return q


def long_table(ate, se, tst, pv, q, genes, n_pert, n_ctrl, dataset):
    p = ate.shape[0]
    ii, jj = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    keep = np.isfinite(ate)
    return pd.DataFrame({
        'dataset': dataset,
        'source': np.asarray(genes)[ii[keep]], 'target': np.asarray(genes)[jj[keep]],
        'source_idx': ii[keep], 'target_idx': jj[keep],
        'ate': ate[keep], 'se': se[keep], 't': tst[keep], 'p': pv[keep], 'q': q[keep],
        'n_perturbed': n_pert[ii[keep]], 'n_control': n_ctrl})


def run(dataset, Y, Z, perturbed_rows, unperturbed_rows, genes, outdir='.'):
    ate, se, tst, pv, n_pert = ate_matrix(Y, Z, perturbed_rows, unperturbed_rows)
    q = bh(pv)
    np.savez(os.path.join(outdir, f'{dataset}_ATE.npz'),
             ate=ate, se=se, t=tst, p=pv, q=q, n_perturbed=n_pert,
             genes=np.asarray(genes, dtype=object))
    tab = long_table(ate, se, tst, pv, q, genes, n_pert, len(unperturbed_rows), dataset)
    tab.to_csv(os.path.join(outdir, f'{dataset}_ATE_long.csv'), index=False)
    n_skipped = int((n_pert < MIN_CELLS).sum())
    print(f'{dataset}: {tab.shape[0]:,} pairs from {int((n_pert >= MIN_CELLS).sum())} '
          f'perturbations ({n_skipped} skipped, <{MIN_CELLS} cells), '
          f'{len(unperturbed_rows):,} control cells; '
          f'{int((tab.q < FDR).sum()):,} significant at FDR {FDR}')
    return tab


def summarize(tables, outdir='.'):
    """One row per dataset: how many ATEs, how many significant, effect-size spread."""
    rows = []
    for t in tables:
        sig = t[t.q < FDR]
        rows.append(dict(
            dataset=t.dataset.iloc[0],
            n_perturbations=t.source_idx.nunique(), n_genes=t.target_idx.nunique(),
            n_pairs=len(t), n_control_cells=int(t.n_control.iloc[0]),
            median_cells_per_perturbation=float(t.groupby('source_idx').n_perturbed.first().median()),
            n_significant=len(sig), frac_significant=len(sig) / len(t),
            median_abs_ate=float(t.ate.abs().median()),
            median_abs_ate_significant=float(sig.ate.abs().median()) if len(sig) else np.nan,
            q95_abs_ate=float(t.ate.abs().quantile(0.95)),
            frac_negative_among_significant=float((sig.ate < 0).mean()) if len(sig) else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, 'ATE_summary.csv'), index=False)
    print('\n' + out.to_string(index=False))
    return out


if __name__ == '__main__' or 'data_X_cont_norm_test' in dir():
    # In the notebook: loop the two cell types, rebuilding the cell-19 objects for each.
    # Here only the currently loaded ctype is available, so run it and append.
    tabs = []
    for _ds in [globals().get('ctype')]:
        if _ds is None:
            raise RuntimeError('set ctype and build the cell-19 objects first')
        tabs.append(run(_ds, data_X_cont_norm_test, Z_test, perturbed_rows,
                        unperturbed_rows, gene_name_targeting))
    _prev = [f for f in os.listdir('.') if f.endswith('_ATE_long.csv')]
    tabs = [pd.read_csv(f) for f in sorted(_prev)]
    summarize(tabs)
