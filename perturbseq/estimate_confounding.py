#!/usr/bin/env python
"""
Estimate how confounded a real CRISPRi Perturb-seq screen is, on the scale of the
simulated latent-factor sweep.

WHAT THIS DOES AND DOES NOT CLAIM
---------------------------------
The confounding-to-perturbation ratio sd(z'u_j)/|v_j| is NOT identifiable in real data:
shared variation among genes is part unmeasured confounding and part the regulatory
network itself, and nothing in the data separates them without assuming a model. So this
script does not try to measure the ratio directly. It runs ONE estimator on the real
screen and on the simulated sweep, and reports which simulated d the real screen matches.
Bias in the estimator cancels as long as it is applied identically to both -- which is
why the calibration mode below re-derives the simulated curve with this same code rather
than reusing a stored one.

Measured on the simulation, where the truth is known, the estimator inflates the true
ratio by roughly 2x and not by a constant factor (mostly because the measured on-target
log fold change attenuates as confounding rises). Its ABSOLUTE value is therefore not
interpretable and should not be quoted; only its position on the curve is.

    # on the real screen, from the objects real_differential_analysis.ipynb already builds
    python estimate_confounding.py --mode estimate

    # regenerate the simulated curve with identical preprocessing (needs scGRAPH on PATH)
    python estimate_confounding.py --mode calibrate

THE TWO QUANTITIES
  numerator    per-gene SD of the top-K principal-component reconstruction of the
               expression matrix restricted to CONTROL cells. Control cells carry no
               perturbation, so this is shared variation that no perturbation explains.
  denominator  per-target |on-target log fold change|: mean expression of gene j in cells
               targeting j, minus its mean in control cells. For CRISPRi this is the
               knockdown, and it anchors the comparison to a measured quantity --
               Replogle et al. report a median knockdown of 85.5% in K562 and 91.6% in
               RPE1, i.e. |LFC| of 1.93 and 2.48 in natural log units.

A ONE-LINE SANITY CHECK that needs no calibration: var_explained_topk, the fraction of
control-cell variance in the top K components. The simulated sweep gives 0.08 at d=0 and
0.59 at d=10, so a real screen above 0.59 is at least as confounded as the d=10 arm.
"""
import argparse
import numpy as np
import pandas as pd

K_FACTORS = 10          # components retained; MUST match between estimate and calibrate
MIN_CELLS = 10          # minimum cells per target for its LFC to be used


# --------------------------------------------------------------------------- core
def _prepare(X, already_normalized):
    """Return a log-scale cells x genes matrix. CPM+log1p unless already normalized."""
    X = np.asarray(X, dtype=float)
    if already_normalized:
        # a matrix that is already log-scale must not be logged again; a linear-scale
        # normalized matrix still needs the log.
        return X if X.max() < 50 else np.log1p(X)
    cpm = X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4
    return np.log1p(cpm)


def _control_rows(mask=None, perturbed_rows=None, unperturbed_rows=None, n_cells=None):
    if unperturbed_rows is not None:
        return np.asarray(unperturbed_rows, dtype=int)
    if mask is not None:
        return np.where(np.asarray(mask).sum(1) == 0)[0]
    if perturbed_rows is not None and n_cells is not None:
        hit = np.zeros(n_cells, dtype=bool)
        for idx in perturbed_rows.values():
            hit[np.asarray(idx, dtype=int)] = True
        return np.where(~hit)[0]
    raise ValueError('supply mask, or perturbed_rows + n_cells, or unperturbed_rows')


def _targets(mask=None, perturbed_rows=None):
    """-> dict {gene column index: row indices of cells targeting it}."""
    if perturbed_rows is not None:
        return {int(k): np.asarray(v, dtype=int) for k, v in perturbed_rows.items()}
    mask = np.asarray(mask)
    return {j: np.where(mask[:, j])[0] for j in range(mask.shape[1])}


def estimate(X, mask=None, perturbed_rows=None, unperturbed_rows=None,
             already_normalized=False, k=K_FACTORS, min_cells=MIN_CELLS,
             covariates=None):
    """Run the estimator. `covariates` (cells x q) are regressed out of the control-cell
    matrix before the SVD -- pass total counts / percent-mito here ONLY if you consider
    them nuisance rather than part of the confounding you want to measure."""
    L = _prepare(X, already_normalized)
    n, p = L.shape
    ctrl = _control_rows(mask, perturbed_rows, unperturbed_rows, n)
    if len(ctrl) < 50:
        raise ValueError(f'only {len(ctrl)} control cells; the numerator needs more')

    Lc = L[ctrl] - L[ctrl].mean(0)
    if covariates is not None:
        C = np.asarray(covariates, dtype=float)[ctrl]
        C = np.hstack([np.ones((len(ctrl), 1)), (C - C.mean(0)) / (C.std(0) + 1e-12)])
        Lc = Lc - C @ np.linalg.lstsq(C, Lc, rcond=None)[0]

    kk = int(min(k, min(Lc.shape) - 1))
    U, s, Vt = np.linalg.svd(Lc, full_matrices=False)
    latent_sd = ((U[:, :kk] * s[:kk]) @ Vt[:kk]).std(0)          # per gene

    tgt = _targets(mask, perturbed_rows)
    rows = []
    ctrl_mean = L[ctrl].mean(0)
    for j, idx in tgt.items():
        if j >= p or len(idx) < min_cells:
            continue
        lfc = abs(L[idx, j].mean() - ctrl_mean[j])
        if lfc > 1e-6:
            rows.append(dict(gene=j, n_cells=len(idx), ontarget_lfc=lfc,
                             latent_sd=latent_sd[j], ratio=latent_sd[j] / lfc))
    per_target = pd.DataFrame(rows)
    if per_target.empty:
        raise ValueError('no target passed the min_cells / nonzero-LFC filters')

    return dict(n_cells=n, n_genes=p, n_control=len(ctrl), k=kk,
                n_targets=len(per_target),
                latent_sd_median=float(np.median(latent_sd)),
                ontarget_lfc_median=float(per_target.ontarget_lfc.median()),
                ratio_median=float(per_target.ratio.median()),
                var_explained_topk=float((s[:kk] ** 2).sum() / (s ** 2).sum())), per_target


# -------------------------------------------------------------------- calibration
def calibrate(ds=(0, 2, 4, 6, 8, 10, 15, 20), n=10000, p=100, s0=None,
              seed=0, k=K_FACTORS, zscore=False):
    """Re-derive the simulated curve with THIS file's preprocessing. Needs the scGRAPH
    package importable (utils, sim_base)."""
    from utils import simulate_dag, simulate_parameter, set_random_seed
    from sim_base import simulate_NB_tgt
    s0 = s0 if s0 is not None else 4 * p          # keep edge density at the paper's 4p

    out = []
    for d in ds:
        set_random_seed(seed)
        B = simulate_dag(p, s0, 'ER')
        W = simulate_parameter(B, w_ranges=((-1, -.5), (.5, 1.0)))
        X, mask, Z, Um, Ur, Vm, Vr = simulate_NB_tgt(
            W=W, theta_m=-np.ones(p), theta_r=np.ones(p), d=d, intercept=True, n=n,
            g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
            hard_intervention=False)
        true_ratio = (np.nan if d == 0 else
                      float(((Z[:, 1:] @ Um[1:]).std(0) / np.abs(Vm.ravel())).mean()))
        if zscore:                                 # match data_X_cont_norm (cell 4)
            off = X.sum(1, keepdims=True) / np.mean(X.sum(1))
            Lz = np.log(1 + X / np.maximum(off, 1e-12))
            Lz = (Lz - Lz.mean(0)) / np.maximum(Lz.std(0), 1e-12)
            s, _ = estimate(Lz, mask=mask, already_normalized=True, k=k)
        else:
            s, _ = estimate(X, mask=mask, k=k)
        out.append(dict(d=d, true_ratio=true_ratio, est_ratio=s['ratio_median'],
                        latent_sd_median=s['latent_sd_median'],
                        ontarget_lfc_median=s['ontarget_lfc_median'],
                        var_explained_topk=s['var_explained_topk']))
        print(f"  d={d:<3} true {true_ratio if d else float('nan'):.3f}  "
              f"est {s['ratio_median']:.3f}  varexpl {s['var_explained_topk']:.3f}",
              flush=True)
    return pd.DataFrame(out)


def locate(est_ratio, cal):
    """Where does an estimator value sit on the simulated curve?"""
    cal = cal.sort_values('est_ratio')
    d_hat = float(np.interp(est_ratio, cal.est_ratio, cal.d))
    sub = cal[cal.d > 0].sort_values('d')
    return d_hat, float(np.interp(d_hat, sub.d, sub.true_ratio))


# --------------------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['estimate', 'calibrate'], default='estimate')
    ap.add_argument('--calibration', default='confounding_calibration.csv')
    ap.add_argument('--k', type=int, default=K_FACTORS)
    ap.add_argument('--out-prefix', default='confounding')
    ap.add_argument('--n', type=int, default=10000, help='cells, for --mode calibrate')
    ap.add_argument('--p', type=int, default=100, help='genes, for --mode calibrate')
    ap.add_argument('--s0', type=int, default=None, help='edges; defaults to 4*p')
    ap.add_argument('--zscore', action='store_true',
                    help='z-score each gene, matching data_X_cont_norm')
    a = ap.parse_args()

    if a.mode == 'calibrate':
        cal = calibrate(n=a.n, p=a.p, s0=a.s0, k=a.k, zscore=a.zscore)
        cal.to_csv(a.calibration, index=False)
        print(f"\nwrote {a.calibration}\n{cal.round(3).to_string(index=False)}")
    else:
        # Objects from real_differential_analysis.ipynb (cell 19). data_X_cont_norm_test
        # is already normalized; perturbed_rows maps gene index -> cell rows.
        s, per_target = estimate(
            data_X_cont_norm_test, perturbed_rows=perturbed_rows,
            unperturbed_rows=unperturbed_rows, already_normalized=True, k=a.k)
        per_target.to_csv(f'{a.out_prefix}_per_target.csv', index=False)
        for key, val in s.items():
            print(f"   {key:<24} {val:.4f}" if isinstance(val, float) else
                  f"   {key:<24} {val}")
        cal = pd.read_csv(a.calibration)
        d_hat, true_hat = locate(s['ratio_median'], cal)
        print(f"\n   matches simulated d ~ {d_hat:.1f}  ->  true ratio about {true_hat:.2f}")
        print(f"   variance check: top-{s['k']} PCs hold {s['var_explained_topk']:.3f} of "
              f"control-cell variance; the simulated d=10 arm holds "
              f"{float(cal.loc[cal.d == cal.d.max(), 'var_explained_topk'].iloc[0]):.3f}")
