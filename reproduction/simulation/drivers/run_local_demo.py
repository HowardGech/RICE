"""
Local (CPU) KD-efficacy simulation. Reduced scale relative to the cluster
pipeline in eff_pipeline/, but fully replicated and parallel.

Four arms per dataset, all sharing the same optimizer so only the intervention
parameterization differs:
    hard             xi fixed at 0
    soft             xi fixed at 1
    adaptive_both    xi estimated, scaling parent + control-function terms
    adaptive_parent  xi estimated, scaling the parent term only

Datasets are stored so other methods (inspre / IBCD / dotears) can be run on
exactly the same data later.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, os.path.abspath('.'))
from utils import simulate_dag, simulate_parameter, measurement, to_dag  # noqa: E402
from sim_partial_efficacy import simulate_NB_efficacy                     # noqa: E402
from scRICE_eff import scRICE_CF_eff                                     # noqa: E402

ARMS = [('hard', 'hard', 'both'),
        ('soft', 'soft', 'both'),
        ('adaptive_both', 'efficacy', 'both'),
        ('adaptive_parent', 'efficacy', 'parent')]


def make_data(center, repl, cfg):
    """DAG and weights depend only on `repl`, so they are identical across
    centers and d values. xi is drawn from a separate stream."""
    p, s0, n, hw = cfg['p'], cfg['s0'], cfg['n'], cfg['half_width']
    random.seed(repl)
    np.random.seed(repl)
    B = simulate_dag(p, s0, 'ER')
    W_true = simulate_parameter(B, w_ranges=((-1, -.5), (.5, 1.0)))
    xi_rng = np.random.default_rng(90000 + 1000 * repl + int(round(center * 100)))
    xi = np.clip(xi_rng.uniform(center - hw, center + hw, size=p), 0.0, 1.0)
    X, mask, Z, Um, Ur, Vm, Vr, xi_vec = simulate_NB_efficacy(
        W=W_true, theta_m=-np.ones(p), theta_r=np.ones(p), n=n, xi=xi,
        d=cfg['d'], intercept=True, g=lambda x: np.log(1 + x),
        shift=2, coef=1, covar=1)
    return X, mask, W_true, xi_vec, Z, Um, Ur, Vm, Vr


def one_cell(task):
    center, repl, cfg = task
    X, mask, W_true, xi_true, Z, Um, Ur, Vm, Vr = make_data(center, repl, cfg)
    p = cfg['p']
    indeg = (W_true != 0).sum(axis=0)
    has_parent = indeg > 0

    if cfg['save_data']:
        os.makedirs(cfg['datadir'], exist_ok=True)
        stem = f"n{cfg['n']}_p{p}_s{cfg['s0']}_ER_d{cfg['d']}_xic{int(round(center*100))}_repl{repl}"
        np.savez_compressed(os.path.join(cfg['datadir'], stem + '.npz'),
                            X=X, W=W_true, mask=mask, Z=Z, Um=Um, Ur=Ur,
                            Vm=Vm, Vr=Vr, xi=xi_true, xi_center=center,
                            half_width=cfg['half_width'], indeg=indeg, d=cfg['d'])

    predictor = np.log(1 + X)
    res = np.zeros_like(X)
    for i in range(p):
        lr = LR().fit(mask[:, i].reshape(-1, 1), predictor[:, i])
        res[:, i] = predictor[:, i] - lr.predict(mask[:, i].reshape(-1, 1))

    rows = []
    for arm, itype, scope in ARMS:
        mo = scRICE_CF_eff(backend='numpy')
        mo.prep(X=X, predictor=predictor, confounding_res=res,
                intervention_effect=mask.astype(float), intervention_type=itype,
                add_intercept=True, eta_init=0.0, xi_scope=scope)
        t0 = time.time()
        mo.fit(pen1=cfg['pen1'], pen2=0, pen1_c=0, pen2_c=1e-2,
               regularizer='expm', lr=cfg['lr'], max_iter=cfg['max_iter'],
               epochs=cfg['epochs'], checkpoint=500, h_tol=1e-2,
               loss_tol=1e-6, lr_eta_multiplier=cfg['eta_mult'],
               progress_bar=False)
        secs = time.time() - t0
        r = mo.result(w_threshold=cfg['w_threshold'])
        W_raw = mo.result(w_threshold=0.0)['W']
        acc = measurement(W_true, to_dag(r['W']))
        acc = {k: float(v) for k, v in acc.items()}
        acc['auc'] = float(roc_auc_score((W_true != 0).astype(int).ravel(),
                                         np.abs(W_raw).ravel()))
        acc['time'] = secs
        if itype == 'efficacy':
            xh = r['xi']
            acc['xi_rmse_all'] = float(np.sqrt(np.mean((xh - xi_true) ** 2)))
            acc['xi_bias_all'] = float(np.mean(xh - xi_true))
            acc['xi_mean_hat'] = float(xh.mean())
            if has_parent.sum() > 1:
                acc['xi_rmse_parents'] = float(np.sqrt(np.mean(
                    (xh[has_parent] - xi_true[has_parent]) ** 2)))
                sp = spearmanr(xh[has_parent], xi_true[has_parent]).statistic
                acc['xi_spearman_parents'] = float(sp) if np.isfinite(sp) else np.nan
            if (~has_parent).sum() > 1:
                acc['xi_rmse_parentless'] = float(np.sqrt(np.mean(
                    (xh[~has_parent] - xi_true[~has_parent]) ** 2)))
        for k, v in acc.items():
            rows.append(dict(d=cfg['d'], xi_center=center, repl=repl, arm=arm,
                             measure=k, value=v))
    rows.append(dict(d=cfg['d'], xi_center=center, repl=repl, arm='truth',
                     measure='xi_true_mean', value=float(xi_true.mean())))
    rows.append(dict(d=cfg['d'], xi_center=center, repl=repl, arm='truth',
                     measure='n_parentless', value=float((~has_parent).sum())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--p', type=int, default=20)
    ap.add_argument('--s0', type=int, default=80)
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--d', type=int, nargs='+', default=[0, 3])
    ap.add_argument('--centers', type=float, nargs='+',
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument('--half-width', type=float, default=0.1)
    ap.add_argument('--pen1', type=float, default=1e-3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-iter', type=int, default=30)
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--eta-mult', type=float, default=20.0)
    ap.add_argument('--w-threshold', type=float, default=0.3)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--datadir', default='local_sim_data')
    ap.add_argument('--out', default='local_sim_results.csv')
    ap.add_argument('--no-save-data', action='store_true')
    a = ap.parse_args()

    tasks = []
    for dv in a.d:
        cfg = dict(n=a.n, p=a.p, s0=a.s0, d=dv, half_width=a.half_width,
                   pen1=a.pen1, lr=a.lr, max_iter=a.max_iter, epochs=a.epochs,
                   eta_mult=a.eta_mult, w_threshold=a.w_threshold,
                   datadir=a.datadir, save_data=not a.no_save_data)
        for c in a.centers:
            for r in range(1, a.reps + 1):
                tasks.append((c, r, cfg))

    print(f'{len(tasks)} datasets x {len(ARMS)} arms = '
          f'{len(tasks)*len(ARMS)} fits on {a.workers} workers', flush=True)
    t0 = time.time()
    out = []
    with Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(one_cell, tasks), 1):
            out.extend(rows)
            if i % 5 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f'  {i}/{len(tasks)} datasets  {el/60:.1f} min elapsed, '
                      f'~{el/i*(len(tasks)-i)/60:.1f} min left', flush=True)
    df = pd.DataFrame(out)
    df.to_csv(a.out, index=False)
    print(f'wrote {a.out}  {df.shape}  in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
