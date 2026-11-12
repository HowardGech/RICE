"""
Fit three intervention parameterizations to one efficacy dataset:

    hard      xi fixed at 0   (published model, intervention_type='hard')
    soft      xi fixed at 1   (published model, intervention_type='soft')
    adaptive  xi_j estimated  (scRICE_eff, intervention_type='efficacy')

All three use the SAME optimizer (scRICE_eff) so that only the intervention
parameterization differs between arms.

Optimizer settings follow job_simu.py (the base benchmark driver): pen1=1e-4,
which is also the lambda1 value reported in Methods. NOTE that the other
scenario drivers -- job_simu_comp_cf.py, job_simu_cova.py, job_simu_missing.py,
job_simu_highMOI.py, job_simu_cpd.py -- all use pen1=1e-3 instead, so results
here are comparable to the base benchmark and to the manuscript's stated
lambda1, but NOT directly to Figures 2, 4A and 4B. Pass pen1 explicitly to
match those instead.

usage: python fit_efficacy.py <model> <n> <p> <s0> <graph_type> <xi_center> [pen1] [xi_scope]

xi_scope ('both' | 'parent') controls what xi multiplies in the ADAPTIVE arm
only; the hard and soft arms always use the published fixed-mask behaviour.
See README_efficacy.md -- local diagnostics show 'both' carries a large upward
bias in xi-hat (+0.21 at 20 genes / 135 cells per perturbation) that 'parent'
removes at no cost in graph recovery.
"""
import numpy as np
import os
import sys
import time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils import measurement, to_dag, set_random_seed
from scRICE_eff import scRICE_CF_eff
import pandas as pd

# Backend: 'cupy' (dbeigpu, default) or 'numpy' (cceb_normal fallback when the
# GPU host is closed). scRICE_eff is backend-agnostic and the maths is identical.
BACKEND = os.environ.get('RICE_BACKEND', 'cupy')
if BACKEND == 'cupy':
    import cupy as cp
    DTYPE = cp.float64
else:
    DTYPE = np.float64
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression as LR
from scipy.stats import spearmanr

job_id = int(os.environ['LSB_JOBINDEX'])
args = sys.argv[1:]
model = str(args[0])
n, p, s0, graph_type = int(args[1]), int(args[2]), int(args[3]), args[4]
xi_center = float(args[5])
pen1 = float(args[6]) if len(args) > 6 else 1e-4
xi_scope = args[7] if len(args) > 7 else 'both'

TAG = f'xic{int(round(xi_center * 100))}'
STEM = f'n{n//1000}k_p{p}_s{s0}_{graph_type}_{TAG}_repl{job_id}'
DATA = f'result/simu_eff/data/{model}/{STEM}.npz'
RESDIR = f'result/simu_eff/scGRAPH/{model}'
os.makedirs(RESDIR, exist_ok=True)
print(f'[fit_efficacy] {STEM}  backend={BACKEND}', flush=True)

set_random_seed(job_id)
z = np.load(DATA, allow_pickle=True)
X, W_true, mask = z['X'], z['W'], z['mask']
xi_true = z['xi']
indeg = z['indeg'] if 'indeg' in z.files else (W_true != 0).sum(axis=0)
has_parent = indeg > 0

# reduced control function: each gene on its OWN perturbation indicator only
predictor = np.log(1 + X)
res = np.zeros_like(X)
for i in range(p):
    lr = LR().fit(mask[:, i].reshape(-1, 1), predictor[:, i])
    res[:, i] = predictor[:, i] - lr.predict(mask[:, i].reshape(-1, 1))

# RICE_SCHEDULE=production reproduces job_simu.py's optimizer budget;
# 'reduced' is the CPU fallback.
# Runtime cost basis (all measured under NumPy; NONE measured on GPU)
#
#   (a) Per-epoch rate at PRODUCTION shape (n=10000, p=100, s0=400), 10 cores,
#       max_iter=1 so every nominal epoch really executes: 85 ms/epoch.
#       => UPPER BOUNDS, assuming the full nominal epoch count is reached:
#            production budget (1e6 epochs)  <= 23.6 h per fit
#            reduced budget    (9e4 epochs)  <=  2.1 h per fit
#
#   (b) What the two early-stopping breaks actually save, A/B measured on
#       identical data at REDUCED shape (n=3000, p=20, s0=80, d=3):
#            loss_tol=1e-6 vs loss_tol=0   ->  0.46x (reduced), 0.53x (production)
#       Separately, the h_tol=1e-2 OUTER break exits well before max_iter: the
#       full production budget finished in 138 s at this shape, far short of its
#       1e6 nominal epochs.
#
#   The bounds in (a) are at production shape and are sound. The factors in (b)
#   are at p=20 and DO NOT transfer reliably to p=100 -- convergence depends on
#   problem shape. So size -W on (a), not on (a) scaled by (b), and replace both
#   with the `time` rows of the first element that completes at real scale.
if os.environ.get('RICE_SCHEDULE', 'production') == 'reduced':
    SCHED = dict(lr=1e-3, max_iter=30, epochs=3000, checkpoint=500)
else:
    SCHED = dict(lr=1e-4, max_iter=100, epochs=int(1e4), checkpoint=2000)
FIT = dict(pen1=pen1, pen2=0, pen1_c=0, pen2_c=1e-2, regularizer='expm',
           h_tol=1e-2, loss_tol=1e-6, progress_bar=False, **SCHED)
W_TH = 0.3

rows = []
W_store = {}
for arm, itype in [('hard', 'hard'), ('soft', 'soft'), ('adaptive', 'efficacy')]:
    mo = scRICE_CF_eff(dtype=DTYPE, backend=BACKEND)
    mo.prep(X=X, predictor=predictor, confounding_res=res,
            intervention_effect=mask.astype(float), intervention_type=itype,
            add_intercept=True, eta_init=0.0, xi_scope=xi_scope)
    t0 = time.time()
    mo.fit(**FIT)
    secs = time.time() - t0

    r = mo.result(w_threshold=W_TH)
    W_est = to_dag(r['W'])
    acc = measurement(W_true, W_est)
    W_raw = mo.result(w_threshold=0.0)['W']
    acc['auc'] = roc_auc_score((W_true != 0).astype(int).ravel(), np.abs(W_raw).ravel())
    acc['time'] = secs
    if itype == 'efficacy':
        xh = r['xi']
        acc['xi_rmse_all'] = float(np.sqrt(np.mean((xh - xi_true) ** 2)))
        acc['xi_bias_all'] = float(np.mean(xh - xi_true))
        acc['xi_mean_hat'] = float(xh.mean())
        if has_parent.sum() > 1:
            acc['xi_rmse_parents'] = float(np.sqrt(np.mean((xh[has_parent] - xi_true[has_parent]) ** 2)))
            sp = spearmanr(xh[has_parent], xi_true[has_parent]).statistic
            acc['xi_spearman_parents'] = float(sp) if np.isfinite(sp) else np.nan
        if (~has_parent).sum() > 1:
            acc['xi_rmse_parentless'] = float(np.sqrt(np.mean((xh[~has_parent] - xi_true[~has_parent]) ** 2)))
        # Manuscript units: e = 1 - xi is the perturbation EFFICACY (e=1 hard, e=0 soft).
        # RMSE is invariant under the flip; bias and Spearman negate; the mean becomes 1-mean.
        eh, e_true = 1.0 - xh, 1.0 - xi_true
        acc['e_rmse_all'] = acc['xi_rmse_all']
        acc['e_bias_all'] = -acc['xi_bias_all']
        acc['e_mean_hat'] = float(eh.mean())
        acc['e_true_mean'] = float(e_true.mean())
        if 'xi_rmse_parents' in acc:
            acc['e_rmse_parents'] = acc['xi_rmse_parents']
            acc['e_spearman_parents'] = -acc['xi_spearman_parents']
        if 'xi_rmse_parentless' in acc:
            acc['e_rmse_parentless'] = acc['xi_rmse_parentless']
        W_store['xi_hat'] = xh
        W_store['e_hat'] = eh
    W_store[f'W_{arm}'] = W_raw
    for k, v in acc.items():
        rows.append({'method': arm, 'measure': k, 'value': v})
    print(f'[fit_efficacy] {arm:<9} mcc={acc["mcc"]:.3f} shd={acc["shd"]:.0f} '
          f'{secs:.0f}s', flush=True)

rows.append({'method': 'truth', 'measure': 'xi_true_mean', 'value': float(xi_true.mean())})
rows.append({'method': 'truth', 'measure': 'e_true_mean', 'value': float(1.0 - xi_true.mean())})
rows.append({'method': 'truth', 'measure': 'e_center', 'value': float(1.0 - xi_center)})
rows.append({'method': 'truth', 'measure': 'xi_center', 'value': xi_center})
rows.append({'method': 'truth', 'measure': 'n_parentless', 'value': float((~has_parent).sum())})
rows.append({'method': 'settings', 'measure': 'pen1', 'value': pen1})
rows.append({'method': 'settings', 'measure': 'backend_is_cupy',
             'value': float(BACKEND == 'cupy')})
rows.append({'method': 'settings', 'measure': 'max_iter', 'value': float(FIT['max_iter'])})
rows.append({'method': 'settings', 'measure': 'epochs', 'value': float(FIT['epochs'])})
rows.append({'method': 'settings', 'measure': 'lr', 'value': float(FIT['lr'])})
rows.append({'method': 'settings', 'measure': 'xi_scope_is_parent',
             'value': float(xi_scope == 'parent')})
rows.append({'method': 'settings', 'measure': 'w_threshold', 'value': W_TH})
pd.DataFrame(rows).to_csv(f'{RESDIR}/{STEM}.csv', index=False)
np.savez_compressed(f'{RESDIR}/{STEM}_West.npz', xi_true=xi_true,
                    indeg=indeg, **W_store)
print('[fit_efficacy] done', flush=True)
