"""
Fit rcf / fcf / nocf on each configuration of one replicate.

    rcf    reduced control function -- gene i regressed on its OWN indicator D_i
    fcf    full control function    -- gene i regressed on ALL indicators
    nocf   no control function

Backend from RICE_BACKEND ('cupy' default, 'numpy' for the CPU queue). Optimizer budget
from RICE_SCHEDULE ('production' default, 'reduced' for the CPU fallback).

pen1 defaults to 1e-4, the Methods value used by job_simu.py. The published full-vs-reduced
comparison in job_simu_comp_cf.py used 1e-3. Use ONE value across every arm and
configuration or the curves are not comparable; a reduced-scale check found the penalty
did NOT change the no-violation baseline, so either is defensible as long as it is uniform.

usage: python fit_cf_comparison.py <model> <n> <p> <s0> <graph_type> [d] [pen1]
"""
import numpy as np
import os
import sys
import time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils import *                                      # noqa: F401,F403
from scRICE_eff import scRICE_CF_eff
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

BACKEND = os.environ.get('RICE_BACKEND', 'cupy')
if BACKEND == 'cupy':
    import cupy as cp
    DTYPE = cp.float64
else:
    DTYPE = np.float64

args = sys.argv[1:]
model, n, p, s0, graph_type = args[0], int(args[1]), int(args[2]), int(args[3]), args[4]
d = int(args[5]) if len(args) > 5 else 10
pen1 = float(args[6]) if len(args) > 6 else 1e-4

CONFIGS = ['none', 'mediator', 'direct_s0', 'direct_s05']
ARMS = ['rcf', 'fcf', 'nocf']
job_id = int(os.environ.get('LSB_JOBINDEX', 1))

if os.environ.get('RICE_SCHEDULE', 'production') == 'reduced':
    SCHED = dict(lr=1e-3, max_iter=30, epochs=3000, checkpoint=500)
else:
    SCHED = dict(lr=1e-4, max_iter=100, epochs=int(1e4), checkpoint=2000)
FIT = dict(pen1=pen1, pen2=0, pen1_c=0, pen2_c=1e-2, regularizer='expm',
           h_tol=1e-2, loss_tol=1e-6, progress_bar=False, **SCHED)
W_TH = 0.3

DATADIR = f'result/simu_cfcompare/data/{model}'
RESDIR = f'result/simu_cfcompare/scGRAPH/{model}'
os.makedirs(RESDIR, exist_ok=True)

rows = []
for cfg in CONFIGS:
    stem = f'n{n // 1000}k_p{p}_s{s0}_{graph_type}_{cfg}_um{d}_repl{job_id}'
    z = np.load(f'{DATADIR}/{stem}.npz', allow_pickle=True)
    X, W_true, mask = z['X'], z['W'], z['mask']
    predictor = np.log(1 + X)
    M = mask.astype(float)

    for arm in ARMS:
        res = None
        if arm != 'nocf':
            res = np.zeros_like(predictor)
            for i in range(p):
                Zi = M[:, [i]] if arm == 'rcf' else M
                lm = LinearRegression().fit(Zi, predictor[:, i])
                res[:, i] = predictor[:, i] - lm.predict(Zi)

        t0 = time.time()
        mo = scRICE_CF_eff(dtype=DTYPE, backend=BACKEND)
        mo.prep(X=X, predictor=predictor, confounding_res=res, intervention_effect=M,
                intervention_type='soft', add_intercept=True)
        mo.fit(**FIT)
        secs = time.time() - t0

        W_raw = mo.result(w_threshold=W_TH)['W']
        acc = measurement(W_true, to_dag(W_raw.copy()))
        acc['auc'] = roc_auc_score((W_true != 0).astype(int).ravel(),
                                   np.abs(W_raw).ravel())
        acc['time'] = secs
        for k, v in acc.items():
            rows.append({'config': cfg, 'method': arm, 'measure': k, 'value': float(v)})
        print(f'[fit] repl {job_id} {cfg:<11} {arm:<5} mcc={acc["mcc"]:.3f} '
              f'shd={acc["shd"]:.0f} {secs:.0f}s', flush=True)

    rows.append({'config': cfg, 'method': 'truth', 'measure': 'leak_to_signal',
                 'value': float(z['leak_to_signal'])})
    rows.append({'config': cfg, 'method': 'truth', 'measure': 'n_leak',
                 'value': float(z['n_leak'])})

for k, v in [('pen1', pen1), ('w_threshold', W_TH), ('d', d), ('replicate', job_id),
             ('max_iter', FIT['max_iter']), ('epochs', FIT['epochs']), ('lr', FIT['lr']),
             ('backend_is_cupy', float(BACKEND == 'cupy'))]:
    rows.append({'config': 'all', 'method': 'settings', 'measure': k, 'value': float(v)})

stem = f'n{n // 1000}k_p{p}_s{s0}_{graph_type}_cfcompare_um{d}_repl{job_id}'
pd.DataFrame(rows).to_csv(f'{RESDIR}/{stem}.csv', index=False)
print(f'[fit] wrote {RESDIR}/{stem}.csv', flush=True)
