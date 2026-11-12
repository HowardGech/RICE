"""
Fit RICE under a direct cross-gene guide effect, comparing control-function variants.

Three arms on the same dataset:
    rcf    reduced control function  -- gene i regressed on its OWN indicator D_i
    fcf    full control function     -- gene i regressed on ALL indicators
    nocf   no control function

This is the arm where the reduced and full forms are predicted to SEPARATE IN THE
OPPOSITE DIRECTION from the published hidden-mediator result: a leak carried by D_k
cannot be absorbed by a first stage that only sees D_j, but can be absorbed by one that
sees every indicator. Whichever way it comes out, report it against the hidden-mediator
curve rather than pooled with it.

Backend from RICE_BACKEND ('cupy' default, 'numpy' for the CPU queue).
Optimizer budget from RICE_SCHEDULE ('production' default, 'reduced' for CPU).

usage: python fit_crossguide.py <model> <n> <p> <s0> <graph_type> <leak_density>
                                     [leak_effect] [d] [itype] [pen1]
"""
import numpy as np
import os
import sys
import time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils import *                      # noqa: F401,F403
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
leak_density = float(args[5])
leak_effect = float(args[6]) if len(args) > 6 else 3.0
d = int(args[7]) if len(args) > 7 else 10
itype = args[8] if len(args) > 8 else 'soft'
pen1 = float(args[9]) if len(args) > 9 else 1e-4

job_id = int(os.environ.get('LSB_JOBINDEX', 1))
TAG = (f'xg{int(round(leak_density * 1000))}_b{int(round(leak_effect))}_um{d}'
       f'{"" if itype == "soft" else "_hard"}')
DATADIR = f'result/simu_crossguide/data/{model}'
RESDIR = f'result/simu_crossguide/scGRAPH/{model}'
STEM = f'n{n // 1000}k_p{p}_s{s0}_{graph_type}_{TAG}_repl{job_id}'
os.makedirs(RESDIR, exist_ok=True)

if os.environ.get('RICE_SCHEDULE', 'production') == 'reduced':
    SCHED = dict(lr=1e-3, max_iter=30, epochs=3000, checkpoint=500)
else:
    SCHED = dict(lr=1e-4, max_iter=100, epochs=int(1e4), checkpoint=2000)
FIT = dict(pen1=pen1, pen2=0, pen1_c=0, pen2_c=1e-2, regularizer='expm',
           h_tol=1e-2, loss_tol=1e-6, progress_bar=False, **SCHED)
W_TH = 0.3

z = np.load(f'{DATADIR}/{STEM}.npz', allow_pickle=True)
X, W_true, mask = z['X'], z['W'], z['mask']
print(f'[fit_crossguide] {STEM} backend={BACKEND} density={leak_density} '
      f'n_leak={int(z["n_leak"])} off_edge={int(z["n_leak_off_edge"])}', flush=True)

predictor = np.log(1 + X)


def residuals(which):
    if which == 'nocf':
        return None
    res = np.zeros_like(predictor)
    for i in range(p):
        Zi = mask[:, [i]] if which == 'rcf' else mask      # reduced vs full first stage
        lm = LinearRegression().fit(Zi, predictor[:, i])
        res[:, i] = predictor[:, i] - lm.predict(Zi)
    return res


rows, W_store = [], {}
for arm in ['rcf', 'fcf', 'nocf']:
    t0 = time.time()
    mo = scRICE_CF_eff(dtype=DTYPE, backend=BACKEND)
    mo.prep(X=X, predictor=predictor, confounding_res=residuals(arm),
            intervention_effect=mask.astype(float), intervention_type=itype,
            add_intercept=True)
    mo.fit(**FIT)
    secs = time.time() - t0
    W_raw = mo.result(w_threshold=W_TH)['W']
    acc = measurement(W_true, to_dag(W_raw.copy()))
    acc['auc'] = roc_auc_score((W_true != 0).astype(int).ravel(), np.abs(W_raw).ravel())
    acc['time'] = secs
    W_store[f'W_{arm}'] = W_raw
    for k, v in acc.items():
        rows.append({'method': arm, 'measure': k, 'value': v})
    print(f'[fit_crossguide] {arm:<5} mcc={acc["mcc"]:.3f} shd={acc["shd"]:.0f} '
          f'{secs:.0f}s', flush=True)

rows += [{'method': 'truth', 'measure': 'leak_density', 'value': float(leak_density)},
         {'method': 'truth', 'measure': 'leak_effect', 'value': float(leak_effect)},
         {'method': 'truth', 'measure': 'n_leak', 'value': float(z['n_leak'])},
         {'method': 'truth', 'measure': 'leak_per_gene', 'value': float(z['leak_per_gene'])},
         {'method': 'truth', 'measure': 'leak_to_signal', 'value': float(z['leak_to_signal'])},
         {'method': 'truth', 'measure': 'frac_genes_leaked',
          'value': float(z['frac_genes_leaked'])},
         {'method': 'truth', 'measure': 'n_leak_on_true_edge',
          'value': float(z['n_leak_on_true_edge'])},
         {'method': 'truth', 'measure': 'n_leak_off_edge', 'value': float(z['n_leak_off_edge'])},
         {'method': 'truth', 'measure': 'd', 'value': float(d)},
         {'method': 'settings', 'measure': 'itype_is_hard', 'value': float(itype == 'hard')},
         {'method': 'settings', 'measure': 'pen1', 'value': pen1},
         {'method': 'settings', 'measure': 'w_threshold', 'value': W_TH},
         {'method': 'settings', 'measure': 'backend_is_cupy', 'value': float(BACKEND == 'cupy')},
         {'method': 'settings', 'measure': 'max_iter', 'value': float(FIT['max_iter'])},
         {'method': 'settings', 'measure': 'epochs', 'value': float(FIT['epochs'])},
         {'method': 'settings', 'measure': 'lr', 'value': float(FIT['lr'])}]

pd.DataFrame(rows).to_csv(f'{RESDIR}/{STEM}.csv', index=False)
np.savez_compressed(f'{RESDIR}/{STEM}_West.npz', Beta=z['Beta'], **W_store)
print('[fit_crossguide] done', flush=True)
