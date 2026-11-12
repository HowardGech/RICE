"""
Fit RICE under guide misassignment, comparing control-function variants.

Three arms on the same dataset, all using the RECORDED (corrupted) mask:
    rcf    reduced control function  -- each gene regressed on its OWN indicator
    fcf    full control function     -- each gene regressed on ALL indicators
    nocf   no control function

The control function cannot repair measurement error in the instrument, so the
question is how fast each variant degrades, not whether it corrects the problem.

Backend from RICE_BACKEND ('cupy' default, 'numpy' for the CPU queue).
Optimizer budget from RICE_SCHEDULE ('production' default, 'reduced' for CPU).

usage: python fit_mismatch.py <model> <n> <p> <s0> <graph_type> <rho> [mode] [d]
                                   [itype] [pen1]

INTERVENTION TYPE MATTERS MORE HERE THAN ELSEWHERE. Under 'soft' the mask does not gate
the parental sum at all (D_j^e = 1), so a misassigned record can only damage the
perturbation shift v_j D_j and the control-function residuals -- the parental structure is
untouched. Under 'hard' the mask decides which parental terms are zeroed in which cells,
so misassignment corrupts the structural equations directly. Expect the corruption effect
to be markedly larger under 'hard'; running both is the informative comparison.
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
rho = float(args[5])
mode = args[6] if len(args) > 6 else 'swap'
d = int(args[7]) if len(args) > 7 else 10
itype = args[8] if len(args) > 8 else 'soft'
pen1 = float(args[9]) if len(args) > 9 else 1e-4

job_id = int(os.environ.get('LSB_JOBINDEX', 1))
TAG = (f'mm{int(round(rho * 100))}{"" if mode == "swap" else mode[0]}_um{d}'
       f'{"" if itype == "soft" else "_hard"}')
DATADIR = f'result/simu_mismatch/data/{model}'
RESDIR = f'result/simu_mismatch/scGRAPH/{model}'
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
mask_true = z['mask_true']
print(f'[fit_mismatch] {STEM} backend={BACKEND} rho={rho} mode={mode} '
      f'realised={float(z["realised_rate"]):.4f}', flush=True)

predictor = np.log(1 + X)


def residuals(which):
    """Control-function residuals from the RECORDED mask."""
    if which == 'nocf':
        return None
    res = np.zeros_like(predictor)
    for i in range(p):
        Zi = mask[:, [i]] if which == 'rcf' else mask     # reduced vs full
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
    r = mo.result(w_threshold=W_TH)
    W_raw = r['W']
    acc = measurement(W_true, to_dag(W_raw.copy()))
    acc['auc'] = roc_auc_score((W_true != 0).astype(int).ravel(), np.abs(W_raw).ravel())
    acc['time'] = secs
    W_store[f'W_{arm}'] = W_raw
    for k, v in acc.items():
        rows.append({'method': arm, 'measure': k, 'value': v})
    print(f'[fit_mismatch] {arm:<5} mcc={acc["mcc"]:.3f} shd={acc["shd"]:.0f} '
          f'{secs:.0f}s', flush=True)

# how corrupted this replicate actually was, for the x axis
rows += [{'method': 'truth', 'measure': 'rho', 'value': float(rho)},
         {'method': 'truth', 'measure': 'realised_rate', 'value': float(z['realised_rate'])},
         {'method': 'truth', 'measure': 'frac_records_incorrect',
          'value': float(z['frac_records_incorrect'])},
         {'method': 'truth', 'measure': 'n_swapped', 'value': float(z['n_swapped'])},
         {'method': 'truth', 'measure': 'n_dropped', 'value': float(z['n_dropped'])},
         {'method': 'truth', 'measure': 'mode_is_swap', 'value': float(mode == 'swap')},
         {'method': 'truth', 'measure': 'd', 'value': float(d)},
         {'method': 'settings', 'measure': 'itype_is_hard', 'value': float(itype == 'hard')},
         {'method': 'settings', 'measure': 'pen1', 'value': pen1},
         {'method': 'settings', 'measure': 'w_threshold', 'value': W_TH},
         {'method': 'settings', 'measure': 'backend_is_cupy', 'value': float(BACKEND == 'cupy')},
         {'method': 'settings', 'measure': 'max_iter', 'value': float(FIT['max_iter'])},
         {'method': 'settings', 'measure': 'epochs', 'value': float(FIT['epochs'])},
         {'method': 'settings', 'measure': 'lr', 'value': float(FIT['lr'])}]

pd.DataFrame(rows).to_csv(f'{RESDIR}/{STEM}.csv', index=False)
np.savez_compressed(f'{RESDIR}/{STEM}_West.npz',
                    mask_true_sum=mask_true.sum(axis=0),
                    mask_obs_sum=mask.sum(axis=0), **W_store)
print('[fit_mismatch] done', flush=True)
