"""
Generate Perturb-seq data, then corrupt the RECORDED guide assignment.

X is generated from the TRUE mask by the published generator; the corrupted mask is
what the fitting driver uses. Both are stored, so the true-assignment fit can be run
on identical data as the rho = 0 reference.

usage: python generate_mismatch.py <model> <n> <p> <s0> <graph_type> <rho> [mode] [d]
       mode in {swap, dropout, both}, default swap
       d = number of unmeasured confounders, default 10
       itype in {soft, hard}, default soft (sets hard_intervention in the DGP)

d MATTERS for this experiment. The fitting driver compares reduced, full and no control
function, and with d = 0 there is no confounding for any of them to correct, so the
comparison is uninformative -- the no-CF arm simply wins. The default of 10 is the top of
the published 2 -> 10 sweep and was calibrated at reduced scale (p=20, n=3000, one seed):
d=3 leaves MCC near ceiling (rcf 0.93 / fcf 0.97), d=10 puts it at ~0.75 with the expected
rcf >= fcf > nocf ordering, and d=20 with coef=2 INVERTS the ordering (nocf 0.364 beats
rcf 0.287) with every arm near the floor. Do not raise it past 10 without re-checking that
the ordering survives.
"""
import numpy as np
import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils import *                      # noqa: F401,F403
from sim_base import *         # noqa: F401,F403
from sim_guide_mismatch import corrupt_guide_assignment

args = sys.argv[1:]
model, n, p, s0, graph_type = args[0], int(args[1]), int(args[2]), int(args[3]), args[4]
rho = float(args[5])
mode = args[6] if len(args) > 6 else 'swap'
d = int(args[7]) if len(args) > 7 else 10
itype = args[8] if len(args) > 8 else 'soft'

job_id = int(os.environ.get('LSB_JOBINDEX', 1))
set_random_seed(job_id)                  # DAG and weights identical across rho

TAG = (f'mm{int(round(rho * 100))}{"" if mode == "swap" else mode[0]}_um{d}'
       f'{"" if itype == "soft" else "_hard"}')
DATADIR = f'result/simu_mismatch/data/{model}'
STEM = f'n{n // 1000}k_p{p}_s{s0}_{graph_type}_{TAG}_repl{job_id}'
os.makedirs(DATADIR, exist_ok=True)

B_true = simulate_dag(p, s0, graph_type)
W_true = simulate_parameter(B_true, w_ranges=((-1, -0.5), (0.5, 1.0)))
theta_m, theta_r = -1 * np.ones(p), 1 * np.ones(p)

gen = {'NB': simulate_NB_tgt, 'Pois': simulate_Pois_tgt, 'PLN': simulate_PLN_tgt}[model]
kw = dict(W=W_true, theta_m=theta_m, theta_r=theta_r, n=n, d=d, intercept=True,
          g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
          hard_intervention=(itype == 'hard'))
if model == 'PLN':
    kw['noise_scale'] = 0.5

X, mask_true, Z, Um, Ur, Vm, Vr = gen(**kw)

# corruption uses its own Generator so the count data are byte-identical across rho
mask_obs, info = corrupt_guide_assignment(
    mask_true, rho, mode=mode, rng=np.random.default_rng(90000 + job_id))

print(f'[generate_mismatch] {STEM} {info}', flush=True)
np.savez_compressed(
    f'{DATADIR}/{STEM}.npz',
    X=X, W=W_true, mask=mask_obs, mask_true=mask_true,
    Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr,
    rho=rho, mode=np.array(mode), d=d, itype=np.array(itype),
    realised_rate=info['realised_rate'],
    frac_records_incorrect=info['frac_records_incorrect'],
    n_swapped=info['n_swapped'], n_dropped=info['n_dropped'])
print('[generate_mismatch] done', flush=True)
