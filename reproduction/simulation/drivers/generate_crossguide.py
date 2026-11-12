"""
Generate data with a direct cross-gene guide effect (exclusion violation, design B2).

usage: python generate_crossguide.py <model> <n> <p> <s0> <graph_type> <leak_density>
                                    [leak_effect] [d] [itype]
       leak_density  fraction of ordered (k, j), k != j, pairs that leak; 0 = no violation
       leak_effect   beta ~ U(-leak_effect, leak_effect); default 3.0, matching the
                     hidden-mediator arm's beta in [-3, 3]
       d             unmeasured confounders, default 10 (a control-function comparison at
                     d = 0 is vacuous; d = 10 is the top of the published 2 -> 10 sweep)
       itype         soft | hard, default soft

The DAG, weights and every legacy-RNG draw depend only on the array index, and beta is
drawn from a separate Generator, so for a given replicate the data at leak_density = 0
are bit-identical to simulate_NB_tgt and differences across densities are attributable to
the violation alone.
"""
import numpy as np
import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils import *                      # noqa: F401,F403
from sim_crossguide import simulate_NB_crossguide

args = sys.argv[1:]
model, n, p, s0, graph_type = args[0], int(args[1]), int(args[2]), int(args[3]), args[4]
leak_density = float(args[5])
leak_effect = float(args[6]) if len(args) > 6 else 3.0
d = int(args[7]) if len(args) > 7 else 10
itype = args[8] if len(args) > 8 else 'soft'

if model != 'NB':
    raise SystemExit(f'simulate_NB_crossguide is negative-binomial only, got {model!r}')

job_id = int(os.environ.get('LSB_JOBINDEX', 1))
set_random_seed(job_id)

TAG = (f'xg{int(round(leak_density * 1000))}_b{int(round(leak_effect))}_um{d}'
       f'{"" if itype == "soft" else "_hard"}')
DATADIR = f'result/simu_crossguide/data/{model}'
STEM = f'n{n // 1000}k_p{p}_s{s0}_{graph_type}_{TAG}_repl{job_id}'
os.makedirs(DATADIR, exist_ok=True)

B_true = simulate_dag(p, s0, graph_type)
W_true = simulate_parameter(B_true, w_ranges=((-1, -0.5), (0.5, 1.0)))

X, mask, Z, Um, Ur, Vm, Vr, Beta, info = simulate_NB_crossguide(
    W=W_true, theta_m=-1 * np.ones(p), theta_r=1 * np.ones(p), n=n,
    leak_density=leak_density, leak_effect=leak_effect,
    leak_rng=np.random.default_rng(70000 + job_id),
    d=d, intercept=True, g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
    hard_intervention=(itype == 'hard'))

print(f'[generate_crossguide] {STEM} {info}', flush=True)
np.savez_compressed(
    f'{DATADIR}/{STEM}.npz',
    X=X, W=W_true, mask=mask, Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr,
    Beta=Beta, leak_density=leak_density, leak_effect=leak_effect, d=d,
    itype=np.array(itype), n_leak=info['n_leak'],
    leak_per_gene=info['leak_per_gene'], leak_to_signal=info['leak_to_signal'],
    mean_leak_sd=info['mean_leak_sd'], mean_signal_sd=info['mean_signal_sd'],
    frac_genes_leaked=info['frac_genes_leaked'],
    n_leak_on_true_edge=info['n_leak_on_true_edge'],
    n_leak_off_edge=info['n_leak_off_edge'])
print('[generate_crossguide] done', flush=True)
