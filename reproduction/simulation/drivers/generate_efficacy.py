"""
Generate Perturb-seq data with per-gene knockdown efficacy.

xi_j ~ Uniform(center - half_width, center + half_width), clipped to [0, 1],
drawn independently per gene, where xi_j = 1 - efficacy_j (xi=0 hard, xi=1 soft).

The DAG and edge weights are drawn from the LEGACY global RNG seeded by the
array index, BEFORE any center-dependent draw, and xi is drawn from a separate
Generator. Consequence: for a given replicate, W_true and every covariate /
intervention-assignment draw are IDENTICAL across xi centers, so differences
between centers are attributable to xi alone.

usage: python generate_efficacy.py <model> <n> <p> <s0> <graph_type> <xi_center> [half_width]
"""
import numpy as np
import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils import *                      # noqa: F401,F403
from sim_base import *         # noqa: F401,F403

job_id = int(os.environ['LSB_JOBINDEX'])
args = sys.argv[1:]
model = str(args[0])
n, p, s0, graph_type = int(args[1]), int(args[2]), int(args[3]), args[4]
xi_center = float(args[5])
half_width = float(args[6]) if len(args) > 6 else 0.1

if model not in ('NB', 'GNB'):
    raise ValueError('efficacy generator is negative-binomial only')

TAG = f'xic{int(round(xi_center * 100))}'
OUTDIR = f'result/simu_eff/data/{model}'
os.makedirs(OUTDIR, exist_ok=True)
out = f'{OUTDIR}/n{n//1000}k_p{p}_s{s0}_{graph_type}_{TAG}_repl{job_id}.npz'
print(f'[generate_efficacy] {TAG} repl{job_id} -> {out}', flush=True)

# --- structure: identical across xi centers for this replicate --------------
set_random_seed(job_id)
B_true = simulate_dag(p, s0, graph_type)
W_true = simulate_parameter(B_true, w_ranges=((-1, -.5), (.5, 1.0)))

# --- xi: separate stream, does not perturb the legacy stream ----------------
xi_rng = np.random.default_rng(90000 + 1000 * job_id + int(round(xi_center * 100)))
xi = np.clip(xi_rng.uniform(xi_center - half_width,
                            xi_center + half_width, size=p), 0.0, 1.0)

theta_m = -1 * np.ones(p)
theta_r = 1 * np.ones(p)
X, mask, Z, Um, Ur, Vm, Vr, xi_vec = simulate_NB_efficacy(
    W=W_true, theta_m=theta_m, theta_r=theta_r, n=n, xi=xi,
    d=0, intercept=True, g=lambda x: np.log(1 + x),
    shift=2, coef=1, covar=1)

indeg = (W_true != 0).sum(axis=0)
np.savez(out,
         X=X, W=W_true, mask=mask, Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr,
         xi=xi_vec, xi_center=xi_center, half_width=half_width, indeg=indeg)
print(f'[generate_efficacy] saved  X{X.shape}  xi mean {xi_vec.mean():.4f} '
      f'(nominal {xi_center})  parentless genes {(indeg == 0).sum()}', flush=True)
