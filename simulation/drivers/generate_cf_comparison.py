"""
Generate all four control-function comparison configurations from ONE DAG per replicate.

The array index is the replicate. Every configuration is built from that replicate's DAG
and the same data-RNG start, so the arms are PAIRED within replicate: the analysis is a
within-replicate difference, not a difference of independent means. Getting this wrong is
the easy mistake in an array job, where each element would normally draw its own data.

Configurations
    none        no violation -- the REFERENCE the other three are measured against
    mediator    the published hidden-mediator violation (density 0.05, in/out effect 3.0)
    direct_s0   direct cross-gene guide leak, deterministic carrier D_k
    direct_s05  the same leak with carrier D_k + 0.5 * eta_k (eta shared across recipients)

The two direct arms are matched on per-gene leak severity via leak_total_sd, so sigma
changes only the ABSORBABILITY of the leak, not its size.

usage: python generate_cf_comparison.py <model> <n> <p> <s0> <graph_type> [d]
"""
import numpy as np
import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils import *                                      # noqa: F401,F403
from sim_base import simulate_NB_tgt, simulate_NB_hidden_mediator
from sim_crossguide import simulate_NB_crossguide

args = sys.argv[1:]
model, n, p, s0, graph_type = args[0], int(args[1]), int(args[2]), int(args[3]), args[4]
d = int(args[5]) if len(args) > 5 else 10

DENS = 0.05                     # 5% of gene pairs, matching the published mediator arm
CONFIGS = ['none', 'mediator', 'direct_s0', 'direct_s05']
job_id = int(os.environ.get('LSB_JOBINDEX', 1))

DATADIR = f'result/simu_cfcompare/data/{model}'
os.makedirs(DATADIR, exist_ok=True)

set_random_seed(job_id)                                  # DAG: shared by all configs
B_true = simulate_dag(p, s0, graph_type)
W_true = simulate_parameter(B_true, w_ranges=((-1, -0.5), (0.5, 1.0)))

q = (int(n * 0.9) // p) / n                              # per-cell per-guide probability
TOTAL_SD = float(np.sqrt(DENS * (p - 1) * (3.0 ** 2 / 3.0) * q * (1 - q)))
common = dict(W=W_true, theta_m=-np.ones(p), theta_r=np.ones(p), n=n, d=d,
              intercept=True, g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
              hard_intervention=False)

print(f'[gen] replicate {job_id}: p={p} n={n} s0={s0} d={d} '
      f'leak_total_sd={TOTAL_SD:.4f}', flush=True)


def validate_counts(X, cfg, max_plausible=1e7):
    """
    Fail here rather than in the fit stage.

    The NB rescale in the generators can leave exp(m) beyond int64, in which case
    nbinom.rvs returns saturated or NEGATIVE counts; a negative count makes the fit
    stage's np.log(1 + X) produce NaN and the first-stage regression dies with
    "Input y contains NaN", several hours and one queue cycle later. Run
    patch_count_rescale.py on sim_base.py and sim_crossguide.py if this
    trips.
    """
    bad = []
    if not np.isfinite(X).all():
        bad.append(f'{int((~np.isfinite(X)).sum())} non-finite entries')
    if X.min() < 0:
        worst = int(np.unravel_index(np.argmin(X), X.shape)[1])
        bad.append(f'min={X.min():.4g} (gene {worst}) -- int64 overflow in nbinom.rvs')
    if X.max() > max_plausible:
        genes = np.where(X.max(axis=0) > max_plausible)[0]
        bad.append(f'max={X.max():.4g} on {len(genes)} gene(s) {genes[:8].tolist()} '
                   f'-- implausible counts, the rescale cap was undone')
    if bad:
        raise SystemExit(f'[gen] FAILED replicate {job_id} config {cfg}: '
                         + '; '.join(bad))

for cfg in CONFIGS:
    set_random_seed(100000 + job_id)                     # same data-RNG start per config
    if cfg == 'none':
        X, mask = simulate_NB_tgt(**common)[:2]
        info = {}
    elif cfg == 'mediator':
        X, mask = simulate_NB_hidden_mediator(
            mediator_density=DENS, mediator_in_effect=3.0, mediator_out_effect=3.0,
            **common)[:2]
        info = {}
    else:
        sigma = 0.0 if cfg == 'direct_s0' else 0.5
        out = simulate_NB_crossguide(
            leak_density=DENS, leak_total_sd=TOTAL_SD, leak_noise_sd=sigma,
            leak_rng=np.random.default_rng(70000 + job_id), **common)
        X, mask, info = out[0], out[1], out[-1]

    validate_counts(X, cfg)
    stem = f'n{n // 1000}k_p{p}_s{s0}_{graph_type}_{cfg}_um{d}_repl{job_id}'
    np.savez_compressed(
        f'{DATADIR}/{stem}.npz', X=X, W=W_true, mask=mask,
        leak_to_signal=info.get('leak_to_signal', 0.0),
        n_leak=info.get('n_leak', 0), leak_per_gene=info.get('leak_per_gene', 0.0))
    print(f'[gen] {stem}  mean_count={X.mean():.1f}  max_count={X.max():.0f}  '
          f'n_leak={info.get("n_leak", 0)}  '
          f'leak/signal={info.get("leak_to_signal", 0.0):.3f}', flush=True)

print('[gen] done', flush=True)
