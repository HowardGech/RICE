# KD-efficacy simulation — what to install where

Copy into `$SCGRAPH_DIR/`:

| file | action |
|---|---|
| `sim_partial_efficacy.py` | append its `simulate_NB_efficacy` to `sim_base.py` (drivers do `from sim_base import *`), or keep it standalone and add an explicit import |
| `scRICE_eff.py` | drop in as-is |
| `generate_efficacy.py`, `fit_efficacy.py` | drop in as-is |
| `generate_efficacy.sh`, `job_eff.sh` | drop in as-is; `mkdir -p out` first |

## Run order

    mkdir -p out
    bsub < generate_efficacy.sh        # CPU, cceb_normal, 20 elements x 6 centers
    bjobs -A <jobid>             # poll; EMPTY result means "not registered yet", not "finished"
    bsub < job_eff.sh            # GPU, dbeigpu, 20 elements x 6 centers x 3 fits

## Design

xi_j ~ Uniform(center - 0.1, center + 0.1) clipped to [0, 1], drawn per gene,
centers 0.0 0.2 0.4 0.6 0.8 1.0. xi = 1 - efficacy, so xi=0 is a complete
knockdown (hard intervention) and xi=1 is no residual suppression of parent
influence (soft intervention).

Clipping puts a point mass at the boundary in the end arms: at center 0.0 half
the genes land exactly at xi=0 and the realized mean is 0.025, not 0.0
(symmetrically 0.975 at center 1.0). The realized `xi` vector is stored in every
`.npz`, so plot against `xi_true_mean` from the results CSV rather than the
nominal center.

For a given replicate index, `W_true` and all covariate / intervention draws are
identical across the six centers: the DAG and weights come from the legacy global
RNG seeded by `LSB_JOBINDEX` before any center-dependent draw, and xi is drawn
from a separate `default_rng`. Differences across centers are therefore
attributable to xi alone.

## Sparsity penalty

`fit_efficacy.py` defaults to `pen1=1e-4`, which is what `job_simu.py` uses and
what Methods reports as lambda1. Every OTHER scenario driver in the repo
(`job_simu_comp_cf.py`, `job_simu_cova.py`, `job_simu_missing.py`,
`job_simu_highMOI.py`, `job_simu_cpd.py`) uses `pen1=1e-3`. So these results are
comparable to the base benchmark and to the manuscript text, but not directly to
Figures 2, 4A and 4B. Pass a sixth argument to override; `job_eff.sh` sets it in
one place. The value used is recorded in each results CSV under
`method=settings, measure=pen1`.

## Stored data

`simu_eff/data/NB/n10k_p100_s400_ER_xic{C}_repl{R}.npz` holds
`X, W, mask, Z, Um, Ur, Vm, Vr, xi, xi_center, half_width, indeg`.

The keys `X`, `W`, `mask` match what `IBCD_prepare.py`, `dotears_run.py` and the
inspre scripts already read, so running those baselines needs only a path/tag
change, not a data change. 120 files at roughly 8 MB each, about 1 GB total.

`simu_eff/scGRAPH/NB/{stem}_West.npz` additionally holds the unthresholded
`W_hard`, `W_soft`, `W_adaptive` and the estimated `xi_hat`, so figures can be
rebuilt without refitting.

## Reported per element

Long-format CSV with columns `method,measure,value`, `method` in
{hard, soft, adaptive, truth}. Measures are the usual `measurement()` keys
(precision, recall, f1, mcc, fdr, fnr, shd, five matrix norms) plus `auc` and
`time`. The adaptive arm adds `xi_rmse_all`, `xi_bias_all`, `xi_mean_hat`, and
— split by whether a gene has parents — `xi_rmse_parents`,
`xi_spearman_parents`, `xi_rmse_parentless`.

The split matters: xi_j enters the likelihood only multiplied by that gene's
parent signal, so it is unidentified for parentless genes (a mean of 12 of 100
under these ER settings). Report xi-hat on the parent-carrying subset and quote
the parentless RMSE as the null reference.

## Before trusting the comparison

All three arms are fit with `scRICE_eff` so that only the intervention
parameterization differs. `scRICE_eff` is a faithful transcription of
`scRICE_CF` for the fixed-mask cases, but it is not the identical file, and its
outer loop omits the `logdet` retry branch (unused here since every driver uses
`regularizer='expm'`). Run one `hard` and one `soft` fit through your existing
`job_simu.py` on the same `.npz` and confirm the metrics agree before using the
three-arm comparison in the paper.

`eta_clip=8.0` bounds xi to [3.4e-4, 0.99966], so the adaptive arm approaches
but never exactly reaches the hard and soft endpoints.
