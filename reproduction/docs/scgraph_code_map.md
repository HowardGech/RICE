# RICE / scGRAPH simulation pipeline — code map

Derived from the uploaded `scGRAPH.zip` (394 entries; 146 `.py`, 52 `.ipynb`, 38 `.sh`).
Everything below was read from the code, not inferred from the manuscript.

## Two-stage design

Every simulation scenario is two LSF array jobs, with `LSB_JOBINDEX` doubling as the
replicate index and the RNG seed (`set_random_seed(job_id)`):

1. **Generate** — `job_gen*.py` builds `B_true` via `simulate_dag(p, s0, graph_type)`,
   draws `W_true` via `simulate_parameter(B_true, w_ranges=((-1,-0.5),(0.5,1.0)))`,
   calls a generator from `sim_base.py`, and saves an `.npz` holding
   `X, W, mask, Z, Um, Ur, Vm, Vr`.
2. **Fit** — `job_simu*.py` loads that `.npz`, fits RICE, and writes one CSV of
   long-format `measure,value` rows per replicate.

Paths are hardcoded absolute, so scripts must run with cwd `$SCGRAPH_DIR`
(each driver does `os.chdir` + `sys.path.append` on that literal path):

| stage | path template |
|---|---|
| data | `$RESULT_DIR/simu_{intervention}/data/{model}/n{n//1000}k_p{p}_s{s0}_{graph_type}[_TAG]_repl{job_id}.npz` |
| results | `$RESULT_DIR/simu_{intervention}/scGRAPH/{model}/...repl{job_id}.csv` |

`{intervention}` is `soft` or `hard` and also sets `hard_intervention` in the generator.
`TAG` distinguishes scenarios: `_MOI{k}`, `_missing{k}`, `_um{k}`, `_um{k}_hm`.

## Model API — `scRICE_CF` (scRICE_CF.py, 554 lines, CuPy)

Single class, GPU-only (`import cupy as cp` at module scope; `dtype=cp.float64` default,
with `clip_value` 300 for f64 / 30 for f32). Call order is `prep` -> `fit` -> `result`:

- `prep(X, intervention_effect, confounding_res=None, predictor=None, intervention_type='soft', Zm_g=, Zr_g=, Zm_p=, Zr_p=, add_intercept=True, Gamma=, weights=)`
- `fit(pen1=0.03, pen2=0.03, pen1_c=0.0, pen2_c=0.0, regularizer='logdet', s=[1.0,.9,.8,.7,.6], epochs=1e4, max_iter=100, lr=0.0003, rho_init=1.0, rho_multiplier=10.0, rho_max=1e16, h_tol=1e-3, binsum_k=7, optim_param=(0.9,0.99), ...)`
- `result(w_threshold=0.0) -> {'W': ...}`; `fit_path(pen1_list, ...)` sweeps `pen1`.

Acyclicity is available as `logdet` or `expm` (`_h_value` / `_h_grad`), with
`trace_binsum` / `block_mat_sum` truncated matrix-power variants controlled by `binsum_k`.

### Settings actually used in `job_simu.py` (these override the signature defaults)

    regularizer='expm', binsum_k=p, pen1=1e-4, pen2=0, pen1_c=0, pen2_c=1e-2,
    lr=1e-4, max_iter=100, checkpoint=2000, h_tol=1e-2, loss_tol=1e-6
    W_est = result(w_threshold=0.3) -> to_dag(...) -> measurement(W_true, W_est)

So the manuscript's lambda1 = 1e-4 maps to `pen1` and lambda2 = 0.01 to `pen2_c`.
Edge threshold 0.3 is applied before `to_dag`; AUC is computed on the unthresholded `W`.
Reported metrics per replicate: the keys of `measurement()`, plus `auc`, `time`,
`peak_gpu_mem_mb` (sampled by a PID-scoped NVML monitor thread, MIG-correct).

### First-stage residuals are built in the driver, not the class

`job_simu.py` computes the reduced control function explicitly before `prep`:
`predictor = log(1+X)`, then for each gene `i` an OLS of `predictor[:,i]` on
`mask[:,i]` alone, with `res[:,i]` the residual — i.e. each gene regressed on its OWN
perturbation indicator only, which is exactly the "reduced" formulation. `res` is passed
as `confounding_res`. A full control function would regress on all of `mask`.

## Generators in `sim_base.py` (902 lines)

| generator | extra knobs | used by | in manuscript |
|---|---|---|---|
| `simulate_Pois_tgt` | — | `job_gen.py`, `job_gen_cova.py` | yes (Poisson arm) |
| `simulate_NB_tgt` | — | `job_gen.py`, `job_gen_missing.py`, `job_gen_cova.py` | yes (default NB arm) |
| `simulate_PLN_tgt` | `noise_scale` | `job_gen.py`, `job_gen_cova.py` | yes (PLN arm, noise_scale=0.5) |
| `simulate_NB_highMOI` | `MOI` | `job_gen_highMOI.py` | yes (Fig 4B) |
| `simulate_NB_hidden_mediator` | `mediator_density=0.05`, `mediator_in_effect=3.0`, `mediator_out_effect=3.0` | `job_gen_cova.py` (ACTIVE) | matches the exclusion-restriction violation in Methods (5% of pairs, beta in [-3,3]) |
| `simulate_NB_structured` | `n_modules=5`, `module_noise_scale`, `module_assignment` | `job_gen_cova.py` (commented out) | **no** |
| `simulate_NB_perturbation_interaction` | `interaction_strength=2.0` | **nothing** | **no** |
| `simulate_NB_mixture` | `n_states=3`, `state_effect_scale`, `state_perturbation_scale` | **nothing** | **no** |

Shared knobs: `n`, `d` (number of unmeasured confounders), `shift`, `coef`, `covar`,
`intervene_prob=0.9`, `hard_intervention`, and `g` (the monotone transform, normally
`lambda x: np.log(1+x)`).

## Scenario drivers present

| driver | varies | notes |
|---|---|---|
| `job_simu.py` | base case | default `n=10000, p=100, s0=400, ER`; large arm `100000/500/2000` |
| `job_simu_comp_cf.py` | full vs reduced control function | Figure 2 |
| `job_simu_cova.py` | `um` — unmeasured-confounder count | Figure 4A |
| `job_simu_highMOI.py` | `MOI` | Figure 4B |
| `job_simu_missing.py` | `missing` — unobserved genes | see below; `missing=50` in `job_simu.sh` |
| `job_simu_cpd.py` | (changepoint / cpd variant) | not yet read |
| `job_simu_ebic.py` | eBIC-based selection | not yet read |

### `job_gen_missing.py` already implements partial network observation
It builds the DAG on `p + missing` genes with `s0*(1+missing/p)**2` edges, simulates
`n*(1+missing/p)` cells, then keeps only cells in which none of the extra `missing`
genes were perturbed and truncates `X`, `mask`, `W_true` to the first `p`. The retained
network therefore has unobserved parents — the standard "not every gene is measured or
perturbed" robustness check. This is not reported in the manuscript.

## Baselines
`dotears.py` + `dotears_run.py`; `IBCD-main/` (vendored, `src/ibcd.py`, plus
`IBCD_prepare.py` / `IBCD_postrun.py` and `_highMOI` variants); inspre via
`inspre_postrun.py` and `inspre_h5.ipynb` (R side in `*.R`, 28 files). Each has its own
`job_*_simu*.sh`.

## Submission pattern — and a discrepancy worth noting

`job_simu.sh` requests:

    #BSUB -J simu[1-20] -q dbeigpu -gpu "num=1:gmem=15G" -n 1 -R "rusage[mem=16G]" -W 24:00
    module load cuda/12.8

That is `-n 1` and `gmem=15G`, whereas the conventions recorded for this cluster are
`-n 2` with `gmem=16G` for arrays. Worth reconciling before resubmitting, since
`gmem=15G` may land a different MIG geometry than the 16 GiB slice assumed elsewhere.
The script drives scenarios by commenting/uncommenting `python job_simu_*.py` lines
rather than by argument, so any new scenario follows that same convention.

## Current state of the working copy

`job_gen_cova.py` is the live scratch driver. Its active line calls
`simulate_NB_hidden_mediator` (output tagged `_hm`), while two alternatives sit
commented immediately above it: `simulate_NB_structured`, and a misspecified transform
`g = lambda x: (np.log(1+x**3))**(1/3)` in place of `log(1+x)`. So exclusion violation,
module-structured confounding, and a misspecified `g` are all mid-experiment;
`simulate_NB_perturbation_interaction` and `simulate_NB_mixture` are written but never
called from any driver.


---

# Findings from a full read of the core files

Read in full: `utils.py`, `sim_base.py`, `scRICE_CF.py`, all four `job_gen*.py`,
all six `job_simu*.py`, `dotears.py` + `dotears_run.py`, the four IBCD prepare/postrun
scripts, `inspre_postrun.py`, and `job_simu.sh`. Items are ordered by how much they
affect the revision; some may be deliberate, but each is a place where the code and the
manuscript (or the code and itself) disagree.

## 1. `pen1` in the scenario arms is 1e-3, not the 1e-4 the manuscript reports

Methods states lambda1 = 1e-4 "unless otherwise specified". Only the base driver uses it:

| driver | pen1 | pen2_c | w_threshold | h_tol |
|---|---|---|---|---|
| `job_simu.py` | **1e-4** | 1e-2 | 0.3 | 1e-2 |
| `job_simu_comp_cf.py` | **1e-3** | 1e-2 | 0.3 | 1e-2 |
| `job_simu_cova.py` | **1e-3** | 1e-2 | 0.3 | 1e-2 |
| `job_simu_missing.py` | **1e-3** | 1e-2 | 0.3 | 1e-2 |
| `job_simu_highMOI.py` | **1e-3** | 1e-2 | **0.2** | **1e-4** |
| `job_simu_cpd.py` | **1e-3** | 1e-2 | sweep 0.1-0.5 | **1e-4** |

So Figures 2, 4A and 4B were produced at a sparsity penalty 10x the reported value, and
the MOI arm additionally used a different edge threshold (0.2 vs 0.3) and a stricter
acyclicity tolerance. The MOI panel is therefore not threshold-comparable with the base
benchmark. Either the Methods text needs a per-figure table, or the arms need rerunning
at a common setting.

## 2. `job_simu_missing.py` never passes the control function

It computes the first-stage residuals `res` exactly as the other drivers do, then calls

    GNB.prep(X=X, predictor=predictor, intervention_effect=mask,
             intervention_type=intervention, add_intercept=True)   # no confounding_res

`confounding_res` is omitted, so `gX_c` is zeros and the `W_c` block is inert — this arm
is RICE *without* RCF. The output name `..._noCF.csv` suggests that was intended, but
there is no with-CF counterpart driver, so the partial-observation scenario currently
cannot show what RCF contributes.

## 3. High-MOI results are written into the `cpd/` directory

    results.to_csv(f'.../simu_{intervention}/scGRAPH/cpd/n..._MOI{MOI}_repl{job_id}.csv')

`job_simu_highMOI.py` writes under `scGRAPH/cpd/`, not `scGRAPH/{model}/`. It also
imports `ruptures` and `kneed` without using them. Both look like leftovers from copying
`job_simu_cpd.py`.

## 4. `job_gen_cova.py` hardcodes the `_hm` tag for all three of its generators

The `savez` path is fixed at `..._um{um}_hm_repl{job_id}.npz` while the generator is
selected by commenting lines. Uncommenting `simulate_NB_structured` or the cubic-`g`
variant writes module-confounding or misspecified-`g` data **into the hidden-mediator
filename**, silently overwriting it. Compounding this, `job_simu_cova.py` reads
`..._um{um}_cubic_repl{job_id}.npz` — a tag the current generator never produces. Any
new scenario should take its tag from an argument rather than a comment toggle.

## 5. Count-rescaling constants are inconsistent across generators

The heuristic that keeps counts in range differs by code path:

| code path | max cap | mean floor | used by |
|---|---|---|---|
| module-level `_simulate_single_equation` | 7.5 | **-0.5** | structured, interaction, mixture, hidden_mediator |
| inner copy in `simulate_NB_tgt` | **6.5** | 0.5 | the default NB arm |
| inner copies in `simulate_Pois_tgt`, `simulate_PLN_tgt` | 7.5 | 0.5 | Poisson, PLN arms |

Methods p.29 describes 7.5 and 0.5, which matches only the Poisson/PLN path. The default
NB arm caps at 6.5, and every newer generator uses a mean floor of -0.5 — a materially
sparser count regime. So a new DGP built on the module-level helper is **not on the same
count scale** as the NB baseline it would be compared against. Note also
`simulate_Pois_tgt` uses `elif` for the mean adjustment (so it is skipped whenever the
max cap fires) while the others use `if`.

## 6. `simulate_NB_perturbation_interaction` returns misaligned covariates

It hides the last column of `Z` to act as the latent confounder
(`Z_observed = Z[:, :-1]`) but returns `Um`/`Ur` at full width. The saved `Um`/`Ur` then
have one more row than `Z_observed` has columns, so any downstream use of them together
is misaligned. It also returns 7 values where `structured`, `mixture`, and
`hidden_mediator` return 8 — drivers must unpack per-generator. Neither this generator
nor `simulate_NB_mixture` has ever been run.

## 7. `job_simu_cpd.py` and `job_simu_ebic.py` read `_noZ` inputs that the NB branch never writes

Both load `..._repl{job_id}_noZ.npz`. In `job_gen.py` the `_noZ` suffix is written only
in the PLN and Pois branches; the NB/GNB branch writes without it. Running either driver
with `model=NB` will not find its input.

## 8. Smaller notes

- `job_simu_ebic.py` saves `X, mask, res, predictor` into the path `.npz` alongside the
  results; at n=100k, p=500 float64 that is roughly 400 MB per array, per replicate.
- `IBCD_prepare_highMOI.py` builds its DataFrame by `pd.concat` of one row at a time
  inside a double loop — quadratic, and slow at n=10,000.
- `to_dag` runs a fresh `igraph` DAG check for every candidate edge, so cost grows with
  the number of nonzeros in `W`; at p=500 that is thousands of `is_dag` calls.
- `scRICE_CF.result()` converts every output to NumPy except `W_c`, which is returned as
  a CuPy array.
- In `run()`, `rho` is only escalated when `optimize_param` reports success; on the
  `logdet` path a failure instead raises `s[i]` by 0.1 and retries, so termination there
  depends on `s` growing until the barrier is feasible. The `expm` path used in all
  simulation drivers always reports success, so this does not arise there.
- `prep()` asserts at least one perturbation per gene AND at least one fully unperturbed
  cell (`(intervention_effect!=0).sum(axis=1).min() == 0`) — worth remembering for any
  high-MOI design where control cells might be dropped.
- Baseline settings for the record: dotears `lambda1=1e-4, h_tol=1e-4`, edges thresholded
  at 0.1; IBCD edges kept at `pip > 0.95`; both consume `log(1+X)` mean-centred, not
  variance-scaled. `simulate_dag` supports only ER, SF, BP — the hub/chain/clusters
  topologies in Figure 3A must be curated outside these files.
