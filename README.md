# RICE 

**R**obust **I**dentification of **C**ausal **E**ntanglements

RICE is a framework for identifying causal relationships from perturbation-based single-cell
data. It models raw UMI counts with negative binomial regression, uses transformed expression
as candidate regulatory predictors, and estimates a sparse directed acyclic network under an
acyclicity constraint. A gene-matched residual adjustment reduces spurious edges caused by
shared unmeasured variation.

## Quick start

```bash
git clone https://github.com/HowardGech/RICE.git
cd RICE
pip install -r requirements.txt          # CuPy must match your CUDA version
export PYTHONPATH=$PWD/source:$PWD/simulation:$PWD/perturbseq
```

Model fitting requires a GPU: `source/scRICE_CF.py` imports CuPy at module scope and has no
NumPy fallback. Data generation and all analysis scripts run on CPU.

`simulation/example.ipynb` is the shortest path from a clone to a fitted network.

## Repository structure

```
source/        the RICE model
simulation/    data generators, cluster drivers, analysis and figures
perturbseq/    real CRISPRi screens: analysis scripts and notebook cells
docs/          design notes, a code map, and a per-file manifest
```

## Source

- `scRICE_CF.py` — the RICE model
- `scRICE_eff.py` — RICE with a per-gene efficacy parameter `e_j` in [0, 1] scaling the
  upstream regulation retained in perturbed cells. `e_j = 0` and `e_j = 1` reproduce the
  fixed soft and hard masks of `scRICE_CF.py`; intermediate values represent partial
  knockdown, and `e_j` can be estimated jointly with the network.
- `utils.py` — graph simulation, seeding, and the `measurement` scoring used everywhere

## Simulation

### Basic generators

- `gen_simulation.py` — generates a synthetic dataset. Arguments, in order:
  1. number of cells
  2. number of genes
  3. number of true causal effects
  4. graph type: `ER` (Erdős–Rényi), `SF` (scale-free), `BP` (bipartite)
  5. intervention type: `soft` or `hard`
  6. random seed
- `run_simulation.py` — runs RICE on a generated dataset; same arguments
- `utils_gen.py` — `simulate_Pois`, `simulate_NB`, `simulate_PLN`, `simulate_NB_highMOI`
- `example.ipynb` — generate, fit, and score a network end to end

### Targeted-intervention generator

`sim_base.py` provides `simulate_NB_tgt` and relatives (`simulate_Pois_tgt`,
`simulate_PLN_tgt`, `simulate_NB_highMOI`, `simulate_NB_structured`,
`simulate_NB_perturbation_interaction`, `simulate_NB_mixture`,
`simulate_NB_hidden_mediator`). It is the generator the run_sensitivity_arms arms below are built on,
and every module in the next section imports it. Its `apply_clip` argument guards against an
overflow that otherwise yields negative counts; it defaults to `False`, which reproduces the
published generator exactly, and `True` changes ordinary draws as well, so the two must not be
pooled. See `docs/MANIFEST.md`.

### Sensitivity arms

Each generator takes a severity parameter whose zero setting reproduces `simulate_NB_tgt`
exactly, so an arm can be compared against a no-violation reference generated from the same
graph and the same random state.

| module | violation |
|---|---|
| `sim_crossguide.py` | a guide has a direct effect on a gene other than its target |
| `sim_invalid_guides.py` | a fraction of matched guides act on the wrong gene |
| `sim_nonlinear_confounding.py` | nonlinear, non-additive latent-factor effects |
| `sim_dependent_assignment.py` | perturbation assignment depends on the latent state |
| `sim_sparse_confounding.py` | latent factors load on a sparse subset of genes |
| `sim_partial_efficacy.py` | partial knockdown at a specified per-gene efficacy |
| `sim_guide_mismatch.py` | guide-to-cell misassignment at the perturbation level |
| `sim_guide_mismatch_cells.py` | guide-to-cell misassignment at the cell level |

Naming convention: `sim_*` generates data, `generate_*` is the per-replicate driver that
writes one file, `fit_*` runs RICE on it, and `submit_*.sh` launches the array.

Every generator ships a `--selftest` that checks its zero setting against `sim_base`. All of
them pass as committed; run one after any change:

```bash
SCGRAPH_DIR=$PWD/simulation python simulation/sim_nonlinear_confounding.py --selftest
```

### Drivers, submission scripts, analysis, figures

- `drivers/` — one `generate_<arm>.py` and one `fit_<arm>.py` per experiment, plus
  `run_local_demo.py` for a small CPU-only run and `patch_count_rescale.py`
- `submit/` — LSF array scripts. Each resolves the code directory as
  `CODE="${SCGRAPH_DIR:-<repo>/simulation}"` and honours `DATA` and `PY`; the `*_cpu.sh`
  variants run the generation half without a GPU.
- `analysis/` — `compare_control_functions.py` (reduced vs full vs no control function),
  `run_sensitivity_arms.py`, `sortability_stats.py`, `ordering_baselines.py`, `sortability_sweep.py`
  (random, variance and oracle ordering baselines), `summarize_efficacy.py`,
  `confounder_effect_size_theory*.py`, `read_assignment_correlation.py`
- `figures/` — `plot_control_function_violation.py`, `plot_confounding_sweep.py`, `make_ordering_table_tex.py`,
  `plot_selection_brackets.py`

## Perturb-seq

- `run_Replogle_2022.py` — causal gene networks in K562 and RPE1 from
  [Replogle et al. (2022)](https://www.sciencedirect.com/science/article/pii/S0092867422005979).
  Processed data: [Figshare](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387).
- `topk_significance.py` — proportions of significant perturbation effects among each
  method's top-`k` edges, with Fisher and paired cluster bootstrap tests
- `write_significance_csvs.py` — writes the per-dataset CSVs the figure cells read
- `compute_ate.py` — average treatment effect for every ordered gene pair
- `estimate_confounding.py` — latent-variation estimate on the real screens, calibrated
  against the simulated sweep
- `export_enrichment_tables.py` — enrichment tables across all modules
- `cells/` — notebook cells, meant to be pasted into an analysis notebook rather than
  imported: `effect_sizes.py` and `effect_size_figure.py`, `significance_figure.py`,
  the `ate_*.py` family, `go_enrichment_figure.py`, `covariate_direction.py`,
  `perturbation_balance.py`, `confounding_estimate.py`, `assignment_correlation.py`

## Paths

No script contains an absolute path. Data and result directories default to `./result`,
`./data` and `./log` relative to the working directory, and the submission scripts take
`DATA`, `PY` and `SCGRAPH_DIR` from the environment. Set `SCGRAPH_DIR` to the `simulation/`
directory when a script is run from elsewhere.

## Status of the code in this repository

Not every file here has been executed in this form. `docs/MANIFEST.md` tags each file:

| tag | meaning |
|---|---|
| self-test | ships a `--selftest` reproducing the published generator at its zero setting |
| stand-in tested | exercised against synthetic arrays in the real data layout, not on the real data |
| GPU only | imports CuPy at module scope; cannot be validated without a GPU |
| syntax only | parses or passes `bash -n`; never executed |

## Documentation

- `docs/MANIFEST.md` — every file, its status tag, what it does, and a table mapping the
  previous file names to the current ones
- `docs/scgraph_code_map.md` — map of the codebase
- `docs/additional_simulations_design.md` — design of the run_sensitivity_arms arms
- `docs/design_independence_violation.md` — the independence-violation arm
- `docs/efficacy_diagnostics.md`, `docs/README_efficacy.md` — the efficacy extension

## License

See `LICENSE`.
