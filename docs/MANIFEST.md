# File manifest

Every file in the repository, what it does, and how far it has been verified.

| tag | meaning |
|---|---|
| original | unchanged from the previous version of this repository |
| self-test passes | `--selftest` was run and passed; for the generators it checks that the zero setting reproduces `sim_base` |
| stand-in tested | exercised against synthetic arrays in the real data layout, not on real data |
| GPU only | imports CuPy at module scope; cannot be validated without a GPU |
| syntax only | parses, or passes `bash -n`; never executed |

All eight self-tests pass, including `sim_nonlinear_confounding.py`, which had not been run
before this cleanup.

### `source/`

- **__pycache__** [doc]
- **scRICE_CF.py** [GPU only]
- **scRICE_eff.py** [GPU only] — scRICE with learnable per-gene knockdown efficacy. Extends the published scRICE_CF model by replacing the binary soft/hard interve
- **utils.py** [original]

### `simulation/`

- **__pycache__** [doc]
- **analysis** [doc]
- **drivers** [doc]
- **example.ipynb** [doc]
- **figures** [doc]
- **gen_simulation.py** [original]
- **run_simulation.py** [GPU only]
- **sim_base.py** [syntax only]
- **sim_crossguide.py** [syntax only] — Exclusion-restriction violation by DIRECT CROSS-GENE GUIDE EFFECT (design item B2). For a fraction of ordered pairs (k, j), k != j
- **sim_dependent_assignment.py** [self-test passes] — VIOLATION OF THE INDEPENDENCE ASSUMPTION: effective uptake depends on latent cell state. Every other run_sensitivity_arms arm leav
- **sim_guide_mismatch.py** [syntax only] — Guide-assignment error (mismatched guide) for Perturb-seq simulations. The data are generated from the TRUE assignment by any of t
- **sim_guide_mismatch_cells.py** [self-test passes] — Generate data in which A FRACTION OF CELLS HAS A MISMATCH between the perturbation it actually received and the perturbation indic
- **sim_invalid_guides.py** [self-test passes] — Generate data with INVALID MATCHED GUIDES at a sweep of prevalences. A fraction of guides is invalid: guide k acquires a direct ef
- **sim_nonlinear_confounding.py** [self-test passes] — Nonlinear latent confounding: two data-generating processes. Both act ONLY on the latent contribution to gene j's linear predictor
- **sim_partial_efficacy.py** [syntax only] — Partial knockdown: upstream regulation is retained in perturbed cells in proportion to a per-gene efficacy e_j in [0, 1]. e_j = 0 
- **sim_sparse_confounding.py** [self-test passes] — SPARSE NONLINEAR CONFOUNDING: only a fraction of genes is confounded. The mechanism and its parameters are FIXED; the only thing t
- **submit** [doc]
- **utils_gen.py** [original]

### `simulation/drivers/`

- **fit_cf_comparison.py** [GPU only] — Fit rcf / fcf / nocf on each configuration of one replicate. rcf reduced control function -- gene i regressed on its OWN indicator
- **fit_crossguide.py** [GPU only] — Fit RICE under a direct cross-gene guide effect, comparing control-function variants. Three arms on the same dataset: rcf reduced 
- **fit_efficacy.py** [GPU only] — Fit three intervention parameterizations to one efficacy dataset: hard xi fixed at 0 (published model, intervention_type='hard') s
- **fit_mismatch.py** [GPU only] — Fit RICE under guide misassignment, comparing control-function variants. Three arms on the same dataset, all using the RECORDED (c
- **generate_cf_comparison.py** [syntax only] — Generate all four control-function comparison configurations from ONE DAG per replicate. The array index is the replicate. Every c
- **generate_confounding.py** [syntax only] — Generate the latent-confounding sweep (d = 0, 2, ..., 10) used for the confounding figure. One replicate per invocation; the repli
- **generate_crossguide.py** [syntax only] — Generate data with a direct cross-gene guide effect (exclusion violation, design B2). usage: python generate_crossguide.py <model>
- **generate_efficacy.py** [syntax only] — Generate Perturb-seq data with per-gene knockdown efficacy. xi_j ~ Uniform(center - half_width, center + half_width), clipped to [
- **generate_mismatch.py** [syntax only] — Generate Perturb-seq data, then corrupt the RECORDED guide assignment. X is generated from the TRUE mask by the published generato
- **patch_count_rescale.py** [syntax only] — Fix the count-rescale overflow in the negative-binomial simulators. python patch_count_rescale.py sim_base.py sim_crossguide.py TH
- **run_local_demo.py** [syntax only] — Local (CPU) KD-efficacy simulation. Reduced scale relative to the cluster pipeline in eff_pipeline/, but fully replicated and para

### `simulation/submit/`

- **job_cfcompare.sh** [syntax only]
- **job_cfcompare_cpu.sh** [syntax only]
- **job_cfcompare_single.sh** [syntax only]
- **job_crossguide.sh** [syntax only]
- **job_crossguide_cpu.sh** [syntax only]
- **job_eff.sh** [syntax only]
- **job_eff_cpu.sh** [syntax only]
- **job_gen_crossguide.sh** [syntax only]
- **job_gen_eff.sh** [syntax only]
- **job_gen_invalid.sh** [syntax only]
- **job_gen_mismatch.sh** [syntax only]
- **job_gen_mismatch_cells.sh** [syntax only]
- **job_gen_nonlinear.sh** [syntax only]
- **job_gen_selection.sh** [syntax only]
- **job_gen_sparseconf.sh** [syntax only]
- **job_mismatch.sh** [syntax only]
- **job_mismatch_cpu.sh** [syntax only]
- **job_sensitivity.sh** [syntax only]

### `simulation/analysis/`

- **__pycache__** [doc]
- **compare_control_functions.py** [GPU only] — Control-function comparison, generation through fitting, in one file. python compare_control_functions.py --stage both # one repli
- **confounder_effect_size_measured.py** [syntax only] — Effect size of the unmeasured confounders, measured from a stored simulation file. Nothing is re-simulated: job_gen_cova.py saves 
- **confounder_effect_size_theory.py** [syntax only] — Latent-confounder effect sizes for the d = 2..10 sweep, for ALL THREE NB generators that job_gen_cova.py can produce, because the 
- **ordering_baselines.py** [self-test passes] — Is the variance ordering ENOUGH to recover the graph? Varsortability says the causal order is almost readable off the marginal var
- **read_assignment_correlation.py** [syntax only] — Realised dependence per selection strength, for the brackets under the figure's x axis. python read_assignment_correlation.py resu
- **run_sensitivity_arms.py** [GPU only] — Sensitivity of RICE to INVALID MATCHED GUIDES and to NONLINEAR CONFOUNDING. Two mechanisms, each with a severity sweep whose zero 
- **sortability.py** [syntax only] — Sortability statistics and the ordering baseline, shared by the scripts in this directory. These four functions were previously co
- **sortability_stats.py** [self-test passes] — Sortability statistics for the stored simulation files. Reports, per .npz and aggregated per confounding setting: vs_counts varsor
- **sortability_sweep.py** [self-test passes] — Complete sortability / order-sufficiency result over the stored confounding sweep. Covers both intervention regimes, every confoun
- **summarize_efficacy.py** [GPU only] — Summarise and plot the adaptive-efficacy job array. Reads the per-replicate CSVs written by fit_efficacy.py, aggregates over repli

### `simulation/figures/`

- **make_ordering_table_tex.py** [stand-in tested] — LaTeX table: RICE against the three ordering baselines, per intervention and per confounding setting. Baselines are read from sort
- **plot_confounding_sweep.py** [stand-in tested] — Figure for the latent-confounding sweep: recovery metrics against the number of latent factors.
- **plot_control_function_violation.py** [stand-in tested] — Control-function comparison across exclusion-restriction violation regimes. Reads the per-replicate CSVs written by fit_cf_compari
- **plot_selection_brackets.py** [stand-in tested] — Annotation row for the dependent-assignment figure: the realised correlation between the perturbation indicator and the latent sta

### `perturbseq/`

- **cells** [doc]
- **compute_ate.py** [syntax only] — Vanilla average treatment effects for every (perturbed gene i -> gene j) pair. This is the same regression real_differential_analy
- **estimate_confounding.py** [syntax only] — Estimate how confounded a real CRISPRi Perturb-seq screen is, on the scale of the simulated latent-factor sweep. WHAT THIS DOES AN
- **export_enrichment_tables.py** [syntax only] — Full GO enrichment result for all modules, as one CSV -- queried the way the original plotting loop queries, with no cache in the 
- **run_Replogle_2022.py** [GPU only]
- **topk_significance.py** [syntax only] — Proportion of significant perturbation effects among the top-k edges, for several network-inference methods at matched edge counts
- **write_significance_csvs.py** [syntax only] — {ctype}_{method}_significance_{te,de}.csv for ALL FOUR methods. Generalizes real_differential_analysis.ipynb cell 68 (= real_soft_

### `perturbseq/cells/`

- **assignment_correlation.py** [stand-in tested] — Realised dependence per selection strength, for the brackets under the x axis. The generator stores corr_perturbed_u in each .npz'
- **ate_acyclicity.py** [stand-in tested] — Build a graph from the FDR-significant ATEs and test whether it is acyclic. Edge i -> j iff the ATE of perturbing i on j is signif
- **ate_agreement.py** [stand-in tested] — How closely does each method agree with the vanilla ATE? Three axes, because "agreement" means different things and the fair compa
- **ate_agreement_figure.py** [stand-in tested] — Compare the ATE-agreement metrics across the four methods, both cell lines. Reads {ds}_ate_method_agreement.csv for each dataset. 
- **ate_estimate.py** [stand-in tested] — Vanilla average treatment effects for every (perturbed gene i -> gene j) pair. Same regression as the total-effect p-value loop (g
- **ate_rank_alignment.py** [stand-in tested] — How well does each method's edge RANKING align with the measured ATE? For each method, take its top k edges by |W| (IBCD by PIP if
- **ate_reachability.py** [stand-in tested] — RICE vs the vanilla ATE: the ATE measures total effects, which a causal DAG predicts topologically, but the ATE is not itself a ca
- **confounding_estimate.py** [stand-in tested] — How confounded is this screen, on the scale of the simulated latent-factor sweep? Uses the objects cell 19 already built: data_X_c
- **covariate_direction.py** [stand-in tested] — Which way does the association run? (Follow-up to the balance diagnostics.) The balance cell found that perturbation identity expl
- **effect_size_figure.py** [stand-in tested] — Effect sizes at matched top-k, both cell lines, drawn to match significant_proportion_topk.pdf: columns are cell lines, rows are e
- **effect_sizes.py** [stand-in tested] — Effect sizes for the perturbation tests, and the matched-top-k figure. Self-contained: recomputes the regressions rather than rely
- **go_enrichment_figure.py** [stand-in tested] — GO enrichment dot plot, one figure per ontology, ALL clusters and ALL terms. Differences from the previous version: * Biological P
- **perturbation_balance.py** [stand-in tested] — Is perturbation assignment associated with cell-level covariates, and does effective uptake depend on them? (Reviewer Comment 5.) 
- **significance_figure.py** [stand-in tested] — Top-k significance figure, both datasets, read from the per-method CSVs written by write_significance_csvs.py: {ctype}_{method}_si

### `docs/`

- **MANIFEST.md** [doc]
- **README_efficacy.md** [doc]
- **additional_simulations_design.md** [doc]
- **design_independence_violation.md** [doc]
- **efficacy_diagnostics.md** [doc]
- **scgraph_code_map.md** [doc]

## Consolidated in this cleanup

**One generator.** `simulate_NB_tgt`, `_simulate_single_equation` and `is_dag` had been
copied into `compare_control_functions.py` and `run_sensitivity_arms.py`, where the copies
carried an overflow guard the base module lacked. They now live only in `sim_base.py`.
Verified: with the default `apply_clip=False`, `sim_base.simulate_NB_tgt` reproduces the
previous `simulation_soft_IV.simulate_NB_tgt` bit-for-bit on the same seed.

**The overflow guard is opt-in and is not cosmetic.** The mean floor can undo the cap when
one cell is extreme; `exp(m)` then exceeds the representable integer range and `nbinom.rvs`
returns negative counts. `apply_clip=True` clips elementwise at the cap and prevents that,
but it also changes ordinary draws: at p=12, n=400, d=3 with g=log(1+x) it moved the maximum
count from 7518 to 376 on one seed. Clipped and unclipped runs must not be pooled. The two
scripts that previously always clipped now pass `apply_clip=True` explicitly, so their
behaviour is unchanged.

**One sortability module.** `parse_tag`, `varsortability`, `r2sortability` and `sortnregress`
were copied into three scripts and had drifted. They now live in
`simulation/analysis/sortability.py`; the bodies kept are the ones the published sweep ran.
A stale docstring was corrected: it described the L1 penalty as chosen by cross-validation
where all three copies in fact used `LassoLarsIC(criterion='bic')`.

`stem_for` and `validate_counts` remain per-file: each arm names its outputs after its own
parameters and checks its own severity knob, so these are different functions sharing a name,
not copies.

## Renamed

| was | is |
|---|---|
| `ate_acyclicity_cell.py` | `ate_acyclicity.py` |
| `ate_agreement_cell.py` | `ate_agreement.py` |
| `ate_agreement_plot_cell.py` | `ate_agreement_figure.py` |
| `ate_alignment_cell.py` | `ate_rank_alignment.py` |
| `ate_cell.py` | `ate_estimate.py` |
| `ate_vs_rice_cell.py` | `ate_reachability.py` |
| `cfcompare.py` | `compare_control_functions.py` |
| `check_sortability.py` | `sortability_stats.py` |
| `confounder_effect_size.py` | `confounder_effect_size_theory.py` |
| `confounder_effect_size_from_npz.py` | `confounder_effect_size_measured.py` |
| `confounding_cell.py` | `confounding_estimate.py` |
| `covariate_direction_cell.py` | `covariate_direction.py` |
| `effect_size_topk_styled.py` | `effect_size_figure.py` |
| `effect_sizes_and_plot_cell.py` | `effect_sizes.py` |
| `enrichment_all_clusters.py` | `go_enrichment_figure.py` |
| `estimate_confounding_crispri.py` | `estimate_confounding.py` |
| `export_enrichment_csv.py` | `export_enrichment_tables.py` |
| `job_gen_cfcompare.py` | `generate_cf_comparison.py` |
| `job_gen_cova_restored.py` | `generate_confounding.py` |
| `job_gen_crossguide.py` | `generate_crossguide.py` |
| `job_gen_eff.py` | `generate_efficacy.py` |
| `job_gen_mismatch.py` | `generate_mismatch.py` |
| `job_simu_cfcompare.py` | `fit_cf_comparison.py` |
| `job_simu_crossguide.py` | `fit_crossguide.py` |
| `job_simu_eff.py` | `fit_efficacy.py` |
| `job_simu_mismatch.py` | `fit_mismatch.py` |
| `latex_table_cell.py` | `make_ordering_table_tex.py` |
| `make_significance_csvs.py` | `write_significance_csvs.py` |
| `order_sufficiency.py` | `ordering_baselines.py` |
| `patch_rescale.py` | `patch_count_rescale.py` |
| `perturbation_balance_cell.py` | `perturbation_balance.py` |
| `plot_cfm_violation.py` | `plot_control_function_violation.py` |
| `plot_simu_cova.py` | `plot_confounding_sweep.py` |
| `read_selection_corr.py` | `read_assignment_correlation.py` |
| `run_local_sim.py` | `run_local_demo.py` |
| `selection_brackets_snippet.py` | `plot_selection_brackets.py` |
| `selection_corr_cell.py` | `assignment_correlation.py` |
| `sensitivity.py` | `run_sensitivity_arms.py` |
| `simulation_crossguide.py` | `sim_crossguide.py` |
| `simulation_efficacy.py` | `sim_partial_efficacy.py` |
| `simulation_invalid_guides.py` | `sim_invalid_guides.py` |
| `simulation_mismatch.py` | `sim_guide_mismatch.py` |
| `simulation_mismatch_cells.py` | `sim_guide_mismatch_cells.py` |
| `simulation_nonlinear.py` | `sim_nonlinear_confounding.py` |
| `simulation_selection.py` | `sim_dependent_assignment.py` |
| `simulation_soft_IV.py` | `sim_base.py` |
| `simulation_sparse_confounding.py` | `sim_sparse_confounding.py` |
| `topk_cell.py` | `significance_figure.py` |

## Removed

- `effect_size_both_cell.py` — superseded by effect_size_figure.py, which draws both cell lines
- `confounding_in_real_data.py` — superseded by estimate_confounding.py

