# Additional simulations for reviewer comment on simulation-dependence

Two separable objections. Experiment A answers the scale / varsortability half,
Experiment B answers the exclusion-violation-mechanism half. They share no code and
can run independently.

---

## Measured starting point (not a design choice -- this is your current DGP)

`simulate_NB_tgt`, p=100, s0=400, n=10,000, ER, soft, 3 seeds:

| statistic | d = 0 | d = 10 |
|---|---|---|
| varsortability, raw counts | 0.992 - 0.997 | 0.969 - 0.980 |
| varsortability, log(1+y) | 0.992 - 0.996 | 0.969 - 0.987 |
| varsortability, column-standardized log(1+y) | 0.500 | 0.500 |
| R2-sortability, log(1+y) | 0.758 - 0.830 | 0.419 - 0.536 |

Varsortability (Reisach et al., 2021) is the fraction of directed paths along which
marginal variance increases; 0.5 is chance, 1.0 means variance ordering reproduces the
topological order exactly. At ~0.98 the reviewer's concern is well founded and must be
addressed rather than argued away.

R2-sortability (Reisach et al., 2023) is the scale-invariant analogue. That it falls to
chance at d=10 is a genuinely favourable fact: the confounded settings, which are the
ones the paper's main claims rest on, are already not exploitable by ordering heuristics.

---

## Experiment A -- is the performance gain a scale artifact?

### A1. Report the statistic for every setting already in the paper

Compute varsortability and R2-sortability for each simulation configuration (DGP family,
topology, n, p, d, MOI) and put the table in the supplement. Costs no fitting: the data
already exist. This alone answers "the authors should assess whether".

### A2. Add the trivial ordering baselines

The decisive test in Reisach et al. is whether a method that uses *only* the variance
ordering is competitive.

- **sortnregress**: order genes by marginal variance of log(1+y); regress each gene on all
  predecessors in that order with an L1 penalty (LassoLarsIC, BIC); keep non-zero
  coefficients. Scale-exploiting by construction.
- **randomsort** (control): identical but with a random permutation instead of the
  variance order. Isolates how much of sortnregress's performance is the ordering.

Run both on exactly the stored `.npz` files the other methods use. If sortnregress
approaches RICE on the current data, the benchmark is compromised and A3 becomes the
headline. If RICE clearly beats it even at varsortability 0.98, the reviewer's concern is
answered directly.

### A3. Re-run on data where the ordering carries no information

Two routes; run the first, and the second only if a reviewer presses.

**A3a -- standardize the predictor (fit-time, cheap).** Column-standardize
`M(y) = log(1+y)` before it enters the model, leaving the NB response on raw counts.
Varsortability of the design matrix becomes exactly 0.5. Applies to RICE and to every
baseline that consumes log(1+X), which currently mean-centre but do not variance-scale.

  CAVEAT that must be handled: standardizing rescales column j of W by the standard
  deviation of predictor j, so the fixed edge threshold `w_threshold = 0.3` is no longer
  comparable. Report AUC (already computed, threshold-free) as the primary metric for
  this arm, and either re-select the threshold on a held-out replicate or report the
  full precision-recall curve. Do not compare thresholded MCC across the two arms.

**A3b -- neutralize varsortability in the DGP (data-time, stronger).** Variance
accumulates because each gene's log-mean adds its parents' contributions. Choose the
per-gene NB dispersion `r_j` so that Var(log(1+y_j)) is approximately constant across
genes -- downstream genes get lower dispersion, upstream genes higher. This leaves `W`
and all effect sizes untouched, so recovery is comparable with the existing benchmark;
only the marginal variance profile changes. Target varsortability 0.5 +/- 0.05 and report
the achieved value.

  Alternative if dispersion tuning proves awkward: rescale the incoming weights,
  `W[:,j] -> c_j W[:,j]` with `c_j` chosen to equalize variance. Simpler, but it changes
  effect sizes and therefore detectability, which muddies the comparison.

### A4. What each outcome licenses

| outcome | what you may claim |
|---|---|
| RICE > sortnregress on the current data, and the ranking holds under A3 | the gain is not a scale artifact; state both facts explicitly |
| sortnregress competitive on current data, RICE > sortnregress under A3 | the original benchmark was exploitable; the corrected one is the evidence |
| ranking changes under A3 | report it; the honest claim narrows to the confounded regimes, where R2-sortability is already at chance |

---

## Experiment B -- does the violation mechanism favour RCF by construction?

### The mechanism that matters is WHERE the leak lands

RCF forms the residual for gene i by regressing log(1+x_i) on D_i alone. Therefore:

| violation | reduced CF | full CF |
|---|---|---|
| extra direct effect of D_k on its OWN target x_k | absorbed by construction | absorbed |
| direct effect of D_k on another gene x_j, j != k | NOT absorbed | absorbed (regresses on all D) |
| leak through an unmeasured mediator gene | partially | partially |
| global response to perturbation shared across genes | partially | partially |

So the reviewer is right that the answer is mechanism-dependent, and the fix is to run
mechanisms that differ in exactly this respect and report them SEPARATELY rather than
pooled.

### The four arms

Let D_k be the perturbation indicator for gene k and eta_j the log-mean of gene j.

- **V1 target contamination.** `eta_k += alpha_k D_k` -- an extra direct effect of the
  guide on its own target, beyond the modelled shift v_k. Predicted to favour RCF, since
  the first stage removes it by construction. This is the arm that would confirm the
  reviewer's suspicion, so it must be included.
- **V2 cross-gene direct guide effect.** `eta_j += beta_kj D_k` for a fraction of (k, j)
  pairs with j != k. Predicted to favour the FULL control function, which regresses on all
  indicators. This is the adversarial arm for RCF and the most important one to run.
- **V3 hidden mediator** -- your existing `simulate_NB_hidden_mediator`, unchanged, as the
  reference point so the new arms are anchored to the published result.
- **V4 global perturbation response.** A shared latent factor switched on in every
  perturbed cell, `eta_j += gamma_j * 1{any perturbation}`, representing a transduction or
  stress response. Absorbed by neither first stage.

Sweep violation severity in each arm rather than fixing it at one level: fraction of
affected pairs in {0, 0.02, 0.05, 0.10, 0.20} at the paper's current effect size, so each
arm produces a dose-response curve rather than a point. Note the manuscript currently
pins V3 at 5% with beta in [-3, 3]; keep that as one point on the V3 curve so the new
figure contains the old result.

Methods per arm: RICE with RCF, RICE with full CF, RICE without CF, plus the existing
baselines. d = 10, since a control-function comparison at d = 0 is vacuous.

### What to write

The claim that survives this is narrower and much more defensible than the current one:
RCF is more robust than the full control function under violations that do not act
through the guide's own target, and both degrade gracefully; where the violation is
confined to the target gene, the reduced form absorbs it by construction and the
advantage is expected rather than surprising. If V2 instead shows the full CF ahead, say
so -- a stated boundary condition costs far less than a reviewer finding it.

---

## Cost

Experiment A1 is free (no fitting). A2 is CPU-only and fast: two penalized regressions per
gene per dataset. A3a re-fits every method once on standardized predictors. A3b needs a
generator change plus a full re-run of the affected arms.

Experiment B is the expensive one: 4 mechanisms x 5 severities x 3 CF variants x 20
replicates = 1,200 fits, before baselines. Run V2 and V3 first at three severities; that
is the comparison the reviewer actually asked for, and it is a quarter of the cost.


---

# Where does the FULL control function fail? Four hypotheses, one survivor

All figures below are REDUCED-SCALE DIAGNOSTICS, not results: p=20, s0=60, n as stated,
d=10, ER, soft, single seed, reduced optimizer schedule (lr 1e-3, max_iter 30, epochs
3000), w_threshold 0.3. They locate regimes and orderings; absolute values and crossing
points will move with p. Entries are MCC, and `fcf - rcf` is the quantity of interest.

Context: the cross-gene leak arm (B2) found the FULL control function ahead of the reduced
one at every leak severity -- the opposite of the published hidden-mediator result. These
runs look for the complementary regime.

## H1 -- dense graph: does the full first stage absorb propagated causal signal? NO

Regressing gene j on all D removes the part of x_j explained by D_k, which includes signal
arriving through the causal path k -> j. That should worsen as the graph densifies.
It does not (cells per perturbation fixed at 135, no violation):

| in-degree | rcf | fcf | fcf - rcf |
|---|---|---|---|
| 2 | 0.788 | 0.835 | +0.047 |
| 4 | 0.749 | 0.821 | +0.072 |
| 8 | 0.659 | 0.707 | +0.048 |

Flat. Discard.

## H2 -- diffuse weak leak at matched severity: NO consistent effect

Per-gene leak SD held constant (see `effect_for_total_sd` in sim_crossguide.py)
while the per-pair effect scales as 1/sqrt(density), so concentration moves and severity
does not. n=3000, n/p=150:

| density | per-pair effect | leaks/gene | rcf | fcf | fcf - rcf |
|---|---|---|---|---|---|
| 0.05 | 3.00 | 0.95 | 0.616 | 0.700 | +0.084 |
| 0.15 | 1.73 | 2.85 | 0.624 | 0.730 | +0.106 |
| 0.35 | 1.13 | 6.65 | 0.603 | 0.735 | +0.132 |
| 0.75 | 0.78 | 14.25 | 0.583 | 0.659 | +0.076 |

The advantage GREW through density 0.35. Crossed with sample size, concentration still
does not drive the sign -- n/p does:

| n/p | density 0.05 | density 0.75 |
|---|---|---|
| 150 | +0.084 | +0.076 |
| 40 | +0.061 | +0.123 |
| 20 | **-0.052** | **-0.012** |

## H3 -- high MOI: collinear indicators break the full first stage? NO, premise inverted

| n/p | MOI | mean abs off-diag corr | cond(first stage) | rcf | fcf | nocf |
|---|---|---|---|---|---|---|
| 150 | 1 | 0.047 | 241 | 0.744 | 0.785 | 0.689 |
| 150 | 3 | 0.035 | 47 | 0.840 | 0.905 | 0.752 |
| 150 | 6 | 0.016 | 40 | 0.931 | 0.952 | 0.793 |
| 150 | 10 | 0.043 | 65 | 0.931 | 1.000 | 0.835 |
| 40 | 1 | 0.047 | 241 | 0.630 | 0.605 | 0.629 |
| 40 | 3 | 0.041 | 47 | 0.708 | 0.810 | 0.735 |
| 40 | 6 | 0.028 | 40 | 0.794 | 0.807 | 0.729 |
| 40 | 10 | 0.047 | 65 | 0.845 | 0.875 | 0.804 |

`simulate_NB_highMOI` assigns MOI guides per cell uniformly without replacement, so the
indicator columns are near-independent Bernoulli and MOI does NOT induce collinearity.
The ill-conditioned case is MOI = 1, where the one-hot rows sum to one in every perturbed
cell and are nearly collinear with the intercept -- condition number 241 at MOI 1 versus
40-65 above it. Higher MOI also adds signal, so every arm improves. Do not propose MOI as
a stress test for the full control function in this DGP.

## H4 -- small n/p: SURVIVES

The full first stage fits p coefficients per gene where the reduced one fits a single
coefficient, so its variance cost scales with p/n while the bias it removes does not.
Below roughly 20 cells per first-stage parameter the cost exceeds the benefit:

| n | n/p | cells/perturbation | rcf | fcf | fcf - rcf |
|---|---|---|---|---|---|
| 3000 | 150 | 135 | 0.747 | 0.810 | +0.063 |
| 1000 | 50 | 45 | 0.689 | 0.725 | +0.036 |
| 400 | 20 | 18 | 0.578 | 0.563 | **-0.015** |

Monotone, and it reproduces under the leak conditions of H2 as well.

### Proposed scenario

Sweep n at fixed p=100: **2,000 / 5,000 / 10,000 / 20,000 / 50,000**, i.e. n/p = 20, 50,
100, 200, 500. Arms rcf / fcf / nocf, d=10, NO exclusion violation, so the contrast is
purely first-stage variance. This spans the measured crossing at ~20, the paper's current
benchmark at 100, and the Replogle regime at ~200.

Why this is the one to run: a focused screen of 2,000 cells over 100 genes sits at the
crossing, so the regime is realistic rather than contrived; it is the complement to B2
(cross-gene leaks favour the full form, thin data favour the reduced one), and the two
together define WHEN each is preferable instead of asserting a winner; and it places the
published settings on the curve rather than off it, which forecloses the objection that
the reported regime was chosen to suit the method.

Needs no new generator -- `simulate_NB_tgt` at varying n -- so it is the cheapest of the
scenarios in this document to put on the cluster.


## H5 -- first-stage misspecification ("error aggregation"): NO

If E[x_k | z, D] is not the additive linear function the first stage assumes, every
included indicator carries a misspecification error into the residual -- p of them for the
full form, one for the reduced. Tested with the two generators already present in
sim_base.py and called from no driver, at n/p = 150 where the full form
otherwise wins, so a reversal could only come from misspecification:

| mechanism | strength | rcf | fcf | fcf - rcf |
|---|---|---|---|---|
| `simulate_NB_perturbation_interaction` (shift v_j + gamma_j z_um, z_um unmeasured) | 0 | 0.666 | 0.690 | +0.024 |
| | 2 | 0.555 | 0.552 | -0.003 |
| | 5 | 0.602 | 0.619 | +0.017 |
| `simulate_NB_mixture` (3 latent states modulating perturbation response) | 0 | 0.602 | 0.595 | -0.007 |
| | 1 | 0.638 | 0.666 | +0.028 |
| | 3 | 0.631 | 0.666 | +0.035 |

No trend; the gap wanders within +/-0.035, and under latent states the full form pulls
slightly AHEAD as misspecification strengthens. Both mechanisms misspecify each gene's
response to its OWN perturbation -- the single coefficient the reduced form relies on --
while the full form's extra columns act as weak proxies for the hidden modulator and
absorb part of it. Aggregation across p errors does not dominate that.

Note these two generators call the MODULE-LEVEL `_simulate_single_equation` (cap 7.5,
floor -0.5) rather than `simulate_NB_tgt`'s inner version (6.5, 0.5), so their
strength = 0 rows are not on the same count scale as the NB baseline and should not be
compared across tables.

## Scoreboard

| hypothesis | full CF worse than reduced? |
|---|---|
| H1 dense graph | no |
| H2 diffuse weak leak at matched severity | no |
| H3 high MOI | no -- premise inverted |
| H4 small n/p | **yes, sign flips at n/p ~ 20** |
| H5 first-stage misspecification | no |

Five mechanisms tested, one survivor. Build the additional simulation on H4.


---

# CONTROL: is the cross-gene result a mechanism effect or a scale artifact?

The cross-gene leak (B2) put the FULL control function ahead, the opposite of the
published hidden-mediator result. Before attributing that to the mechanism, the published
mechanism was re-run at the diagnostic scale -- same seed, same settings, same optimizer
schedule, `simulate_NB_hidden_mediator` with the manuscript's parameterization
(`mediator_in_effect = mediator_out_effect = 3.0`):

| mediator density | rcf | fcf | nocf | fcf - rcf |
|---|---|---|---|---|
| 0.00 | 0.652 | 0.642 | 0.563 | -0.010 |
| 0.05 | 0.511 | 0.445 | 0.506 | **-0.066** |
| 0.10 | 0.582 | 0.529 | 0.533 | **-0.053** |

It reproduces the manuscript's ordering: the two control-function arms are tied at zero
violation, and the reduced form pulls ahead only once the mediator is switched on.

Side by side, on matched configurations:

| violation | fcf - rcf |
|---|---|
| none | -0.010 |
| hidden mediator, 5% | -0.066 |
| hidden mediator, 10% | -0.053 |
| cross-gene guide leak, 5% | +0.084 |
| cross-gene guide leak, 35% | +0.132 |

**The discrepancy is the mechanism, not the scale.** The gaps differ by 0.15 to 0.20 in
opposite directions on configurations that differ only in how the violation is carried.

Mechanistic account, consistent with both halves: a mediator-borne leak reaches gene j
through an unmeasured node, so no first stage can absorb it -- but the full form fits p
coefficients' worth of noise attempting to, which is why it does worse. A direct
D_k -> x_j leak IS in the span of the full first stage, so that form removes it while the
reduced form, seeing only D_j, cannot.

This control also limits how much the reduced-scale caveat matters for the B2 conclusion:
the comparison is paired across matched configurations rather than an extrapolation of
absolute magnitudes, and it independently recovers a published result.


---

# H6, H7 and a resolution limit

## H6 -- leaks aligned with TRUE EDGES: NO (opposite of the prediction)

Prediction was that a leak D_k -> x_j on a true edge k -> j would make the full first
stage absorb the instrument identifying that edge. Matched leak count (30), p=20, n=3000:

| leak lands on | rcf | fcf | fcf - rcf | fcf precision |
|---|---|---|---|---|
| nothing | 0.790 | 0.916 | +0.126 | 0.892 |
| non-edges | 0.707 | 0.778 | +0.071 | 0.724 |
| true edges | 0.683 | 0.845 | +0.162 | 0.855 |

The full form wins MORE on edge-aligned leaks. Absorbing D_k out of x_j removes the
spurious direct component while the genuine k -> j edge stays identified through x_k,
which remains a predictor in the second stage; the reduced form leaves the spurious part
in and inflates that edge (precision 0.649 vs 0.855).

## H7 -- stochastic fraction of the leak (author's decomposition)

A hidden mediator M_k = D_k + eps_k feeding its recipients equals a direct off-target
effect plus noise: the deterministic part is in the span of the full first stage and is
absorbed, eps_k is not and acts as a correlated-loading confounder. `leak_noise_sd` in
sim_crossguide.py sweeps that fraction at fixed deterministic severity.

| concentration | sigma | realised leak SD | rcf | fcf | nocf | fcf - rcf |
|---|---|---|---|---|---|---|
| few strong (beta 2.12, 1.9 sources/gene) | 0 | 0.30 | 0.614 | 0.613 | 0.729 | -0.001 |
| | 1 | 1.49 | 0.694 | 0.718 | 0.605 | +0.024 |
| | 2 | 2.94 | 0.325 | 0.335 | 0.319 | +0.010 |
| many weak (beta 0.78, 14.25 sources/gene) | 0 | 0.33 | 0.583 | 0.659 | 0.657 | +0.076 |
| | 1 | 1.63 | 0.677 | 0.654 | 0.725 | -0.023 |
| | 2 | 3.21 | 0.584 | 0.605 | 0.403 | +0.021 |

At sigma = 2 the leak SD reaches 3.2 and every arm collapses (nocf 0.403), so that column
is destruction rather than a regime. No consistent effect of sigma or of concentration.

## THE RESOLUTION LIMIT -- read this before trusting any single-seed table above

All sigma = 0 points lie on ONE matched-severity curve: same anchor SD, same seed, only
concentration moving. In order of density:

| density | 0.05 | 0.10 | 0.15 | 0.35 | 0.75 |
|---|---|---|---|---|---|
| fcf - rcf | +0.084 | **-0.001** | +0.106 | +0.132 | +0.076 |

The 0.10 point should lie between +0.084 and +0.106; it reads -0.001. That is a
single-seed swing of ~0.09 on a smooth curve, and it is the same size as nearly every
effect screened in this document. The correct reading of H1, H2, H3, H5, H6, H7 and the
control-pool sweep is therefore "effect, if any, is BELOW NOISE" -- not "mechanism
excluded".

Two findings exceed that threshold and can be quoted:

1. **Hidden mediator vs cross-gene leak**: -0.066 against +0.084 on matched
   configurations, a separation of 0.15, with the mediator arm independently reproducing
   a published result.
2. **n/p**: 0.078 monotone across three points, reproducing under both leak
   concentrations.

Next step is replication, not more mechanisms: 20 replicates of the mediator-versus-direct
contrast (sweeping `leak_noise_sd`) on the GPU queue would settle whether the stochastic
fraction matters, and is one array job.


---

# THE REGIME FINDING -- read before using any p=20 number above

Ten paired replicates at p=20 gave, for `fcf - rcf` (MCC, mean +/- SE):

| configuration | rcf | fcf | fcf - rcf | rcf ahead |
|---|---|---|---|---|
| **no violation** | 0.708 | 0.765 | **+0.057 +/- 0.013** | **0/10** |
| hidden mediator | 0.530 | 0.514 | -0.017 +/- 0.024 | 7/10 |
| direct leak, sigma=0 | 0.565 | 0.580 | +0.015 +/- 0.023 | 3/10 |
| direct leak, sigma=0.5 | 0.646 | 0.630 | -0.016 +/- 0.011 | 7/10 |
| direct leak, sigma=1 | 0.592 | 0.587 | -0.005 +/- 0.020 | 7/10 |

Two consequences. The single-seed mediator result quoted earlier (-0.066) does NOT
replicate: over ten replicates it is -0.017 +/- 0.024. And with NO violation at all the
full control function leads by +0.057 +/- 0.013 in 10/10 replicates -- so every mechanism
screened in this document was measured on top of a baseline already biased toward the full
form.

## The baseline offset is a function of p, not of the penalty

No violation, n/p = 150 and s0/p = 3 held fixed, 5 replicates each:

| p | n | pen1 | rcf | fcf | fcf - rcf | rcf ahead |
|---|---|---|---|---|---|---|
| 20 | 3,000 | 1e-4 | 0.717 | 0.773 | +0.056 +/- 0.023 | 0/5 |
| 20 | 3,000 | 1e-3 | 0.727 | 0.782 | +0.055 +/- 0.022 | 0/5 |
| 40 | 6,000 | 1e-4 | 0.820 | 0.825 | +0.005 +/- 0.009 | 3/5 |
| 60 | 9,000 | 1e-4 | 0.889 | 0.886 | -0.003 +/- 0.004 | 3/5 |

A partial rerun at p=40 with 8 replicates gives +0.0105 +/- 0.0065, consistent.

pen1 is NOT the explanation (+0.055 vs +0.056). p IS: the offset collapses monotonically
and is gone by p=40-60. Extrapolated to p=100 the clean-case baseline is neutral or mildly
favours the reduced form, which is consistent with the manuscript's Figure 2 rather than
in conflict with it.

**Therefore: every hypothesis screened at p=20 in this document was tested in the wrong
regime.** H1-H7 should be read as untested at the manuscript's scale, not as refuted. The
cluster pipeline (generate_cf_comparison.py / fit_cf_comparison.py) re-runs the four
configurations at p=100 with 20 paired replicates and the no-violation reference included,
which is the experiment that can actually answer the reviewer.
