# KD-efficacy extension — local validation and diagnostics

All numbers below were produced on CPU with `scRICE_eff(backend='numpy')`, the
same implementation that runs on GPU with `backend='cupy'`. These are
DIAGNOSTICS at reduced scale (single seed unless stated), not the experiment:
they exist to decide the model specification before GPU time is spent.

Common reduced settings: n=3000, p=20, s0=60, ER, NB, d=0 unless stated,
`pen1=1e-3, pen2=0, pen1_c=0, pen2_c=1e-2`, `regularizer='expm'`, `lr=1e-3`,
`max_iter=30`, `epochs=3000`, `w_threshold=0.3`, `eta_init=0` (xi-hat starts at 0.5).

## 1. Correctness

- **Generator.** `simulate_NB_efficacy` reproduces `simulate_NB_tgt` bit-for-bit
  at xi=1 (soft) and xi=0 (hard), with `Z, Um, Ur, Vm, Vr` and the intervention
  assignment identical — it consumes no extra random numbers.
- **Gradients.** Every parameter block finite-differenced against
  `_score_value` (central differences, h=1e-6), for both `xi_scope` settings:

| block | W | Ur_g | Um_g | Vr | Vm | W_c | Eta |
|---|---|---|---|---|---|---|---|
| max rel. err, scope='both' | 2e-10 | 1.5e-9 | 4e-10 | 9.2e-9 | 7.2e-10 | 3.4e-9 | 2.1e-8 |
| max rel. err, scope='parent' | 6.7e-11 | 6.9e-9 | 1.7e-10 | 3.5e-8 | 6.8e-10 | 3.4e-9 | 2.1e-8 |

- The `hard` and `soft` arms are provably unaffected by the new option: when
  `intervention_type != 'efficacy'`, `prep` forces `xi_scope='both'` and
  `cf_mask = mask`, i.e. the published fixed-mask behaviour.

## 2. End-to-end recovery, xi_scope='both' (the strict Eq. 3 generalization)

Three centers, all three arms fit on identical data:

| xi center | true mean xi | xi-hat mean | xi RMSE | MCC hard | MCC soft | MCC adaptive |
|---|---|---|---|---|---|---|
| 0.0 | 0.035 | 0.533 | 0.618 | 0.860 | 0.849 | 0.860 |
| 0.5 | 0.517 | 0.773 | 0.371 | 0.889 | 0.889 | 0.889 |
| 1.0 | 0.981 | 0.991 | 0.041 | 0.900 | 0.869 | 0.869 |

Two problems. xi-hat barely leaves its 0.5 initialization except where the truth
is already near 1, and graph recovery hardly separates the three arms — at xi=1
the *hard* fit even scored highest, the opposite of what misspecification predicts.

## 3. It is not an optimizer problem

Raising `lr_eta_multiplier` 20x and 100x moves xi-hat in the fourth decimal:

| center | eta multiplier | xi-hat | RMSE (all genes) |
|---|---|---|---|
| 0.0 | 1 | 0.5327 | 0.6176 |
| 0.0 | 20 | 0.5328 | 0.6178 |
| 0.0 | 100 | 0.5329 | 0.6178 |
| 0.5 | 1 | 0.7730 | 0.3710 |
| 0.5 | 20 | 0.7731 | 0.3713 |
| 0.5 | 100 | 0.7733 | 0.3713 |

So Adam is sitting at a genuine stationary point. Nor is it finite-sample:
quadrupling cells per perturbed gene (135 -> 540, via n=12000) made the bias
*worse*, xi-hat 0.773 -> 0.827 at the same truth of 0.517.

## 4. The cause is the control-function term, not its penalty

Center 0.5, truth 0.517:

| configuration | xi-hat | bias | MCC |
|---|---|---|---|
| baseline | 0.7731 | +0.256 | 0.890 |
| ridge on W_c removed (`pen2_c=0`) | 0.7737 | +0.257 | 0.858 |
| L1 on W removed (`pen1=0`) | 0.7723 | +0.256 | 0.870 |
| CF term removed entirely | 0.5500 | +0.033 | 1.000 |

Neither penalty matters. Removing the CF term collapses the bias. (MCC hits 1.0
there only because these runs use d=0, so the control function has no
confounding to correct and acts as pure nuisance.)

Mechanism: with `xi_scope='both'`, xi_j scales the CF residual channel as well
as the parent channel, and the two can be traded against each other. xi is
identified only by the perturbed-vs-unperturbed contrast in parent signal, so
adding a second xi-scaled channel that carries no such contrast pulls the
estimate upward.

## 5. Parent-only scaling removes the bias at no cost in graph recovery

Center 0.5, truth 0.517, with and without genuine unmeasured confounding:

| d | xi_scope | xi-hat | bias | RMSE (genes with parents) | Spearman | MCC |
|---|---|---|---|---|---|---|
| 0 | both | 0.773 | +0.256 | 0.349 | 0.226 | 0.890 |
| 0 | parent | 0.625 | +0.108 | 0.318 | 0.211 | 0.890 |
| 3 | both | 0.723 | +0.207 | 0.307 | 0.393 | 0.787 |
| 3 | parent | 0.498 | **-0.019** | 0.313 | 0.173 | 0.791 |

With d=3 — the realistic case, where the control function is actually doing work
— parent-only scaling is essentially unbiased in the mean (-0.019 vs +0.207),
and MCC is unchanged (0.791 vs 0.787).

There is also an interpretive argument for it: with `xi_scope='parent'` the
xi -> 1 limit still equals the soft model exactly, while the xi -> 0 limit
differs from the hard model in leaving the confounder correction active on
perturbed cells. Since a gene being knocked down is no reason for its
confounding to vanish, that is arguably the better model, not merely the better
behaved one.

## 6. What these diagnostics do NOT establish

- **Per-gene xi is not recovered at this scale.** RMSE on genes with parents sits
  near 0.31 in every configuration and Spearman is 0.17-0.39. The mean xi is
  estimable; ranking individual genes by efficacy is not, here.
- **Cells per perturbation is the binding constraint, and production is worse.**
  `simulate_NB_efficacy` gives each gene exactly `int(n*0.9)//p` perturbed cells:
  135 in these diagnostics, but only **90** at the paper's base scale
  (n=10,000, p=100) and 180 at the large scale (n=100,000, p=500). xi_j is
  informed only by those cells.
- Single seed, single center for sections 3-5; p=20 rather than 100. Replication
  across seeds is part of the production run, not of this note.
- No GPU path has been executed. `scRICE_eff` has only ever run under NumPy here.
