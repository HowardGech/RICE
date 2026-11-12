# Sensitivity setting: violation of the independence assumption

The assumption under test:

> After conditioning on the experimental design and the observed cell-level covariates z,
> guide assignment or effective uptake is independent of the unmeasured structural errors
> affecting the target genes.

No existing arm violates it. In every generator the perturbed set is drawn with
`np.random.choice(n, size=int(n * intervene_prob), replace=False)`, with no reference to the
latent factors, so the instruments stay valid however large the confounding grows. The other
sweeps break exclusion (off-target activity, hidden mediator, direct contamination) or
relevance (efficacy heterogeneity), and cell-level mismatch breaks neither -- the corruption
is applied at random, so the recorded indicator stays independent of the structural errors
and what fails is instrument *measurement*, not exogeneity.

## The mechanism

Take `u_i = z_{i1}`, the first latent factor, standardised. Draw the cells that receive any
perturbation with probability proportional to `exp(a * u_i)` instead of uniformly. Gene
assignment *within* the perturbed set is left exactly as it is.

This is the reviewer's own mechanism: a perturbation registers preferentially in cells in
particular latent states, so the control cells are no longer exchangeable with the perturbed
ones. Because `u` carries non-zero loadings on every gene, it enters the target equations,
which is what makes the dependence a violation rather than a nuisance.

### Why the perturbed SET and not the gene

The dependence could attach at either of two points, and the perturbation rate decides how
severe a violation each can express. For a binary indicator with rate q and a standard normal
u, the attainable correlation is bounded by phi(z_q)/sqrt(q(1-q)):

| attachment point | q | ceiling on \|corr(D, u)\| |
|---|---|---|
| which gene, among perturbed cells | 0.0090 | 0.257 |
| whether a cell is perturbed at all | 0.90 | 0.585 |

The per-gene route is capped at 0.257 by the 90 cells per gene, less than half the reach of
the other. It also interacts badly with the existing code: genes draw sequentially from a
shrinking pool (`remaining = np.setdiff1d(remaining, choice)`), so under weighted sampling
the first gene takes the most extreme cells and the last takes what is left, making the
realised violation depend on gene index. Attaching at the set level is gene-symmetric and
leaves that loop untouched.

### Severity, measured rather than nominal

Report the realised correlation, not `a`. Measured at n = 10,000, 90% perturbed:

| a | corr(perturbed, u) | mean u, control | mean u, perturbed |
|---|---|---|---|
| 0.00 | 0.006 | -0.01 | 0.01 |
| 0.25 | 0.180 | -0.53 | 0.07 |
| 0.50 | 0.321 | -0.96 | 0.11 |
| 1.00 | 0.439 | -1.31 | 0.15 |
| 2.00 | 0.524 | -1.56 | 0.18 |

Sweep `a` in {0, 0.25, 0.5, 1, 2}. Beyond 2 the curve saturates against the 0.585 ceiling, so
nothing is gained. Note where the movement is: the perturbed group barely shifts, because it
is 90% of the cells, while the control group is driven to -1.56 standard deviations. The
violation is a *control-group* distortion, which is the right reading for a pooled screen --
the comparison baseline degrades, not the treated population.

## Implementation

One branch in the assignment block, taking the original path at a = 0 so the zero point
reproduces the published generator bit-for-bit and consumes the same random stream:

```python
if a_select:
    uu = Z[:, 1] if intercept else Z[:, 0]          # first latent factor
    uu = (uu - uu.mean()) / uu.std()
    wsel = np.exp(a_select * uu)
    intervene_idx = np.random.choice(n, size=int(n * intervene_prob),
                                     replace=False, p=wsel / wsel.sum())
else:
    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
intervene_idx.sort()
```

Everything downstream is unchanged, so each gene still receives exactly
`int(n * intervene_prob) // p` cells and the unperturbed pool keeps its full size (verified
at 1,000 cells for every a, so `prep()`'s requirement of at least one fully unperturbed cell
is never at risk).

## Diagnostics to print per setting

  corr_perturbed_u        realised corr(perturbed indicator, u) -- the severity axis
  mean_u_control          the quantity actually being distorted; goes to -1.56 at a = 2
  mean_u_perturbed        moves little, by construction
  corr_D_u_per_gene       mean and spread over genes, inherited from the set-level selection
  n_control_cells         must stay > 0
  cells_per_gene_min      unchanged by construction; assert it

## How to read the result, and what it cannot show

Every method should degrade, since the estimand itself is no longer identified -- this arm
is not a contest but a measurement of how fast the bias grows. The specific question for RICE
is whether the control function absorbs part of it: the residual of gene j on D_j is a proxy
for the latent state, and under selection that proxy is itself correlated with D, so the
second stage inherits the contamination rather than removing it. Expect the reduced and full
control functions to separate here differently from the exclusion-violation arms.

Two things this setting does not cover, both worth stating rather than leaving implicit:

  * Gene-specific selection, where different guides are taken up preferentially in different
    states. Here one shared direction drives all genes, with the loadings giving the induced
    bias mixed signs across genes. A gene-specific variant would use z'u_j per gene and is a
    one-line change, but it reintroduces the sequential-depletion problem.
  * Dependence on *observed* covariates. That is not a violation at all: the assumption is
    conditional on z, and the model already conditions on the observed covariates with
    gene-specific coefficients. Worth making this point in the response before the empirical
    one, since the comment as written ("uptake can depend on cell-level covariates") is
    answered by the conditioning for any covariate that is measured.
