#!/usr/bin/env python
"""
Control-function comparison, generation through fitting, in one file.

    python compare_control_functions.py --stage both                      # one replicate, all configs
    python compare_control_functions.py --stage gen  --replicate 7        # generation only
    python compare_control_functions.py --stage fit  --replicate 7        # fitting only
    LSB_JOBINDEX=7 python compare_control_functions.py --stage both       # replicate from the array index

Compares three control-function variants under four exclusion-restriction violation
regimes:

    arms     rcf   reduced control function -- gene i regressed on its OWN indicator D_i
             fcf   full control function    -- gene i regressed on ALL indicators
             nocf  no control function

    configs  none         no violation -- the reference the others are measured against
             direct_s0    direct cross-gene guide leak, deterministic carrier D_k
             direct_s05   the same leak with carrier D_k + 0.5 * eta_k, eta shared
                          across all recipients of source k
             mediator     the published hidden-mediator violation

All configurations in a replicate are built from ONE DAG and the same data-RNG start, so
the arms are PAIRED within replicate. Analyse them as within-replicate differences
against the `none` reference, not as independent group means.

WHAT IS CORRECTED HERE relative to the earlier two-script pipeline
------------------------------------------------------------------
1. The count-rescale overflow. Every NB generator rescaled the log-mean with two
   sequential shifts, and the second could undo the first:

       if np.max(m) > CAP:    m -= (np.max(m) - CAP)
       if np.mean(m) < FLOOR: m += (FLOOR - np.mean(m))

   With one extreme cell the cap subtracts a large constant, dragging the column mean
   far below FLOOR, and the floor then adds that whole distance back. 10,000 cells at
   m = -3 with a single cell at m = 40, CAP = 6.5, FLOOR = 0.5:

       start                  ->  max 40.00, mean  -3.00
       step 1 subtracts 33.50 ->  max  6.50, mean -36.50
       step 2 adds     37.00  ->  max 43.50, mean   0.50
       exp(43.50) = 7.76e18   ->  NB mean beyond int64 max (9.22e18)

   `nbinom.rvs` then saturates on some builds and WRAPS NEGATIVE on others; a negative
   count makes `np.log(1 + X)` NaN and the first-stage regression dies with "Input y
   contains NaN". It also cascades, since a gene with astronomic counts enters its
   children through g(X) = log(1+X) ~ 40. At p = 100 with no violation at all, 27-36 of
   100 genes had maximum counts above 1e6. Every generator below carries the fix --
   `m = np.minimum(m, CAP)` after both shifts -- which is a no-op unless the pathology
   triggers.

2. The generators are EMBEDDED here rather than imported from sim_base.py, so
   the four arms cannot silently differ depending on whether that file was patched.

3. `validate_counts` runs before each save and raises on non-finite, negative, or
   implausible counts, naming the offending genes -- at generation, not hours later in
   the fit stage.

KNOWN ASYMMETRY BETWEEN ARMS, unchanged from the published code
---------------------------------------------------------------
`simulate_NB_hidden_mediator` has no equation of its own and calls the module-level one,
whose rescale constants are cap 7.5 / floor -0.5. `simulate_NB_tgt` and
`simulate_NB_crossguide` each carry an inner copy using 6.5 / 0.5. The mediator arm
therefore generates counts on a slightly different scale from the `none` and direct-leak
arms -- a confound for any cross-arm comparison of absolute metric levels, though the
within-replicate paired difference inside each configuration is unaffected. MODULE_CAP and
MODULE_FLOOR are exposed below so this can be aligned deliberately; the defaults reproduce
the published behaviour.

NOT reproducible against unpatched output: because the pathology triggers on most
p = 100 replicates, these numbers will not match a pre-fix run gene-for-gene.
Regenerate whole arms; a figure mixing patched and unpatched replicates is invalid.

Writes one tidy CSV per replicate, schema (config, method, measure, value), which
plot_control_function_violation.py consumes unchanged.
"""
from sim_base import simulate_NB_tgt, _simulate_single_equation, is_dag
import argparse
import os
import sys
import time

import numpy as np
import igraph as ig
import pandas as pd
from scipy.stats import nbinom

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import simulate_dag, simulate_parameter, set_random_seed, measurement, to_dag

# Rescale constants for the MODULE-LEVEL equation, which is what
# simulate_NB_hidden_mediator uses. These are the published values, and they differ from
# the 6.5 / 0.5 used inside simulate_NB_tgt and simulate_NB_crossguide -- see the note on
# arm comparability in the module docstring. Set them to 6.5 / 0.5 to put every arm on one
# count scale, at the cost of no longer reproducing the published mediator arm.
MODULE_CAP, MODULE_FLOOR = 7.5, -0.5

# ======================================================================================
# Generators -- verbatim from sim_base.py and sim_crossguide.py with the
# rescale fix applied. Do not edit in place; re-derive from those files if they change.
# ======================================================================================


def simulate_NB_hidden_mediator(W, theta_m, theta_r, n,
                                  intercept=True, g=lambda x: x,
                                  d=0, shift=1.0, coef=1.0, covar=1.0,
                                  intervene_prob=0.9,
                                  hard_intervention=True,
                                  mediator_density=0.05,
                                  mediator_in_effect=3.0,
                                  mediator_out_effect=3.0):
    """
    Simulate Perturb-seq data with hidden mediators that bypass the
    observed regulatory cascade.

    For a fraction of (source, target) gene pairs (i, k), the perturbation
    D_i affects x_k through both the observed regulatory path (i -> ... -> k
    in W) AND through an unobserved mediator h that is not part of the
    visible network. The hidden mediator h also affects other observed
    genes downstream of k, creating a perturbation-mediated correlation
    that the full CF's predicted expression x_k_hat captures but x_i_hat
    does not.

    This is the setting under which the full control function reliably
    produces false edges, because the second-stage regression cannot
    screen off the hidden-mediator signal via observed parent expression.

    Parameters
    ----------
    mediator_density : float in [0, 1]
        Fraction of (source perturbation, target gene) pairs that have
        a hidden-mediator bypass.
    mediator_in_effect : float
        Strength of D_source -> h coupling.
    mediator_out_effect : float
        Strength of h -> x_target coupling.
    """
    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    X = np.zeros([n, p])

    Um = np.random.uniform(-coef, coef, size=(d, p))
    Ur = np.random.uniform(-coef, coef, size=(d, p))
    if intercept:
        Um = np.vstack((theta_m, Um))
        Ur = np.vstack((theta_r, Ur))
    Z = np.random.normal(0, covar, size=(n, d))
    if intercept:
        Z = np.hstack((np.ones((n, 1)), Z))

    ones = np.ones((n, 1))
    Vm = np.random.uniform(-1.5 * shift, -shift, size=(1, p))
    Vr = np.random.uniform(-shift / 2, shift / 2, size=(1, p))

    covar_m = Z @ Um
    covar_r = Z @ Ur
    shift_m = ones @ Vm
    shift_r = ones @ Vr

    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    intervene_mask = np.zeros((n, p), dtype=bool)
    remaining_nodes = intervene_idx
    each_intervention_size = len(intervene_idx) // p
    for i in range(p):
        choice = np.random.choice(remaining_nodes, size=each_intervention_size, replace=False)
        intervene_mask[choice, i] = True
        remaining_nodes = np.setdiff1d(remaining_nodes, choice)

    D = intervene_mask.astype(float)  # (n, p)

    # --- Hidden mediator structure ---
    # For each pair (source, target), decide whether a hidden mediator
    # exists. We use a sparse mask H_struct[source, target] = 1 if the
    # pair has a hidden bypass.
    H_struct = (np.random.rand(p, p) < mediator_density).astype(float)
    np.fill_diagonal(H_struct, 0)  # no self-mediation

    # For each (source, target) pair with a mediator, sample the
    # mediator's effect on the target (mediator_out) and on other
    # downstream genes. Crucially, the mediator's downstream effect must
    # touch OBSERVED GENES OTHER THAN the source's observed regulatory
    # descendants, so that the false edge has somewhere to land.
    # Simplest version: for each (source, target) with a mediator,
    # sample h ~ N(coupling * D_source, 1), and add h's effect to a
    # randomly chosen "leak gene" downstream.

    # Implementation: we'll directly inject the hidden-mediator effect
    # into a small number of "leak genes" per active source. For each
    # source gene i with a mediator-bypass row, we choose a leak gene k
    # (other than i and its visible descendants) and add an effect
    # proportional to D_i to x_k.

    # Build leak structure: for each source, list of leak target genes
    # that receive hidden-mediator effects.
    leak_targets = {}
    in_coeffs = {}
    out_coeffs = {}
    for source in range(p):
        targets = np.where(H_struct[source] > 0)[0]
        if len(targets) == 0:
            continue
        # Avoid the visible descendants of `source` to ensure the false
        # edge actually corresponds to an unobserved path.
        descendants = G.subcomponent(source, mode=ig.OUT)
        targets = [t for t in targets if t not in descendants]
        if len(targets) == 0:
            continue
        leak_targets[source] = targets
        in_coeffs[source] = np.random.uniform(-mediator_in_effect,
                                               mediator_in_effect)
        out_coeffs[source] = np.random.uniform(-mediator_out_effect,
                                                mediator_out_effect,
                                                size=len(targets))

    # Now generate the hidden mediator values per cell. We allow one
    # hidden mediator per source (parameter-light); h_source(cell) is a
    # noisy linear function of D[source, cell].
    n_sources_with_mediator = len(leak_targets)
    if n_sources_with_mediator > 0:
        source_keys = list(leak_targets.keys())
        H_values = np.zeros((n, len(source_keys)))
        for idx, source in enumerate(source_keys):
            H_values[:, idx] = in_coeffs[source] * D[:, source] + \
                                np.random.normal(0, 0.3, size=n)

        # Compute the hidden-mediator contribution to each observed gene
        hidden_contribution = np.zeros((n, p))
        for idx, source in enumerate(source_keys):
            for t_idx, target in enumerate(leak_targets[source]):
                hidden_contribution[:, target] += \
                    out_coeffs[source][t_idx] * H_values[:, idx]
    else:
        hidden_contribution = np.zeros((n, p))

    hidden_contribution = np.clip(hidden_contribution, -5, 5)
    shift_m_masked = shift_m * intervene_mask
    shift_r_masked = shift_r * intervene_mask

    for j_idx in ordered_vertices:
        parents = G.neighbors(j_idx, mode=ig.IN)
        cov_m_j = covar_m[:, j_idx] + hidden_contribution[:, j_idx]
        cov_r_j = covar_r[:, j_idx]
        mask_j = intervene_mask[:, j_idx] if hard_intervention else np.zeros(n)
        X[:, j_idx] = _simulate_single_equation(
            g(X[:, parents]), W[parents, j_idx],
            cov_m_j, cov_r_j,
            mask_j,
            shift_m_masked[:, j_idx], shift_r_masked[:, j_idx],
        ).reshape(-1)

    return X, intervene_mask, Z, Um, Ur, Vm, Vr, leak_targets

def effect_for_total_sd(total_sd, leak_density, p, q):
    """
    Per-pair effect size a such that beta ~ U(-a, a) gives each gene a leak term of
    standard deviation `total_sd`, whatever the density.

    Var(leak_j) = m * E[beta^2] * Var(D) = m * (a^2 / 3) * q(1 - q),
    with m = leak_density * (p - 1) leaking guides per gene and q the probability that a
    given cell carries a given guide. Holding total_sd fixed while sweeping density
    separates CONCENTRATION (few strong leaks vs many weak ones) from SEVERITY, which is
    what makes the comparison of control-function variants interpretable: otherwise a
    denser leak is also a bigger one and the two effects cannot be told apart.
    """
    m = leak_density * (p - 1)
    if m <= 0 or q <= 0 or q >= 1:
        return 0.0
    return float(total_sd * np.sqrt(3.0 / (m * q * (1.0 - q))))

def draw_leak_matrix(p, leak_density, leak_effect, rng, W=None):
    """
    Beta[k, j] = direct effect of guide k on gene j, zero diagonal.

    leak_density is the fraction of the p*(p-1) ordered off-diagonal pairs that leak.
    Pairs are drawn uniformly; pairs that coincide with a true edge k -> j are allowed
    (they mimic a real edge) and counted separately, since the more damaging case is a
    leak where no edge exists.
    """
    Beta = np.zeros((p, p))
    off = [(k, j) for k in range(p) for j in range(p) if k != j]
    n_leak = int(round(leak_density * len(off)))
    info = dict(leak_density=float(leak_density), leak_effect=float(leak_effect),
                n_pairs_possible=len(off), n_leak=n_leak,
                # severity is more interpretably indexed by leak SOURCES PER GENE than by
                # density, because density * (p - 1) sources land on each gene: at p = 100
                # a density of 0.05 gives every gene five independent leaking guides, and
                # the leak term then swamps the parental signal
                leak_per_gene=float(leak_density * (p - 1)))
    if n_leak == 0:
        info.update(n_leak_on_true_edge=0, n_leak_off_edge=0)
        return Beta, info
    pick = rng.choice(len(off), size=n_leak, replace=False)
    betas = rng.uniform(-leak_effect, leak_effect, size=n_leak)
    on_edge = 0
    for idx, b in zip(pick, betas):
        k, j = off[idx]
        Beta[k, j] = b
        if W is not None and W[k, j] != 0:
            on_edge += 1
    info.update(n_leak_on_true_edge=int(on_edge), n_leak_off_edge=int(n_leak - on_edge))
    return Beta, info

def simulate_NB_crossguide(W: np.ndarray,
                           theta_m: np.ndarray,
                           theta_r: np.ndarray,
                           n: int,
                           leak_density: float = 0.05,
                           leak_effect: float = 3.0,
                           leak_total_sd: float = None,
                           leak_noise_sd: float = 0.0,
                           leak_rng=None,
                           intercept: bool = True,
                           g: callable = lambda x: x,
                           d: int = 0,
                           shift: float = 1.0,
                           coef: float = 1.0,
                           covar: float = 1.0,
                           intervene_prob: float = 0.9,
                           hard_intervention: bool = True,
                           ) -> tuple:
    def _simulate_single_equation(X, w, cov_m, cov_r, mask, shift_m, shift_r, leak):
        n = X.shape[0]
        cov_m = cov_m.reshape(-1, 1)
        cov_r = cov_r.reshape(-1, 1)
        X = X.reshape(n, -1)
        w = w.reshape(-1, 1)
        Xw = (X @ w).reshape(-1, 1)
        shift_m = shift_m.reshape(-1, 1)
        shift_r = shift_r.reshape(-1, 1)
        mask = mask.reshape(-1, 1)
        leak = leak.reshape(-1, 1)
        # the leak sits OUTSIDE the (1 - mask) factor, beside shift_m: it is a direct
        # effect of another gene's guide, so a hard intervention on gene j does not
        # remove it
        m = (Xw + cov_m) * (1 - mask) + shift_m + leak
        r = cov_r * (1 - mask) + shift_r
        if np.max(m) > 6.5:
            m -= (np.max(m) - 6.5)
        if np.mean(m) < .5:
            m += (.5 - np.mean(m))
        # the mean floor above can undo the cap when one cell is extreme, leaving
        # max(m) far above 6.5 and exp(m) beyond int64, which makes nbinom.rvs
        # return negative counts. Clip elementwise: a no-op unless that happens.
        m = np.minimum(m, 6.5)
        m = np.exp(m)
        r = np.exp(r)
        return nbinom.rvs(r.flatten(), r.flatten() / (m.flatten() + r.flatten()))

    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == p
    X = np.zeros([n, p])

    if leak_rng is None:
        leak_rng = np.random.default_rng()
    # probability that a given cell carries a given guide, exactly as the assignment
    # loop below constructs it
    q = (int(n * intervene_prob) // p) / n
    if leak_total_sd is not None:
        leak_effect = effect_for_total_sd(leak_total_sd, leak_density, p, q)
    Beta, leak_info = draw_leak_matrix(p, leak_density, leak_effect, leak_rng, W=W)
    leak_info['leak_noise_sd'] = float(leak_noise_sd)
    leak_info['leak_total_sd_target'] = (float(leak_total_sd)
                                         if leak_total_sd is not None else None)
    leak_info['q_guide_per_cell'] = float(q)

    Um = np.random.uniform(-coef, coef, size=(d, p))
    Ur = np.random.uniform(-coef, coef, size=(d, p))
    if intercept:
        Um = np.vstack((theta_m, Um))
        Ur = np.vstack((theta_r, Ur))

    Z = np.random.normal(0, covar, size=(n, d))
    if intercept:
        Z = np.hstack((np.ones((n, 1)), Z))
        d += 1
    ones = np.ones((n, 1))

    Vm = np.random.uniform(-1.5 * shift, -shift, size=(1, p))
    Vr = np.random.uniform(-shift / 2, shift / 2, size=(1, p))

    covar_m = Z @ Um
    covar_r = Z @ Ur
    shift_m = ones @ Vm
    shift_r = ones @ Vr

    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    intervene_mask = np.zeros((n, p), dtype=bool)
    remaining_nodes = intervene_idx
    each_intervention_size = len(intervene_idx) // p
    for i in range(p):
        choice = np.random.choice(remaining_nodes, size=each_intervention_size, replace=False)
        intervene_mask[choice, i] = True
        remaining_nodes = np.setdiff1d(remaining_nodes, choice)

    shift_m_masked = shift_m * intervene_mask
    shift_r_masked = shift_r * intervene_mask

    # leak into every gene from every other gene's guide; zero when Beta is zero, so the
    # leak_density = 0 path is numerically identical to simulate_NB_tgt
    Dfloat = intervene_mask.astype(float)
    # Carrier for source k. With leak_noise_sd = 0 this is a purely deterministic direct
    # off-target effect. A hidden mediator M_k = D_k + eps_k feeding its recipients is the
    # same object with leak_noise_sd > 0: the DETERMINISTIC part lies in the span of the
    # full first stage and is absorbed by it, whereas eps_k is not a function of D at all,
    # so no first stage can absorb it -- and being shared across every gene that source
    # feeds, it acts as an unmeasured confounder with correlated loadings. Sweeping
    # leak_noise_sd interpolates between a direct off-target effect and a mediated one.
    # Drawn from leak_rng, so leak_density = 0 (Beta = 0) still reproduces
    # simulate_NB_tgt bit-for-bit whatever the noise setting.
    carrier = Dfloat
    if leak_noise_sd > 0:
        carrier = Dfloat + leak_noise_sd * leak_rng.normal(size=(n, p))
    leak_all = carrier @ Beta                     # (n, p)

    sig_sd, leak_sd = [], []
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        mask_j = intervene_mask[:, j] if hard_intervention else np.zeros(n)
        if len(parents):
            sig_sd.append(float((g(X[:, parents]) @ W[parents, j]).std()))
            leak_sd.append(float(leak_all[:, j].std()))
        X[:, j] = _simulate_single_equation(
            g(X[:, parents]), W[parents, j], covar_m[:, j], covar_r[:, j],
            mask_j, shift_m_masked[:, j], shift_r_masked[:, j], leak_all[:, j]).reshape(-1)

    # realised severity: how large the leak term is relative to the parental signal it
    # competes with, in log-mean units. This is the quantity to put on a figure axis --
    # leak_density alone is not comparable across p.
    sig_sd = np.asarray(sig_sd); leak_sd = np.asarray(leak_sd)
    ok = sig_sd > 0
    # MEAN, not median: at low density most genes have no leaking guide at all, so the
    # median ratio reads 0 even where the affected genes are badly hit
    leak_info['mean_leak_sd'] = float(leak_sd.mean()) if len(leak_sd) else 0.0
    leak_info['mean_signal_sd'] = float(sig_sd.mean()) if len(sig_sd) else 0.0
    leak_info['leak_to_signal'] = (float((leak_sd[ok] / sig_sd[ok]).mean())
                                   if ok.any() else 0.0)
    # and how much of the network is touched at all
    leak_info['frac_genes_leaked'] = (float((leak_sd > 0).mean()) if len(leak_sd) else 0.0)
    return X, intervene_mask, Z, Um, Ur, Vm, Vr, Beta, leak_info

# ======================================================================================
# Pipeline
# ======================================================================================

CONFIGS = ["none", "direct_s0", "direct_s05", "mediator"]
ARMS = ["rcf", "fcf", "nocf"]


def validate_counts(X, cfg, replicate, max_plausible=1e7):
    """Fail at generation rather than in the fit stage several hours later."""
    bad = []
    if not np.isfinite(X).all():
        bad.append(f"{int((~np.isfinite(X)).sum())} non-finite entries")
    if X.min() < 0:
        gene = int(np.unravel_index(np.argmin(X), X.shape)[1])
        bad.append(f"min={X.min():.4g} (gene {gene}) -- int64 overflow in nbinom.rvs")
    if X.max() > max_plausible:
        genes = np.where(X.max(axis=0) > max_plausible)[0]
        bad.append(f"max={X.max():.4g} on {len(genes)} gene(s) "
                   f"{genes[:8].tolist()} -- the rescale cap was undone")
    if bad:
        raise SystemExit(f"[gen] FAILED replicate {replicate} config {cfg}: "
                         + "; ".join(bad))


def stem_for(a, cfg, replicate):
    return (f"n{a.n // 1000}k_p{a.p}_s{a.s0}_{a.graph}_{cfg}_um{a.d}_repl{replicate}")


def generate(a, replicate):
    """All four configurations from one DAG and one data-RNG start."""
    os.makedirs(a.datadir, exist_ok=True)
    set_random_seed(replicate)                        # DAG: shared by all configs
    B_true = simulate_dag(a.p, a.s0, a.graph)
    W_true = simulate_parameter(B_true, w_ranges=((-1, -0.5), (0.5, 1.0)))

    q = (int(a.n * 0.9) // a.p) / a.n                 # per-cell per-guide probability
    total_sd = float(np.sqrt(a.dens * (a.p - 1) * (3.0 ** 2 / 3.0) * q * (1 - q)))
    common = dict(W=W_true, theta_m=-np.ones(a.p), theta_r=np.ones(a.p), n=a.n, d=a.d,
                  intercept=True, g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
                  hard_intervention=False)
    print(f"[gen] replicate {replicate}: p={a.p} n={a.n} s0={a.s0} d={a.d} "
          f"leak_total_sd={total_sd:.4f}", flush=True)

    for cfg in a.configs:
        set_random_seed(100000 + replicate)           # same data-RNG start per config
        if cfg == "none":
            X, mask = simulate_NB_tgt(apply_clip=True, **common)[:2]
            info = {}
        elif cfg == "mediator":
            X, mask = simulate_NB_hidden_mediator(
                mediator_density=a.dens, mediator_in_effect=3.0, mediator_out_effect=3.0,
                **common)[:2]
            info = {}
        elif cfg in ("direct_s0", "direct_s05"):
            sigma = 0.0 if cfg == "direct_s0" else 0.5
            out = simulate_NB_crossguide(
                leak_density=a.dens, leak_total_sd=total_sd, leak_noise_sd=sigma,
                leak_rng=np.random.default_rng(70000 + replicate), **common)
            X, mask, info = out[0], out[1], out[-1]
        else:
            raise SystemExit(f"unknown config {cfg}")

        validate_counts(X, cfg, replicate)
        stem = stem_for(a, cfg, replicate)
        np.savez_compressed(os.path.join(a.datadir, stem + ".npz"),
                            X=X, W=W_true, mask=mask,
                            leak_to_signal=info.get("leak_to_signal", 0.0),
                            n_leak=info.get("n_leak", 0),
                            leak_per_gene=info.get("leak_per_gene", 0.0))
        print(f"[gen] {stem}  mean_count={X.mean():.1f}  max_count={X.max():.0f}  "
              f"n_leak={info.get('n_leak', 0)}  "
              f"leak/signal={info.get('leak_to_signal', 0.0):.3f}", flush=True)
    print("[gen] done", flush=True)


def fit(a, replicate):
    """Fit every arm on every configuration of one replicate."""
    from scRICE_eff import scRICE_CF_eff
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score

    if a.backend == "cupy":
        import cupy as cp
        dtype = cp.float64
    else:
        dtype = np.float64

    if a.schedule == "reduced":
        sched = dict(lr=1e-3, max_iter=30, epochs=3000, checkpoint=500)
    else:
        sched = dict(lr=1e-4, max_iter=100, epochs=int(1e4), checkpoint=2000)
    fit_kwargs = dict(pen1=a.pen1, pen2=0, pen1_c=0, pen2_c=1e-2, regularizer="expm",
                      h_tol=1e-2, loss_tol=1e-6, progress_bar=False, **sched)

    os.makedirs(a.resdir, exist_ok=True)
    rows = []
    for cfg in a.configs:
        stem = stem_for(a, cfg, replicate)
        z = np.load(os.path.join(a.datadir, stem + ".npz"), allow_pickle=True)
        X, W_true, mask = z["X"], z["W"], z["mask"]
        # the data were validated at generation; check again in case of a stale file
        validate_counts(X, cfg, replicate)
        predictor = np.log(1 + X)
        M = mask.astype(float)

        for arm in a.arms:
            res = None
            if arm != "nocf":
                res = np.zeros_like(predictor)
                for i in range(a.p):
                    Zi = M[:, [i]] if arm == "rcf" else M
                    lm = LinearRegression().fit(Zi, predictor[:, i])
                    res[:, i] = predictor[:, i] - lm.predict(Zi)
            t0 = time.time()
            model = scRICE_CF_eff(dtype=dtype, backend=a.backend)
            model.prep(X=X, predictor=predictor, confounding_res=res,
                       intervention_effect=M, intervention_type="soft",
                       add_intercept=True)
            model.fit(**fit_kwargs)
            secs = time.time() - t0

            W_raw = model.result(w_threshold=a.w_threshold)["W"]
            acc = measurement(W_true, to_dag(W_raw.copy()))
            acc["auc"] = roc_auc_score((W_true != 0).astype(int).ravel(),
                                       np.abs(W_raw).ravel())
            acc["time"] = secs
            for k, v in acc.items():
                rows.append({"config": cfg, "method": arm, "measure": k,
                             "value": float(v)})
            print(f"[fit] repl {replicate} {cfg:<11} {arm:<5} mcc={acc['mcc']:.3f} "
                  f"shd={acc['shd']:.0f} {secs:.0f}s", flush=True)

        rows.append({"config": cfg, "method": "truth", "measure": "leak_to_signal",
                     "value": float(z["leak_to_signal"])})
        rows.append({"config": cfg, "method": "truth", "measure": "n_leak",
                     "value": float(z["n_leak"])})

    for k, v in [("pen1", a.pen1), ("w_threshold", a.w_threshold), ("d", a.d),
                 ("replicate", replicate), ("max_iter", fit_kwargs["max_iter"]),
                 ("epochs", fit_kwargs["epochs"]), ("lr", fit_kwargs["lr"]),
                 ("backend_is_cupy", float(a.backend == "cupy"))]:
        rows.append({"config": "all", "method": "settings", "measure": k,
                     "value": float(v)})

    out = os.path.join(a.resdir, f"n{a.n // 1000}k_p{a.p}_s{a.s0}_{a.graph}"
                                 f"_cfcompare_um{a.d}_repl{replicate}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[fit] wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["gen", "fit", "both"], default="both")
    ap.add_argument("--replicate", type=int,
                    default=int(os.environ.get("LSB_JOBINDEX", 1)),
                    help="replicate index; defaults to $LSB_JOBINDEX")
    ap.add_argument("--model", default="NB")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--s0", type=int, default=400)
    ap.add_argument("--graph", default="ER")
    ap.add_argument("--d", type=int, default=10, help="number of latent factors")
    ap.add_argument("--dens", type=float, default=0.05,
                    help="violation density: fraction of gene pairs that leak")
    ap.add_argument("--pen1", type=float, default=1e-4,
                    help="1e-4 is the Methods value; job_simu_comp_cf.py used 1e-3. "
                         "Use ONE value across every arm and configuration.")
    ap.add_argument("--w-threshold", type=float, default=0.3, dest="w_threshold")
    ap.add_argument("--backend", choices=["cupy", "numpy"],
                    default=os.environ.get("RICE_BACKEND", "cupy"))
    ap.add_argument("--schedule", choices=["production", "reduced"],
                    default=os.environ.get("RICE_SCHEDULE", "production"))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--datadir",
                    default="result/simu_cfcompare/data/NB")
    ap.add_argument("--resdir",
                    default="result/simu_cfcompare/scGRAPH/NB")
    a = ap.parse_args()
    a.configs = [c for c in a.configs.split(",") if c]
    a.arms = [x for x in a.arms.split(",") if x]
    if "none" not in a.configs:
        print("[warn] the `none` configuration is the reference for the "
              "difference-in-differences; dropping it makes the other arms "
              "uninterpretable", flush=True)

    if a.stage in ("gen", "both"):
        generate(a, a.replicate)
    if a.stage in ("fit", "both"):
        fit(a, a.replicate)


if __name__ == "__main__":
    main()
