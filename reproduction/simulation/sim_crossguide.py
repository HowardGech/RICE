"""
Exclusion-restriction violation by DIRECT CROSS-GENE GUIDE EFFECT (design item B2).

For a fraction of ordered pairs (k, j), k != j, the guide targeting gene k has a direct
effect on gene j that does not pass through x_k:

    eta_j  +=  sum_k  beta_kj * D_k

Severity and concentration are separable. Pass `leak_effect` to fix the per-pair effect
size, or `leak_total_sd` to fix the per-gene leak magnitude and let the per-pair effect
scale as 1/sqrt(density): the latter sweeps CONCENTRATION -- a few strong leaks versus many
weak ones -- at matched severity, which is the regime where the FULL control function is
expected to lose. It must estimate p first-stage coefficients per gene however many
actually leak, so concentrated leaks give it large coefficients to absorb for that fixed
variance cost while diffuse weak leaks give it almost nothing.

This is the arm that is ADVERSARIAL to the reduced control function. RCF residualizes
gene j on D_j alone, so it cannot absorb an effect carried by D_k; the full control
function regresses gene j on all indicators and can. The prediction to be tested is
therefore that the full CF is favoured here, which is the opposite of what the published
hidden-mediator arm shows -- and that contrast is the answer to the reviewer.

Everything else replicates simulate_NB_tgt exactly, including its inner rescaling
constants (cap log-mean at 6.5, floor mean at 0.5). Note these differ from the
module-level _simulate_single_equation in sim_base.py (7.5 / -0.5) used by the
newer generators; matching simulate_NB_tgt is what keeps this arm on the same count scale
as the NB baseline it is compared against.

beta and the leak pairs are drawn from a SEPARATE Generator, so the legacy global RNG
stream is untouched and leak_density = 0 reproduces simulate_NB_tgt bit for bit.
"""
import numpy as np
import igraph as ig
from scipy.stats import nbinom

from sim_base import is_dag


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
