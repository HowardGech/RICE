#!/usr/bin/env python
"""
SPARSE NONLINEAR CONFOUNDING: only a fraction of genes is confounded.

The mechanism and its parameters are FIXED; the only thing that varies across the sweep is
the fraction of genes carrying latent loadings. Unconfounded genes have their loadings set
to zero in both the mean and the dispersion, so "unconfounded" means exactly that.

For a confounded gene j, with u_ij = z_i' Um[:, j] the linear latent index:

    eta^m_ij = (1 + gamma * tanh(s_ij)) * sum_{k in Pa(j)} w_kj g(x_ik)     <- heterogeneity
               + T_alpha(u_ij)                                             <- additive
               + v^m_j D_ij
    eta^r_ij = z_i' Ur[:, j] + v^r_j D_ij

with s_ij the per-gene standardised index and T_alpha a variance-matched mix of the linear
index and the cubic He3. Both channels are needed and they do different jobs:

  THE ADDITIVE TERM IS WHY THERE IS CONFOUNDING. The same z reaches many genes, inducing
  association between genes with no edge between them. Gain modulation alone would not do
  this: a gene with no parents has no parental sum to modulate, so the latent state has no
  channel into it at all, and E[tanh(s)] = 0 leaves every gene's marginal mean unchanged.
  Gain modulation on its own is effect heterogeneity, not confounding.

  THE GAIN IS WHY RESIDUALISING CANNOT REMOVE IT. Expanding the product gives a term
  gamma * tanh(s_ij) * sum_k w_kj g(x_ik), bilinear in z and x. The model's only latent
  adjustment is additive in the mean and linear in the residuals (A_cf = gX_c @ W_c), so no
  choice of W_c represents a multiplicative distortion of the parental sum. The same
  expression ADDED to eta instead would be nonlinear but additive, and an additive
  correction could still remove it.

WHY THESE SETTINGS

  gamma = 1.0. Admissible only because tanh bounds the state: the gain is 1 + gamma*tanh(s),
  strictly inside (0, 2), so it never changes sign. An unbounded state would flip the sign of
  every regulatory effect in 16% of cells at gamma = 1.

  alpha = 0.5. Hermite polynomials transmit correlation as rho^k, so a cubic transform CUBES
  the cross-gene alignment. At the measured median |cos(u_j, u_k)| = 0.225 for d = 10, a
  purely cubic additive term (alpha = 1) induces gene-gene correlation of about 0.011 instead
  of 0.225 -- nonlinear, but with almost no association left to remove, which would make the
  confounding harmless rather than hard. At alpha = 0.5 the variance splits evenly between
  the linear and cubic parts, so half is orthogonal to every linear function of z while 53%
  of the linear case's alignment survives.

  nonlinearity = He3(t) = (t^3 - 3t)/sqrt(6), the third probabilists' Hermite polynomial. The
  -3t makes it orthogonal to the linear index, E[t^4] - 3E[t^2] = 0, so all of what alpha
  mixes in is new. A raw cube is 0.774 correlated with the index and would mostly re-add the
  linear term. The price is monotonicity: any zero-mean cubic orthogonal to t is a multiple
  of He3, whose derivative 3t^2 - 3 is negative on |t| < 1, so it is non-monotone. That is a
  genuinely different shape from the monotone saturating tanh used for the gain.

READ THE X AXIS AS PAIRS, NOT GENES. Only a pair with BOTH genes confounded carries spurious
association, so the affected share goes as (frac * p)^2. At p = 100 the gene fraction
0.1 -> 0.5 is 5-fold while the pair fraction 0.009 -> 0.247 is 27-fold. frac_confounded_pairs
is written to every file for this reason.

    python sim_sparse_confounding.py                      # the production sweep
    python sim_sparse_confounding.py --selftest
    python sim_sparse_confounding.py --fracs 0,0.3,1.0 --n 3000 --p 20 --s0 60 --d 6

Writes to result/simu_{intervention}/data/{model}, where the
fit driver already looks, as

    n{n/1000}k_p{p}_s{s0}_{graph}_cova{d}_conf{100*frac:03d}_repl{job}.npz

with keys X, W, mask, Z, Um, Ur, Vm, Vr, so the fit driver reads it after a filename change
only. Two notes on that name. The `conf###` tag sits between the factor count and `_repl`,
so a parser must match the setting and the replicate SEPARATELY -- a pattern expecting
`_repl` straight after the cova digits will not match. And these files sit in the SAME
directory as the pre-existing dense `..._cova{d}_repl{job}.npz` arm, so a glob on
`*_cova{d}_*` now matches both; narrow it by `conf` when selecting this arm.
"""
import argparse, json, os, sys
import numpy as np
import igraph as ig
from scipy.stats import nbinom

sys.path.insert(0, os.environ.get("SCGRAPH_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)))))
from sim_base import is_dag
from utils import simulate_dag, simulate_parameter, set_random_seed

FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
APPLY_CLIP = True      # set False only to verify equivalence with the unpatched generator


# ----------------------------------------------------------------------------------------
# generator internals
# ----------------------------------------------------------------------------------------


def _equation(X, w, cov_m, cov_r, mask, shift_m, shift_r, cap=6.5, floor=0.5, gain=None):
    """Inner equation of simulate_NB_tgt, verbatim, including its 6.5 / 0.5 constants.

    Those differ from the module-level helper's 7.5 / -0.5 in sim_base.py; matching
    simulate_NB_tgt is what keeps this arm on the count scale of the published baseline.

    The elementwise clip is the overflow fix: the cap and the floor are two sequential shifts
    and the second can undo the first, after which exp(m) can leave the integer range. It
    binds only where the unpatched code would have overflowed, so output here is NOT
    bit-identical to unpatched output -- do not mix patched and unpatched replicates.
    """
    n = X.shape[0]
    cov_m = cov_m.reshape(-1, 1); cov_r = cov_r.reshape(-1, 1)
    X = X.reshape(n, -1); w = w.reshape(-1, 1)
    Xw = (X @ w).reshape(-1, 1)
    if gain is not None:
        # The gain multiplies the PARENTAL SUM only -- not cov_m, not shift_m. Keeping it off
        # the perturbation shift is deliberate: the instrument stays valid, so this arm tests
        # confounding alone rather than mixing in an exclusion violation.
        Xw = Xw * gain.reshape(-1, 1)
    shift_m = shift_m.reshape(-1, 1); shift_r = shift_r.reshape(-1, 1)
    mask = mask.reshape(-1, 1)
    m = (Xw + cov_m) * (1 - mask) + shift_m
    r = cov_r * (1 - mask) + shift_r
    if np.max(m) > cap:
        m -= (np.max(m) - cap)
    if np.mean(m) < floor:
        m += (floor - np.mean(m))
    if APPLY_CLIP:
        m = np.clip(m, None, cap)
    m = np.exp(m).flatten()
    r = np.exp(r).flatten()
    return nbinom.rvs(r, r / (m + r))


def standardised_index(lin):
    """lin / sd, per gene, guarding genes whose latent contribution is identically zero."""
    sd = lin.std(axis=0)
    return lin / np.where(sd > 0, sd, 1.0), sd


def bounded_state(lin):
    """s_ij = tanh(z_i' u_j / sd_i), bounded in (-1, 1), one value per cell and gene.

    The per-gene standardisation sits INSIDE the tanh so every gene's state has the same
    marginal distribution; without it gamma would mean different things for different genes
    and at different numbers of latent factors, since ||u_j|| grows as sqrt(d/3).

    tanh rather than a clip: at unit input SD a clip pins about 32% of cells to exactly +-1,
    so a third of them would share one gain and the state would stop being continuous. tanh
    leaves under 1% near the bound and is strictly monotone, preserving the ordering of the
    latent index exactly.
    """
    t, _ = standardised_index(lin)
    return np.tanh(t)


def cubic_mix(lin, alpha, winsor=4.0):
    """Mix the additive latent term from linear toward He3, holding its moments fixed.

    Returns the mixed term and diagnostics. alpha = 0 returns `lin` untouched, so the branch
    is skipped entirely and nothing downstream changes.
    """
    info = {"alpha": float(alpha)}
    sd0 = lin.std(axis=0)
    mu0 = lin.mean(axis=0)
    info["additive_sd_before"] = float(sd0.mean())
    if alpha <= 0:
        info["additive_sd_after"] = info["additive_sd_before"]
        info["additive_corr_with_linear"] = 1.0
        info["additive_frac_winsorised"] = 0.0
        return lin, info

    t, _ = standardised_index(lin)
    nl = (t ** 3 - 3.0 * t) / np.sqrt(6.0)
    # He3 has excess kurtosis about 86. The rescale in _equation shifts a whole gene's
    # predictor by (max(m) - cap) whenever the cap is exceeded -- a column-wide shift driven
    # by its single most extreme cell -- so unwinsorised, one far-tail cell drags that gene's
    # entire expression level down and the arm fails mechanically rather than statistically.
    # The bound is symmetric, leaving the function odd and its orthogonality intact.
    if winsor:
        info["additive_frac_winsorised"] = float(np.mean(np.abs(nl) > winsor))
        nl = np.clip(nl, -winsor, winsor)
    else:
        info["additive_frac_winsorised"] = 0.0

    mixed = (1.0 - alpha) * lin + alpha * nl * np.where(sd0 > 0, sd0, 1.0)
    # restore each gene's mean and SD, so alpha changes the FORM of the confounding and not
    # its magnitude -- the comparison would otherwise confound the two
    mu1, sd1 = mixed.mean(axis=0), mixed.std(axis=0)
    mixed = (mixed - mu1) * np.where(sd1 > 0, sd0 / np.where(sd1 > 0, sd1, 1.0), 1.0) + mu0
    info["additive_sd_after"] = float(mixed.std(axis=0).mean())

    cors = [abs(np.corrcoef(lin[:, j], mixed[:, j])[0, 1])
            for j in range(lin.shape[1])
            if lin[:, j].std() > 0 and mixed[:, j].std() > 0]
    info["additive_corr_with_linear"] = float(np.mean(cors)) if cors else 0.0
    return mixed, info


def sparsify_loadings(Um, Ur, frac, rng, intercept=True, scale=1.0):
    """Zero the latent loadings of the unconfounded genes; leave the rest exactly as drawn.

    The intercept row is never touched, and the columns are zeroed AFTER the draws, so the
    global stream is untouched and frac = 1 leaves the published loadings intact.
    """
    p = Um.shape[1]
    r0 = 1 if intercept else 0
    if frac is None or frac >= 1.0:
        keep = np.ones(p, dtype=bool)
        if scale != 1.0:
            Um = Um.copy(); Ur = Ur.copy()
            Um[r0:, :] *= scale
            Ur[r0:, :] *= scale
        return Um, Ur, keep, dict(frac_confounded=1.0, n_confounded_genes=p,
                                  conf_scale=float(scale),
                                  frac_confounded_pairs=1.0, confounded_genes=None)
    n_conf = int(round(frac * p))
    conf = np.sort(rng.choice(p, size=n_conf, replace=False)) if n_conf else np.array([], int)
    keep = np.zeros(p, dtype=bool)
    keep[conf] = True
    Um = Um.copy(); Ur = Ur.copy()
    Um[r0:, ~keep] = 0.0
    Ur[r0:, ~keep] = 0.0
    # Confounded genes are amplified because the confounding is now sparse: with only a
    # minority of genes affected, per-gene strength is raised so the arm probes strong
    # confounding on few genes rather than weak confounding on few genes.
    Um[r0:, keep] *= scale
    Ur[r0:, keep] *= scale
    info = dict(frac_confounded=float(frac), n_confounded_genes=int(n_conf),
                conf_scale=float(scale),
                # Only a pair with BOTH genes confounded carries spurious association, so the
                # affected share goes as (frac*p)^2 and moves far faster than the gene
                # fraction. Plot against this, or state it in the caption.
                frac_confounded_pairs=float((n_conf * (n_conf - 1)) /
                                            max(p * (p - 1), 1)),
                confounded_genes=conf.tolist())
    return Um, Ur, keep, info


def simulate_sparse_confounding(W, theta_m, theta_r, n, d=6, intercept=True,
                                g=lambda x: x, shift=2.0, coef=1.0, covar=1.0,
                                intervene_prob=0.9, hard_intervention=False,
                                frac=1.0, gamma=0.5, alpha=0.5, winsor=4.0,
                                conf_scale=3.0, aux_rng=None):
    """simulate_NB_tgt with sparse, nonlinear, non-additive confounding.

    frac = 1, gamma = 0, alpha = 0, conf_scale = 1 reproduces the published simulate_NB_tgt construction: the
    sparsity, gain and cubic branches are all skipped and the auxiliary stream is never drawn
    from, so the main NumPy stream is untouched and the auxiliary seed makes no difference.
    """
    if aux_rng is None:
        aux_rng = np.random.default_rng()
    p = W.shape[0]
    if not is_dag(W):
        raise ValueError("W must be a DAG")
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

    Um, Ur, keep, info = sparsify_loadings(Um, Ur, frac, aux_rng, intercept=intercept,
                                           scale=conf_scale)

    covar_m = Z @ Um
    covar_r = Z @ Ur

    # the latent part of the mean, with the intercept removed, is what the mechanisms act on
    lat = covar_m - (Z[:, :1] @ Um[:1, :]) if intercept else covar_m
    info["latent_sd_confounded"] = (float(lat[:, keep].std(axis=0).mean())
                                   if keep.any() else 0.0)
    info["latent_sd_unconfounded"] = (float(lat[:, ~keep].std(axis=0).mean())
                                     if (~keep).any() else 0.0)
    # Confounding measured against the designed perturbation signal, both on the linear-
    # predictor scale, so the ratio is unit-free. At 3x the dense loadings this is the number
    # to check before launching: it says how large the confounding is relative to the effect
    # the method is trying to recover.
    info["confounding_to_perturbation"] = float(
        info["latent_sd_confounded"] / (np.abs(Vm).mean() + 1e-12))

    # NOTE: a per-gene constant shift cannot control the count scale here. _equation
    # renormalises every gene's predictor -- m -= (max(m) - cap) then m += (floor - mean(m))
    # -- so any additive per-gene correction is absorbed, and the depth is set by the SPREAD
    # of eta against the 6-unit window between floor 0.5 and cap 6.5. An exp(sd^2/2)
    # correction was tried and made the drift worse (20-fold to 37-fold), because lowering
    # the mean only triggers the floor to push it back. The scale of the loadings is the
    # knob; see --conf-scale.

    if gamma:
        # the state is built from the LINEAR index, so gamma and alpha stay independent
        S = bounded_state(lat)
        gain_all = 1.0 + gamma * S
        info.update(gamma=float(gamma),
                    gain_min=float(gain_all.min()), gain_max=float(gain_all.max()),
                    gain_sd=float(gain_all.std(axis=0).mean()),
                    gain_frac_nonpositive=float(np.mean(gain_all <= 0)),
                    gain_frac_below_0p1=float(np.mean(gain_all < 0.1)),
                    # genes with no parents have no parental sum for the gain to modulate, so
                    # for them the non-additive term is identically zero and only the additive
                    # confounding acts; this bounds how much of the graph gamma can affect
                    n_root_genes=int(((W != 0).sum(axis=0) == 0).sum()))
    else:
        gain_all = None
        info["gamma"] = 0.0

    if alpha:
        mixed, nl_info = cubic_mix(lat, alpha, winsor=winsor)
        covar_m = mixed + (Z[:, :1] @ Um[:1, :]) if intercept else mixed
        info.update(nl_info)
    else:
        info["alpha"] = 0.0

    shift_m = ones @ Vm
    shift_r = ones @ Vr

    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    intervene_mask = np.zeros((n, p), dtype=bool)
    remaining = intervene_idx
    each = len(intervene_idx) // p
    for i in range(p):
        choice = np.random.choice(remaining, size=each, replace=False)
        intervene_mask[choice, i] = True
        remaining = np.setdiff1d(remaining, choice)

    shift_m_masked = shift_m * intervene_mask
    shift_r_masked = shift_r * intervene_mask

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        mask_j = intervene_mask[:, j] if hard_intervention else np.zeros(n)
        X[:, j] = _equation(g(X[:, parents]), W[parents, j], covar_m[:, j], covar_r[:, j],
                            mask_j, shift_m_masked[:, j], shift_r_masked[:, j],
                            gain=None if gain_all is None else gain_all[:, j]).reshape(-1)

    return X, intervene_mask, Z, Um, Ur, Vm, Vr, info


def validate_counts(X, where):
    """Fail at generation, naming the offending genes, not hours later in the fit."""
    bad = []
    if not np.all(np.isfinite(X)):
        bad.append("non-finite")
    neg = np.where((X < 0).any(axis=0))[0]
    if neg.size:
        bad.append(f"negative in genes {neg[:10].tolist()}")
    huge = np.where((X > 1e8).any(axis=0))[0]
    if huge.size:
        bad.append(f"exceeding 1e8 in genes {huge[:10].tolist()}")
    if bad:
        raise RuntimeError(f"{where}: invalid counts -- " + "; ".join(bad))


# ----------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------
def stem_for(a, frac, replicate):
    return (f"n{a.n // 1000}k_p{a.p}_s{a.s0}_{a.graph}_cova{a.d}_"
            f"conf{int(round(frac * 100)):03d}_repl{replicate}")


def generate(a, replicate):
    os.makedirs(a.datadir, exist_ok=True)
    rows = []
    for frac in a.fracs:
        # One graph and one data-RNG start per replicate, shared by every fraction, so the
        # settings are PAIRED within a replicate and the only thing differing between them is
        # which genes are confounded.
        set_random_seed(replicate)
        B = simulate_dag(a.p, a.s0, a.graph)
        W = simulate_parameter(B, w_ranges=((-1, -0.5), (0.5, 1.0)))
        X, mask, Z, Um, Ur, Vm, Vr, info = simulate_sparse_confounding(
            W=W, theta_m=-np.ones(a.p), theta_r=np.ones(a.p), n=a.n, d=a.d,
            intercept=True, g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
            intervene_prob=0.9, hard_intervention=(a.intervention == "hard"),
            frac=frac, gamma=a.gamma, alpha=a.alpha, winsor=a.winsor,
            conf_scale=a.conf_scale,
            aux_rng=np.random.default_rng(70000 + replicate))
        validate_counts(X, f"frac={frac} replicate={replicate}")
        stem = stem_for(a, frac, replicate)
        np.savez(os.path.join(a.datadir, stem + ".npz"),
                 X=X, W=W, mask=mask, Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr,
                 info=json.dumps(info))
        info.update(replicate=replicate, mean_count=float(X.mean()),
                    frac_zero=float((X == 0).mean()))
        rows.append(info)
        print(f"[gen] {stem}  genes={info['n_confounded_genes']:>3} "
              f"pairs={info['frac_confounded_pairs']:.3f} "
              f"latent_sd conf={info['latent_sd_confounded']:.3f} "
              f"unconf={info['latent_sd_unconfounded']:.3f} "
              f"gain=[{info.get('gain_min', 1):.3f},{info.get('gain_max', 1):.3f}] "
              f"corr_lin={info.get('additive_corr_with_linear', 1):.3f} "
              f"mean_count={info['mean_count']:.2f} zero={info['frac_zero']:.3f}",
              flush=True)
    with open(os.path.join(a.datadir, f"gen_sparseconf_repl{replicate}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def selftest():
    global APPLY_CLIP
    p_, n_, d_ = 20, 3000, 6
    common = dict(theta_m=-np.ones(p_), theta_r=np.ones(p_), n=n_, d=d_, intercept=True,
                  g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
                  hard_intervention=False)

    def W_(seed=11):
        set_random_seed(seed)
        B = simulate_dag(p_, 3 * p_, "ER")
        return simulate_parameter(B, w_ranges=((-1, -0.5), (0.5, 1.0)))

    def run(**kw):
        W = W_()
        return simulate_sparse_confounding(W=W, aux_rng=np.random.default_rng(7),
                                           **common, **kw)

    # (a) the published generator, recovered exactly with the clip disabled
    from sim_base import simulate_NB_tgt
    APPLY_CLIP = False
    Wp = W_(); Xpub = simulate_NB_tgt(W=Wp, **common)[0]
    Xmine = run(frac=1.0, gamma=0.0, alpha=0.0, conf_scale=1.0)[0]
    same = np.array_equal(Xpub, Xmine)
    APPLY_CLIP = True
    assert same, "frac=1, gamma=0, alpha=0 differs from simulate_NB_tgt beyond the clip"
    Xclip = run(frac=1.0, gamma=0.0, alpha=0.0, conf_scale=1.0)[0]
    nbad = int((np.abs(Xpub - Xclip).sum(axis=0) > 0).sum())
    print(f"[selftest] frac=1, gamma=0, alpha=0 equals simulate_NB_tgt bit-for-bit with the "
          f"overflow clip disabled; with it enabled {nbad} of {p_} genes differ")

    # (b) the auxiliary seed must not matter when every branch is off
    W = W_()
    alt = simulate_sparse_confounding(W=W, aux_rng=np.random.default_rng(999),
                                      frac=1.0, gamma=0.0, alpha=0.0,
                                      conf_scale=1.0, **common)[0]
    assert np.array_equal(Xclip, alt), "baseline depends on the auxiliary seed"

    # (c) He3 is orthogonal to the linear index, and the mix holds moments fixed
    rng = np.random.default_rng(3)
    lin = rng.normal(size=(40000, 5)) * np.array([1.0, 2.0, 0.5, 1.5, 3.0])
    for al in (0.5, 1.0):
        mixed, ci = cubic_mix(lin, al, winsor=0.0)
        rel = abs(ci["additive_sd_after"] - ci["additive_sd_before"]) / ci["additive_sd_before"]
        assert rel < 1e-9, f"alpha={al}: additive sd moved by {rel:.2e}"
    t = lin[:, 0] / lin[:, 0].std()
    he3 = (t ** 3 - 3 * t) / np.sqrt(6)
    print(f"[selftest] He3 orthogonality: corr with the linear index "
          f"{abs(np.corrcoef(t, he3)[0, 1]):.4f} (0 in closed form), "
          f"variance {np.var(t ** 3 - 3 * t):.3f} (6 in closed form); "
          f"mix holds each gene's mean and sd fixed to 1e-9")

    # (d) sparsity: the right number of genes confounded, the rest carrying exactly nothing
    counts = []
    print(f"[selftest] {'frac':>5} {'genes':>6} {'pairs':>7} {'sd conf':>8} {'sd unconf':>10} "
          f"{'gain range':>18} {'nonpos':>7} {'mean count':>11}")
    for fr in (0.0, 0.1, 0.3, 0.5, 1.0):
        X, _, _, Um, Ur, _, _, i = run(frac=fr, gamma=0.5, alpha=0.5, conf_scale=3.0)
        counts.append(X.mean())
        assert i["n_confounded_genes"] == (p_ if fr >= 1.0 else int(round(fr * p_))), \
            "wrong number of confounded genes"
        assert i["latent_sd_unconfounded"] == 0.0 or fr >= 1.0, \
            "an unconfounded gene carries a non-zero latent contribution"
        if 0 < fr < 1.0:
            nz = (np.abs(Um[1:, :]).sum(axis=0) > 0).sum()
            assert nz == i["n_confounded_genes"], "zeroed columns disagree with the count"
        if fr > 0:
            # the bound is what makes gamma = 1 admissible at all
            assert i["gain_frac_nonpositive"] == 0.0, "gain went non-positive"
            assert i["gain_min"] > 0.5 - 1e-9 and i["gain_max"] < 1.5 + 1e-9, \
                "gain left (1 - gamma, 1 + gamma) at gamma = 0.5"
        validate_counts(X, f"selftest frac={fr}")
        print(f"[selftest] {fr:>5.1f} {i['n_confounded_genes']:>6} "
              f"{i['frac_confounded_pairs']:>7.3f} {i['latent_sd_confounded']:>8.3f} "
              f"{i['latent_sd_unconfounded']:>10.3f} "
              f"[{i.get('gain_min', 1):.3f}, {i.get('gain_max', 1):.3f}]".rjust(19) +
              f" {i.get('gain_frac_nonpositive', 0):>7.4f} {X.mean():>11.2f}")
    # mean count must not drift with the number of confounded genes
    drift = max(counts) / max(min(counts), 1e-12)
    if drift >= 2.0:
        print(f"[selftest] WARNING mean count drifts {drift:.1f}-fold across the sweep; "
              f"lower --conf-scale or the comparison picks up a depth effect")
    print(f"[selftest] mean count across the sweep: "
          f"{', '.join(f'{c:.1f}' for c in counts)} ({drift:.2f}-fold spread)")

    # (e) the scale multiplies the retained loadings exactly
    _, _, _, _, _, _, _, i1 = run(frac=0.3, gamma=0.0, alpha=0.0, conf_scale=1.0)
    _, _, _, _, _, _, _, i3 = run(frac=0.3, gamma=0.0, alpha=0.0, conf_scale=3.0)
    ratio = i3["latent_sd_confounded"] / max(i1["latent_sd_confounded"], 1e-12)
    assert abs(ratio - 3.0) < 1e-6, f"conf_scale=3 gave a latent sd ratio of {ratio:.4f}"
    print(f"[selftest] conf_scale=3 scales the confounded latent sd by {ratio:.4f} "
          f"({i1['latent_sd_confounded']:.3f} -> {i3['latent_sd_confounded']:.3f}); "
          f"confounding/perturbation {i1['confounding_to_perturbation']:.2f} -> "
          f"{i3['confounding_to_perturbation']:.2f}")
    print("[selftest] OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fracs", default=",".join(str(x) for x in FRACS),
                    help="fractions of genes carrying latent loadings. 0 is the "
                         "no-confounding reference and is NOT optional: without it a "
                         "difference across the sweep cannot be told from a baseline that "
                         "was never zero. Add 1.0 if you also want the dense arm in the same "
                         "element, though at conf_scale != 1 it is not the published cova "
                         "arm but a 3x-amplified version of it.")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="heterogeneity: gain = 1 + gamma*tanh(s), held fixed across the "
                         "sweep. Strictly positive for gamma < 1 because tanh bounds s.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="nonlinearity of the additive term: variance-matched mix of the "
                         "linear index and He3. 1.0 cubes the cross-gene alignment and makes "
                         "the confounding nearly harmless; 0.5 keeps 53% of it.")
    ap.add_argument("--conf-scale", type=float, default=3.0, dest="conf_scale",
                    help="multiplier on the loadings of the confounded genes, relative to "
                         "the dense regime. 3.0 compensates for sparsity: strong confounding "
                         "on few genes rather than weak confounding on few genes.")
    ap.add_argument("--winsor", type=float, default=4.0,
                    help="bound in SD units on the cubic component; 0 disables it")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--s0", type=int, default=400)
    ap.add_argument("--graph", default="ER")
    ap.add_argument("--d", type=int, default=6, help="number of latent factors")
    ap.add_argument("--intervention", choices=["soft", "hard"], default="soft")
    ap.add_argument("--replicate", type=int,
                    default=int(os.environ.get("LSB_JOBINDEX", 1)))
    ap.add_argument("--model", default="NB")
    ap.add_argument("--datadir", default=None,
                    help="defaults to the project convention, "
                         "result/simu_{intervention}/data/"
                         "{model}, which is where the fit driver already looks")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    a.fracs = [float(x) for x in a.fracs.split(",")]
    if a.datadir is None:
        a.datadir = (f"result/"
                     f"simu_{a.intervention}/data/{a.model}")
    print(f"[cfg] gamma={a.gamma} alpha={a.alpha} conf_scale={a.conf_scale} "
          f"winsor={a.winsor} d={a.d} "
          f"intervention={a.intervention} -> {a.datadir}", flush=True)
    generate(a, a.replicate)


if __name__ == "__main__":
    main()
