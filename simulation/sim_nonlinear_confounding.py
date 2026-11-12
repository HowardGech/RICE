#!/usr/bin/env python
"""
Nonlinear latent confounding: two data-generating processes.

Both act ONLY on the latent contribution to gene j's linear predictor, leaving the graph,
the weights, the perturbation effects and the assignment untouched, so they are directly
comparable with the published linear-confounding runs.

Write u_ij = z_i' Um[:, j] for the linear latent index of gene j in cell i.

  FORM "saturating" (--form saturating)
      The confounder's influence is monotone but bounded: a latent state such as cell size
      or cycle phase cannot push expression arbitrarily far.

          t_ij = tanh(u_ij / s_j),        s_j = sd_i(u_ij)

      t is strongly correlated with u, so a linear adjustment recovers most of it; what it
      cannot recover is the curvature. This is the MILD, biologically plausible case.

  FORM "interaction" (--form interaction)
      The confounder acts through products of factors rather than through each factor
      separately:

          q_ij = sum_{a < b} c_j,ab * z_ia * z_ib,     c_j,ab ~ U(-1, 1)

      For independent Gaussian factors E[z_a z_b] = 0 and z_a z_b is uncorrelated with
      EVERY linear function of z, so q lies outside the span of the factors themselves. No
      linear adjustment on z, and no control function that is linear in a proxy for z, can
      absorb any part of it. This is the ADVERSARIAL case.

  (A third form, a centred per-gene quadratic, is already implemented as --nonlin-alpha in
  run_sensitivity_arms.py. The three span mild monotone, per-gene even, and shared interaction.)

SEVERITY VERSUS FORM. For each gene the mixed latent term is

    alpha = 0:  lin_ij                                  (published generator, bit-exact)
    alpha > 0:  standardise[(1 - alpha) * lin + alpha * nl]  rescaled to the mean and
                standard deviation of lin, per gene

so alpha moves the FORM of the confounding and holds its first two moments fixed. This is
deliberate and is the point of the design: the earlier cross-guide arms were compared at
unequal magnitude, and a confounding arm that grew as it bent would not identify which of
the two caused any change in performance. `latent_sd_before/after` in the saved info are
the check; they agree to floating point by construction.

    python sim_nonlinear_confounding.py --form saturating  --replicate 3
    python sim_nonlinear_confounding.py --form interaction --alphas 0,0.25,0.5,0.75,1.0
    python sim_nonlinear_confounding.py --selftest

Writes one .npz per (form, alpha, replicate) with keys X, W, mask, Z, Um, Ur, Vm, Vr and
the per-setting diagnostics, so dotears_run.py, IBCD_prepare.py and the inspre scripts read
it with a path change only.
"""
import argparse, json, os, sys
import numpy as np
import igraph as ig
from scipy.stats import nbinom

sys.path.insert(0, os.environ.get("SCGRAPH_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)))))
from sim_base import is_dag
from utils import simulate_dag, simulate_parameter, set_random_seed


APPLY_CLIP = True      # set False only to verify equivalence with the unpatched generator


def _equation(X, w, cov_m, cov_r, mask, shift_m, shift_r, cap=6.5, floor=0.5,
              gain=None, r_clip=None):
    """Inner equation of simulate_NB_tgt, verbatim, including its 6.5 / 0.5 constants.

    Those differ from the module-level helper's 7.5 / -0.5 in sim_base.py;
    matching simulate_NB_tgt is what keeps these arms on the count scale of the published
    NB baseline they are compared against. The elementwise clip after the floor is the
    overflow fix: the cap and the floor are two sequential shifts and the second can undo
    the first, after which exp(m) can leave the integer range.
    """
    n = X.shape[0]
    cov_m = cov_m.reshape(-1, 1); cov_r = cov_r.reshape(-1, 1)
    X = X.reshape(n, -1); w = w.reshape(-1, 1)
    Xw = (X @ w).reshape(-1, 1)
    if gain is not None:
        # The gain multiplies the PARENTAL SUM only -- not cov_m and not shift_m. Keeping it
        # off the perturbation shift is deliberate: the instrument stays valid, so this arm
        # tests confounding alone rather than mixing in an exclusion violation.
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
    if r_clip is not None:
        # _equation clips m but never r. phi = exp(r) and the NB variance is mu + mu^2/phi,
        # so a negative excursion in r sends the variance to enormous values and reproduces
        # the overflow pathology by a second route. Applied only when the dispersion arm is
        # active, so the baseline stays bit-exact.
        r = np.clip(r, -r_clip, r_clip)
    m = np.exp(m).flatten()
    r = np.exp(r).flatten()
    return nbinom.rvs(r, r / (m + r))


def nonlinear_latent(lin, Z, alpha, form, rng, intercept=True, winsor=4.0):
    """Return the mixed latent contribution and its diagnostics.

    `lin` is (n, p) = Z @ Um, the published linear latent term. The returned array has the
    same per-gene mean and standard deviation for every alpha, so only the form changes.
    """
    n, p = lin.shape
    info = {"form": form, "alpha": float(alpha)}
    sd0 = lin.std(axis=0)
    info["latent_sd_before"] = float(sd0.mean())
    if alpha <= 0:
        info["latent_sd_after"] = info["latent_sd_before"]
        info["nl_corr_with_linear"] = 0.0
        return lin, info                      # bit-exact published generator

    Zl = Z[:, 1:] if intercept else Z          # drop the constant column
    if Zl.shape[1] < 2 and form == "interaction":
        raise ValueError("interaction confounding needs at least two latent factors")

    if form == "saturating":
        nl = np.tanh(lin / np.where(sd0 > 0, sd0, 1.0))
    elif form == "cubic":
        # Third probabilists' Hermite polynomial, He3(t) = t^3 - 3t, scaled to unit variance
        # (Var He3 = 6 for standard normal t). The -3t is what makes it ORTHOGONAL to the
        # linear index: E[(t^3 - 3t) t] = E t^4 - 3 E t^2 = 3 - 3 = 0, so every bit of what
        # alpha mixes in is new rather than a rescaling of the term already there. A raw cube
        # is 0.775 correlated with the index, so most of it would just be the linear term
        # again -- the same objection that rules tanh out of the dispersion channel.
        #
        # The price of orthogonality is monotonicity: any zero-mean cubic orthogonal to t is
        # a multiple of He3, whose derivative 3t^2 - 3 is negative on |t| < 1, so it DECREASES
        # through the bulk and increases in the tails. That is a genuinely different shape
        # from the monotone, saturating tanh used for the heterogeneity, which is the point
        # of pairing them.
        #
        # WINSORISED. He3 has excess kurtosis about 86, and the rescale in _equation shifts
        # a whole gene's predictor by (max(m) - cap) whenever the cap is exceeded -- a
        # column-wide shift driven by its single most extreme cell. Unwinsorised, one cell
        # in the far tail drags that gene's entire expression level down, so the arm would
        # fail mechanically rather than statistically. The bound is symmetric, so it leaves
        # the function odd and its orthogonality to the linear index essentially intact.
        t = lin / np.where(sd0 > 0, sd0, 1.0)
        nl = (t ** 3 - 3.0 * t) / np.sqrt(6.0)
        if winsor:
            info["nl_frac_winsorised"] = float(np.mean(np.abs(nl) > winsor))
            nl = np.clip(nl, -winsor, winsor)
    elif form == "interaction":
        d = Zl.shape[1]
        ia, ib = np.triu_indices(d, k=1)
        prods = Zl[:, ia] * Zl[:, ib]                      # (n, d(d-1)/2)
        C = rng.uniform(-1.0, 1.0, size=(prods.shape[1], p))
        nl = prods @ C
    else:
        raise ValueError(f"unknown form {form!r}")

    # correlation of the nonlinear part with the linear index, averaged over genes: near 1
    # for a monotone form, near 0 for the interaction form. This is the quantity that says
    # how much of the confounding a linear adjustment could reach.
    cors = []
    for j in range(p):
        if lin[:, j].std() > 0 and nl[:, j].std() > 0:
            cors.append(abs(np.corrcoef(lin[:, j], nl[:, j])[0, 1]))
    info["nl_corr_with_linear"] = float(np.mean(cors)) if cors else 0.0

    mixed = (1.0 - alpha) * lin + alpha * nl
    # restore each gene's mean and standard deviation, so alpha changes form only
    mu0, mu1 = lin.mean(axis=0), mixed.mean(axis=0)
    sd1 = mixed.std(axis=0)
    scale = np.where(sd1 > 0, sd0 / np.where(sd1 > 0, sd1, 1.0), 1.0)
    mixed = (mixed - mu1) * scale + mu0
    info["latent_sd_after"] = float(mixed.std(axis=0).mean())
    info["latent_excess_kurtosis"] = float(
        np.mean(((mixed - mixed.mean(0)) / mixed.std(0)) ** 4) - 3.0)
    return mixed, info


def bounded_state(lin, intercept=True):
    """s_ij = tanh(z_i' u_j / sd_i(z_i' u_j)), bounded in (-1, 1), one value per cell and gene.

    The per-gene standardisation sits INSIDE the tanh so every gene's state has the same
    marginal distribution; without it gamma would mean different things for different genes
    and at different numbers of latent factors, since ||u_j|| grows as sqrt(d/3).

    tanh rather than a clip: at unit input SD a clip pins about 32% of cells to exactly +-1,
    which makes a third of them share one gain and stops the state being continuous. tanh
    leaves under 1% near the bound and is strictly monotone, so it preserves the ordering of
    the latent index exactly.
    """
    sd = lin.std(axis=0)
    return np.tanh(lin / np.where(sd > 0, sd, 1.0))


def simulate_NB_nonlinear(W, theta_m, theta_r, n, intercept=True, g=lambda x: x, d=0,
                          shift=1.0, coef=1.0, covar=1.0, intervene_prob=0.9,
                          hard_intervention=True, alpha=0.0, form="saturating",
                          gamma=0.0, lam=0.0, r_clip=8.0, nl_form="interaction",
                          winsor=4.0, frac_confounded=None, conf_scale=1.0,
                          scale_mode="total", aux_rng=None):
    """simulate_NB_tgt with the latent contribution passed through a nonlinearity.

    alpha = 0 reproduces the published simulate_NB_tgt construction: the mixing branch is
    skipped entirely and the auxiliary RNG is never drawn from, so the main NumPy stream is
    untouched and the auxiliary seed makes no difference.

    It is NOT bit-identical to UNPATCHED output, and cannot be. The cap and the floor are
    two sequential shifts and the second can undo the first, after which exp(m) leaves the
    integer range; _equation clips elementwise at the cap to prevent that. Measured at
    p = 20, n = 3000 the clip binds on 14 of 20 genes and unpatched counts reach 1.1e7, so
    most genes differ. Setting APPLY_CLIP = False restores bit-identity with simulate_NB_tgt
    and the selftest checks exactly that, which is what establishes that the clip is the
    only difference. Do not mix patched and unpatched replicates in one figure.
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

    # SPARSE CONFOUNDING. Only a fraction of genes carries latent loadings; the rest are
    # unconfounded in both the mean and the dispersion, so "confounded" means exactly that.
    # The columns are zeroed AFTER the draws, so the global stream is untouched and
    # frac_confounded = 1 with conf_scale = 1 reproduces the published generator exactly.
    conf_info = {}
    if frac_confounded is not None and frac_confounded < 1.0:
        n_conf = int(round(frac_confounded * p))
        conf = np.sort(aux_rng.choice(p, size=n_conf, replace=False)) if n_conf else \
            np.array([], int)
        keep = np.zeros(p, dtype=bool)
        keep[conf] = True
        # scale_mode='total' holds the TOTAL latent variance over genes fixed, so the sweep
        # varies how CONCENTRATED the confounding is rather than how much there is;
        # 'fixed' leaves the per-gene strength alone, so the sweep varies prevalence and
        # total together. Neither is wrong, but they answer different questions and the
        # realised per-gene and total SDs are reported either way.
        mult = conf_scale * (1.0 / np.sqrt(frac_confounded) if scale_mode == "total" else 1.0)
        col = np.where(keep, mult, 0.0)
        r0 = 1 if intercept else 0          # never touch the intercept row
        Um[r0:, :] = Um[r0:, :] * col
        Ur[r0:, :] = Ur[r0:, :] * col
        conf_info = dict(frac_confounded=float(frac_confounded),
                         n_confounded_genes=int(n_conf),
                         scale_mode=scale_mode, conf_scale=float(conf_scale),
                         per_gene_multiplier=float(mult),
                         # Pairs, not genes, is what the fit sees: only a pair with BOTH
                         # genes confounded carries spurious association, so the affected
                         # share goes as (frac*p)^2 and moves far faster than the gene
                         # fraction. At p=100 the gene fraction 0.1 -> 0.5 is 5-fold while
                         # the pair fraction 0.009 -> 0.247 is 27-fold. Plot against this.
                         frac_confounded_pairs=float(
                             (n_conf * (n_conf - 1)) / max(p * (p - 1), 1)),
                         confounded_genes=conf.tolist())

    covar_m = Z @ Um
    covar_r = Z @ Ur
    if conf_info:
        lat = covar_m if not intercept else covar_m - (Z[:, :1] @ Um[:1, :])
        conf_info["latent_sd_confounded"] = float(
            lat[:, keep].std(axis=0).mean()) if keep.any() else 0.0
        conf_info["latent_sd_unconfounded"] = float(
            lat[:, ~keep].std(axis=0).mean()) if (~keep).any() else 0.0
    if form == "gain":
        # NON-ADDITIVE confounding. s is nonlinear in z (tanh) and enters as a MULTIPLIER on
        # the parental sum, so the confounding term is gamma * s_ij * sum_k w_kj g(x_ik) --
        # bilinear in z and x, and therefore not representable by any additive, residual-
        # linear correction. The additive latent term z'u_j is left in place, so this arm is
        # the published confounding PLUS a non-additive component rather than a substitute.
        # The state is built from the LINEAR index, before any alpha transform, so the two
        # axes stay independent: alpha changes how nonlinear the ADDITIVE confounding is,
        # gamma how much NON-ADDITIVE confounding there is, and neither moves the other.
        S = bounded_state(covar_m, intercept=intercept)
        gain_all = 1.0 + gamma * S                      # bounded in (1 - gamma, 1 + gamma)
        info = dict(form="gain", gamma=float(gamma), lam=float(lam),
                    state_sd=float(S.std(axis=0).mean()),
                    state_frac_at_bound=float(np.mean(np.abs(S) > 0.99)),
                    gain_min=float(gain_all.min()), gain_max=float(gain_all.max()),
                    gain_sd=float(gain_all.std(axis=0).mean()),
                    gain_frac_nonpositive=float(np.mean(gain_all <= 0)),
                    # Genes with no parents have no parental sum for the gain to modulate,
                    # so for them the non-additive term is identically zero and only the
                    # additive latent confounding acts. This count bounds how much of the
                    # graph gamma can affect and must be reported rather than assumed.
                    n_root_genes=int(((W != 0).sum(axis=0) == 0).sum()),
                    frac_root_genes=float(((W != 0).sum(axis=0) == 0).mean()))
        if alpha:
            # Nonlinearity of the ADDITIVE latent term, the second axis. nonlinear_latent
            # holds each gene's mean and SD fixed, so alpha moves form and not magnitude.
            covar_m, nl_info = nonlinear_latent(covar_m, Z, alpha, nl_form, aux_rng,
                                                intercept=intercept, winsor=winsor)
            info.update({f"additive_{k}": v for k, v in nl_info.items()})
            info["additive_form"] = nl_form
        if lam:
            # NONLINEAR SCALE confounding. The published generator already has a LINEAR
            # dispersion dependence on z through Ur, so a further linear term would add
            # nothing in kind; s^2 - 1 is centred, so mean dispersion is unchanged and only
            # its dependence on the latent state becomes nonlinear.
            # s^2 - 1 rather than tanh(s): tanh is ~93% correlated with the linear index
            # (measured on this generator), so lam*tanh(s) would mostly rescale the linear
            # dispersion term the published generator ALREADY has through Ur, adding little
            # in kind. The centred square is uncorrelated with that index, so all of it is
            # new. Non-additivity is moot here -- r_lin carries no control-function term at
            # all, so ANY dispersion confounding is unreachable; only its form matters.
            Sr = bounded_state(covar_r, intercept=intercept)
            covar_r = covar_r + lam * (Sr ** 2 - np.mean(Sr ** 2, axis=0))
            info["dispersion_sd_before"] = float((Z @ Ur).std(axis=0).mean())
            info["dispersion_sd_after"] = float(covar_r.std(axis=0).mean())
    else:
        gain_all = None
        covar_m, info = nonlinear_latent(covar_m, Z, alpha, form, aux_rng,
                                         intercept=intercept, winsor=winsor)

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
                            gain=None if gain_all is None else gain_all[:, j],
                            r_clip=r_clip if lam else None).reshape(-1)

    info.update(conf_info)
    return X, intervene_mask, Z, Um, Ur, Vm, Vr, info


def validate_counts(X, where):
    """Fail at generation, naming the offending genes, rather than hours later in the fit."""
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


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


SWEEP_TAG = {"frac": "conf", "gamma": "gain", "alpha": "nl"}


def stem_for(a, value, replicate):
    return (f"n{a.n // 1000}k_p{a.p}_s{a.s0}_{a.graph}_um{a.d}_"
            f"{SWEEP_TAG[a.sweep]}{int(round(value * 100)):03d}_repl{replicate}")


def generate(a, replicate):
    os.makedirs(a.datadir, exist_ok=True)
    rows = []
    for value in a.sweep_values:
        # Exactly one quantity moves across the sweep; the other two are held at their
        # scalar values, so any difference between settings is attributable to it alone.
        frac = value if a.sweep == "frac" else a.frac_confounded
        gamma = value if a.sweep == "gamma" else a.gamma
        alpha = value if a.sweep == "alpha" else a.alpha
        # One graph and one data-RNG start per replicate, shared by every alpha, so the
        # settings are PAIRED within a replicate exactly as the control-function comparison
        # is: any difference between alphas is the confounding form and nothing else.
        set_random_seed(100000 + replicate)
        B = simulate_dag(a.p, a.s0, a.graph)
        W = simulate_parameter(B, w_ranges=((-1, -0.5), (0.5, 1.0)))
        X, mask, Z, Um, Ur, Vm, Vr, info = simulate_NB_nonlinear(
            W=W, theta_m=-np.ones(a.p), theta_r=np.ones(a.p), n=a.n, d=a.d,
            intercept=True, g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
            hard_intervention=(a.intervention == "hard"),
            alpha=alpha, gamma=gamma if a.form == "gain" else 0.0,
            lam=a.lam, r_clip=a.r_clip, form=a.form, nl_form=a.nl_form,
            winsor=a.winsor, frac_confounded=frac,
            conf_scale=a.conf_scale, scale_mode=a.scale_mode,
            aux_rng=np.random.default_rng(80000 + replicate))
        validate_counts(X, f"{a.form} {a.sweep}={value} replicate={replicate}")
        stem = stem_for(a, value, replicate)
        np.savez(os.path.join(a.datadir, stem + ".npz"),
                 X=X, W=W, mask=mask, Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr,
                 info=json.dumps(info))
        info.update(replicate=replicate, mean_count=float(X.mean()),
                    frac_zero=float((X == 0).mean()))
        rows.append(info)
        if a.sweep == "frac":
            print(f"[gen] {stem}  confounded genes={info.get('n_confounded_genes')} "
                  f"pairs={info.get('frac_confounded_pairs', 0):.3f} "
                  f"latent_sd conf={info.get('latent_sd_confounded', 0):.3f} "
                  f"unconf={info.get('latent_sd_unconfounded', 0):.3f} "
                  f"mult={info.get('per_gene_multiplier', 1):.2f} "
                  f"mean_count={info['mean_count']:.2f}", flush=True)
        elif a.form == "gain":
            print(f"[gen] {stem}  gain in [{info['gain_min']:.3f}, {info['gain_max']:.3f}] "
                  f"sd={info['gain_sd']:.3f} nonpositive={info['gain_frac_nonpositive']:.4f} "
                  f"state_sd={info['state_sd']:.3f} "
                  f"at_bound={info['state_frac_at_bound']:.4f} "
                  f"root_genes={info['n_root_genes']} "
                  f"mean_count={info['mean_count']:.2f}", flush=True)
        else:
            print(f"[gen] {stem}  sd_before={info['latent_sd_before']:.4f} "
                  f"sd_after={info['latent_sd_after']:.4f} "
                  f"corr_with_linear={info['nl_corr_with_linear']:.3f} "
                  f"mean_count={info['mean_count']:.2f}", flush=True)
    with open(os.path.join(a.datadir, f"gen_{a.form}_repl{replicate}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def selftest():
    """alpha = 0 must reproduce the published generator bit-for-bit, and the latent mean and
    standard deviation must be preserved for every alpha and both forms."""
    p_, n_, d_ = 20, 3000, 6
    common = dict(theta_m=-np.ones(p_), theta_r=np.ones(p_), n=n_, d=d_, intercept=True,
                  g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
                  hard_intervention=False)

    def run(alpha, form, seed=11, gamma=0.0, lam=0.0):
        set_random_seed(seed)
        B = simulate_dag(p_, 3 * p_, "ER")
        W = simulate_parameter(B, w_ranges=((-1, -0.5), (0.5, 1.0)))
        return simulate_NB_nonlinear(W=W, alpha=alpha, form=form, gamma=gamma, lam=lam,
                                     aux_rng=np.random.default_rng(7), **common)

    # (a) with the overflow clip disabled, alpha = 0 must equal the PUBLISHED generator
    # bit-for-bit. This is the check that the clip is the only difference between them.
    from sim_base import simulate_NB_tgt
    global APPLY_CLIP
    APPLY_CLIP = False
    set_random_seed(11)
    Bp = simulate_dag(p_, 3 * p_, "ER")
    Wp = simulate_parameter(Bp, w_ranges=((-1, -0.5), (0.5, 1.0)))
    Xpub = simulate_NB_tgt(W=Wp, **common)[0]
    Xmine = run(0.0, "saturating")[0]
    same = np.array_equal(Xpub, Xmine)
    APPLY_CLIP = True
    assert same, "alpha=0 differs from simulate_NB_tgt for reasons other than the clip"
    Xclip = run(0.0, "saturating")[0]
    nbad = int((np.abs(Xpub - Xclip).sum(axis=0) > 0).sum())
    print(f"[selftest] alpha=0 equals the published simulate_NB_tgt bit-for-bit with the "
          f"overflow clip disabled; with it enabled {nbad} of {p_} genes differ and "
          f"unpatched counts reach {Xpub.max():.3g}")

    # (b) the two forms must agree with each other at alpha = 0
    base = Xclip
    for form in ("saturating", "interaction"):
        assert np.array_equal(base, run(0.0, form)[0]), f"alpha=0 not bit-exact for {form}"

    for form in ("saturating", "interaction"):
        for alpha in (0.25, 0.5, 1.0):
            X, _, Z, Um, _, _, _, info = run(alpha, form)
            rel = abs(info["latent_sd_after"] - info["latent_sd_before"]) / info["latent_sd_before"]
            assert rel < 1e-9, f"{form} alpha={alpha}: latent sd moved by {rel:.2e}"
            validate_counts(X, f"selftest {form} {alpha}")
        X1, _, _, _, _, _, _, i1 = run(1.0, form)
        print(f"[selftest] {form:<12} alpha=1: latent sd fixed to {rel:.1e} relative, "
              f"|corr| with the linear index {i1['nl_corr_with_linear']:.3f}, "
              f"excess kurtosis {i1.get('latent_excess_kurtosis', 0):+.2f}, "
              f"mean count {X1.mean():.2f}")
    # --- form = gain: non-additive confounding -------------------------------------
    Xg0 = run(0.0, "gain", gamma=0.0)[0]
    assert np.array_equal(Xclip, Xg0), "gamma=0 does not reproduce the published generator"
    print("[selftest] gamma=0 reproduces the published generator bit-for-bit")

    print(f"[selftest] {'gamma':>6} {'gain range':>18} {'gain sd':>8} {'nonpos':>7} "
          f"{'state sd':>9} {'at bound':>9} {'mean count':>11}")
    for gm in (0.2, 0.4, 0.6, 0.8):
        Xg, _, _, _, _, _, _, ig = run(0.0, "gain", gamma=gm)
        # the bound is what makes the gain strictly positive for gamma < 1
        assert ig["gain_min"] > 1.0 - gm - 1e-9 and ig["gain_max"] < 1.0 + gm + 1e-9, \
            "gain left the interval the bounded state guarantees"
        assert ig["gain_frac_nonpositive"] == 0.0, "gain went non-positive below gamma=1"
        validate_counts(Xg, f"selftest gain gamma={gm}")
        print(f"[selftest] {gm:>6.1f} "
              f"[{ig['gain_min']:.3f}, {ig['gain_max']:.3f}]".rjust(19) +
              f" {ig['gain_sd']:>8.3f} {ig['gain_frac_nonpositive']:>7.4f} "
              f"{ig['state_sd']:>9.3f} {ig['state_frac_at_bound']:>9.4f} "
              f"{Xg.mean():>11.2f}")

    # dispersion arm: centred, so mean dispersion is unchanged, and r is bounded
    Xl, _, _, _, Url, _, _, il = run(0.0, "gain", gamma=0.4, lam=0.5)
    validate_counts(Xl, "selftest gain+dispersion")
    print(f"[selftest] lam=0.5: dispersion sd {il['dispersion_sd_before']:.3f} -> "
          f"{il['dispersion_sd_after']:.3f}, mean count {Xl.mean():.2f}")
    print("[selftest] OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--form", choices=["saturating", "interaction", "gain"],
                    default="saturating")
    ap.add_argument("--gammas", default="0,0.2,0.4,0.6,0.8",
                    help="form=gain: half-range of the regulatory gain; 0 is the published "
                         "generator. The gain lies in (1-gamma, 1+gamma) and stays positive "
                         "for gamma < 1, which is what bounding the state buys.")
    ap.add_argument("--sweep", choices=["frac", "gamma", "alpha"], default="frac",
                    help="which quantity varies across the settings in one element; the "
                         "other two are held at their scalar values")
    ap.add_argument("--fracs", default="0,0.1,0.2,0.3,0.4,0.5,1.0",
                    help="sweep=frac: fractions of genes carrying latent loadings. 0 is the "
                         "no-confounding reference and 1.0 the published dense arm; keep "
                         "both, since without them a difference cannot be told from a "
                         "baseline that was never zero.")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="held fixed unless --sweep gamma")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="form=gain: nonlinearity of the ADDITIVE latent term, a separate "
                         "axis from gamma. 0 keeps it linear, 1 makes it fully nonlinear, "
                         "with each gene's mean and SD held fixed either way.")
    ap.add_argument("--nl-form", choices=["cubic", "interaction", "saturating"],
                    default="cubic", dest="nl_form",
                    help="form=gain: which nonlinearity --alpha mixes in")
    ap.add_argument("--frac-confounded", type=float, default=None,
                    dest="frac_confounded",
                    help="fraction of genes carrying latent loadings; the rest are "
                         "unconfounded in mean and dispersion. Omit or set 1.0 for the "
                         "published dense confounding.")
    ap.add_argument("--conf-scale", type=float, default=1.0, dest="conf_scale",
                    help="extra multiplier on the loadings of the confounded genes")
    ap.add_argument("--scale-mode", choices=["fixed", "total"], default="fixed",
                    dest="scale_mode",
                    help="'total' scales loadings by 1/sqrt(frac) so the total latent "
                         "variance over genes is held fixed and the sweep varies "
                         "CONCENTRATION; 'fixed' leaves per-gene strength alone so the "
                         "sweep varies PREVALENCE and total together")
    ap.add_argument("--winsor", type=float, default=4.0,
                    help="bound in SD units on the standardised nonlinear component; 0 "
                         "disables it. The cubic has excess kurtosis about 86 unwinsorised, "
                         "and one far-tail cell shifts a whole gene's predictor via the cap "
                         "in _equation.")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="form=gain: weight on the nonlinear dispersion term lam*(s^2-1); "
                         "0 leaves the published linear dispersion confounding alone")
    ap.add_argument("--r-clip", type=float, default=8.0, dest="r_clip",
                    help="bound on the dispersion predictor; applied only when --lam != 0")
    ap.add_argument("--alphas", default=",".join(str(x) for x in ALPHAS),
                    help="comma-separated severities; 0 is the published linear generator")
    ap.add_argument("--replicate", type=int,
                    default=int(os.environ.get("LSB_JOBINDEX", 1)))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--s0", type=int, default=400)
    ap.add_argument("--graph", default="ER")
    ap.add_argument("--d", type=int, default=10, help="number of latent factors")
    ap.add_argument("--intervention", choices=["soft", "hard"], default="soft")
    ap.add_argument("--datadir",
                    default="result/simu_nonlinear/data/NB")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    src = {"frac": a.fracs, "gamma": a.gammas, "alpha": a.alphas}[a.sweep]
    a.sweep_values = [float(x) for x in src.split(",")]
    if a.sweep != "frac" and a.frac_confounded is None:
        a.frac_confounded = 1.0
    generate(a, a.replicate)


if __name__ == "__main__":
    main()
