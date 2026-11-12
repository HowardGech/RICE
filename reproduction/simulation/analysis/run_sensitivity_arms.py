"""
Sensitivity of RICE to INVALID MATCHED GUIDES and to NONLINEAR CONFOUNDING.

Two mechanisms, each with a severity sweep whose zero point reproduces the published
generator bit-for-bit, so every curve starts from the paper's own setting.

MECHANISM 1 -- invalid matched guides (`--mechanism guides`).
A fraction of guides are invalid: guide D_k acquires a direct effect on genes other than
its target, so the exclusion restriction fails for that guide. This is the assumption the
gene-matched (reduced) control function leans on most directly, since it residualizes
gene j on D_j alone. Severity is the FRACTION of guides that are invalid; each invalid
guide gets the same number of recipients and the same effect magnitude, so the sweep
varies prevalence and not per-guide severity. `--invalid-relevance` additionally scales
the on-target knockdown of the invalid guides, covering the other sense in which a
matched guide can be invalid: it fails relevance as well as exclusion.

MECHANISM 2 -- nonlinear confounding (`--mechanism nonlinear`).
The latent contribution to gene j's linear predictor becomes

    sqrt(1 - alpha^2) * u_j  +  alpha * (u_j^2 - Var u_j) / (sqrt(2) * sd u_j),
    where u_j = z' Um[:, j]

alpha = 0 is the published linear confounding; alpha = 1 is purely quadratic. For
Gaussian u the linear and centred-quadratic parts are UNCORRELATED and the standardized
quadratic has the same standard deviation as u, so this weighting holds the total latent
standard deviation EXACTLY fixed across the sweep. alpha therefore varies the FORM of the
confounding, not its magnitude -- which matters because the control function adjusts for
a linear projection, so only the form determines how much of it can be absorbed.

Both mechanisms are measured against the same three arms as the control-function
comparison: `rcf` (gene-matched residual), `fcf` (full control function), `nocf` (none).

KNOWN CONFOUND IN THE NONLINEAR SWEEP, and how to read around it. alpha holds the latent
standard deviation fixed but NOT the count scale: the published rescale re-centres each
gene by its own maximum, and a skewed latent term of the same standard deviation has a
much larger maximum, so the realised mean count moves with alpha (measured: 25 at
alpha = 0 against 34 at alpha = 0.5 in a p=30 test). A constant shift on the eta scale
cannot repair this -- the cap absorbs constant shifts outright. Consequently the alpha
sweep must be read as a WITHIN-alpha arm comparison: at each alpha all three arms see the
identical dataset, so that contrast is clean, whereas the trend in absolute level across
alpha carries the count-scale change as well as the nonlinearity. `mean_count` and
`sd_count` are recorded per configuration so the size of the confound is visible.

Pairing: every configuration within a replicate is generated from ONE graph and ONE seed,
so arms and severities are compared within replicate. The zero-severity configuration is
always included as the reference -- without it a small difference cannot be told from a
baseline that was never zero.

    python run_sensitivity_arms.py --mechanism guides --stage both --replicate 3
    LSB_JOBINDEX=7 python run_sensitivity_arms.py --mechanism both --stage both
    RICE_BACKEND=numpy RICE_SCHEDULE=reduced python run_sensitivity_arms.py --mechanism nonlinear

Outputs, per replicate: one npz per configuration under --datadir and one tidy CSV under
--resdir with columns config / method / measure / value, plus `settings` rows recording
every knob so a later run cannot be pooled with this one by mistake.
"""
from sim_base import simulate_NB_tgt, is_dag
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

# The inner equation below is lifted out of simulate_NB_tgt verbatim, including its
# 6.5 / 0.5 rescale constants and the elementwise clip fix. Those constants differ from
# the module-level helper's 7.5 / -0.5 in sim_base.py; matching simulate_NB_tgt
# is what keeps these arms on the same count scale as the published NB baseline.

# ======================================================================================
# Generators -- verbatim from sim_base.py and sim_crossguide.py with the
# rescale fix applied. Do not edit in place; re-derive from those files if they change.
# ======================================================================================


def _simulate_single_equation_tgt(X, w, cov_m, cov_r, mask, shift_m, shift_r,
                                  cap=6.5, floor=0.5):
    n = X.shape[0]
    p = w.shape[0]
    cov_m = cov_m.reshape(-1, 1)
    cov_r = cov_r.reshape(-1, 1)
    X = X.reshape(n, -1)
    w = w.reshape(-1, 1)
    Xw = (X @ w).reshape(-1, 1)
    shift_m = shift_m.reshape(-1, 1)
    shift_r = shift_r.reshape(-1, 1)
    mask = mask.reshape(-1, 1)
    m = (Xw + cov_m) * (1 - mask) + shift_m
    r = cov_r * (1 - mask) + shift_r
    if np.max(m) > cap:
        m -= (np.max(m) - cap)
    if np.mean(m) < floor:
        m += (floor - np.mean(m))
    # the mean floor above can undo the cap when one cell is extreme, leaving
    # max(m) far above 7.5 and exp(m) beyond int64, which makes nbinom.rvs
    # return negative counts. Clip elementwise: a no-op unless that happens.
    m = np.minimum(m, cap)
    m = np.exp(m)
    r = np.exp(r)
    m = m.flatten()
    r = r.flatten()
    x = nbinom.rvs(r, r / (m + r))
    return x


# ======================================================================================
# Sensitivity generators. Both are simulate_NB_tgt with one term added, so that the
# zero-severity setting reproduces it exactly -- verified in --stage selftest.
# ======================================================================================


def draw_invalid_guide_matrix(p, frac_invalid, n_recipients, effect, rng, W=None):
    """
    Beta[k, j] = direct effect of guide k on gene j, nonzero only for INVALID guides k.

    Severity is the fraction of guides that are invalid. Each invalid guide gets exactly
    `n_recipients` recipients drawn uniformly from the other genes and effects drawn from
    U(-effect, effect), so per-guide severity is constant across the sweep and only
    prevalence changes. Drawn from its own Generator so that frac_invalid = 0 consumes no
    numbers from the global stream and reproduces the published generator bit-for-bit.
    """
    Beta = np.zeros((p, p))
    n_bad = int(round(frac_invalid * p))
    bad = np.sort(rng.choice(p, size=n_bad, replace=False)) if n_bad else np.array([], int)
    n_on_edge = 0
    for k in bad:
        others = np.setdiff1d(np.arange(p), [k])
        recip = rng.choice(others, size=min(n_recipients, len(others)), replace=False)
        Beta[k, recip] = rng.uniform(-effect, effect, size=len(recip))
        if W is not None:
            n_on_edge += int((W[k, recip] != 0).sum())
    info = dict(frac_invalid=float(frac_invalid), n_invalid_guides=int(n_bad),
                n_recipients=int(n_recipients), invalid_effect=float(effect),
                n_leak=int((Beta != 0).sum()), n_leak_on_true_edge=int(n_on_edge),
                # fraction of genes receiving at least one invalid-guide effect; this is
                # what the fit actually sees, and it saturates well before frac_invalid
                # reaches 1 because recipients are drawn with replacement across guides
                frac_genes_affected=float((np.abs(Beta).sum(axis=0) > 0).mean()))
    return Beta, bad, info


def simulate_NB_sensitivity(W, theta_m, theta_r, n,
                            intercept=True, g=lambda x: x, d=0, shift=1.0,
                            coef=1.0, covar=1.0, intervene_prob=0.9,
                            hard_intervention=True,
                            frac_invalid=0.0, n_recipients=10, invalid_effect=3.0,
                            invalid_relevance=1.0, nonlin_alpha=0.0,
                            nonlin_dispersion=False, aux_rng=None):
    """simulate_NB_tgt with invalid matched guides and/or nonlinear confounding.

    frac_invalid = 0 and nonlin_alpha = 0 reproduces simulate_NB_tgt exactly.
    """
    aux_rng = np.random.default_rng(0) if aux_rng is None else aux_rng
    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == p
    X = np.zeros([n, p])

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

    # ---- MECHANISM 2: make the latent contribution nonlinear at fixed magnitude -------
    # Split off the intercept column: only the genuine latent part is transformed, and
    # theta_m / theta_r must pass through untouched.
    nonlin_info = dict(nonlin_alpha=float(nonlin_alpha))
    if intercept and d > 1:
        # record the latent scale even at alpha = 0, so the sweep has a reference value
        # to be constant against; at alpha = 0 nothing below modifies covar_m
        _lin0 = Z[:, 1:] @ Um[1:]
        nonlin_info['latent_sd_after'] = float(_lin0.std(axis=0).mean())
        nonlin_info['latent_skew'] = 0.0
    if nonlin_alpha > 0:
        if not intercept or d <= 1:
            raise ValueError('nonlinear confounding needs latent factors and an intercept')
        for name, (C, U) in {'m': (covar_m, Um), 'r': (covar_r, Ur)}.items():
            if name == 'r' and not nonlin_dispersion:
                continue
            base = Z[:, :1] @ U[:1]                  # theta term, left alone
            lin = Z[:, 1:] @ U[1:]                   # zero-mean Gaussian per gene
            sd = lin.std(axis=0, keepdims=True)
            sd = np.where(sd > 0, sd, 1.0)
            quad = (lin ** 2 - lin.var(axis=0, keepdims=True)) / (np.sqrt(2.0) * sd)
            mixed = np.sqrt(1.0 - nonlin_alpha ** 2) * lin + nonlin_alpha * quad
            C[:] = base + mixed
            if name == 'm':
                nonlin_info['latent_sd_mean'] = float(sd.mean())
                nonlin_info['latent_sd_after'] = float(mixed.std(axis=0).mean())
                # |skew| of the latent term: 0 when linear, ~2*alpha-ish when quadratic.
                # This is the diagnostic that the FORM changed while the scale did not.
                z0 = mixed - mixed.mean(axis=0, keepdims=True)
                nonlin_info['latent_skew'] = float(
                    np.mean((z0 ** 3).mean(axis=0) / (z0.std(axis=0) ** 3 + 1e-12)))

    shift_m = ones @ Vm
    shift_r = ones @ Vr

    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    intervene_mask = np.zeros((n, p), dtype=bool)

    remaining_nodes = intervene_idx
    each_intervention_size = len(intervene_idx) // p
    for i in range(p):
        choice = np.random.choice(remaining_nodes, size=each_intervention_size,
                                  replace=False)
        intervene_mask[choice, i] = True
        remaining_nodes = np.setdiff1d(remaining_nodes, choice)

    shift_m_masked = shift_m * intervene_mask
    shift_r_masked = shift_r * intervene_mask

    # ---- MECHANISM 1: invalid matched guides -----------------------------------------
    Beta, bad_guides, guide_info = draw_invalid_guide_matrix(
        p, frac_invalid, n_recipients, invalid_effect, aux_rng, W=W)
    # a direct effect of guide k on gene j, carried by the recorded perturbation column
    leak_all = intervene_mask.astype(float) @ Beta
    if frac_invalid > 0 and invalid_relevance != 1.0:
        # the same guides also under-perform on their own target: scale their on-target
        # shift. relevance = 1 leaves exclusion-only invalidity, relevance = 0 removes
        # the knockdown entirely while keeping the off-target effect.
        scale = np.ones((1, p))
        scale[0, bad_guides] = invalid_relevance
        shift_m_masked = shift_m_masked * scale
        shift_r_masked = shift_r_masked * scale
    guide_info['invalid_relevance'] = float(invalid_relevance)

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        mask_j = intervene_mask[:, j] if hard_intervention else np.zeros(n)
        X[:, j] = _simulate_single_equation_tgt(
            g(X[:, parents]), W[parents, j],
            covar_m[:, j] + leak_all[:, j],          # the only change for mechanism 1
            covar_r[:, j], mask_j,
            shift_m_masked[:, j], shift_r_masked[:, j]).reshape(-1)

    info = dict(guide_info)
    info.update(nonlin_info)
    # magnitude of the invalid-guide contribution relative to the designed perturbation
    # shift, both on the eta scale, so the two mechanisms can be put on one axis
    info['leak_to_signal'] = float(np.abs(leak_all).mean() / (np.abs(Vm).mean() + 1e-12))
    # The alpha sweep MOVES THE COUNT SCALE and this cannot be corrected on the eta
    # scale: the published rescale re-centres each gene by its own maximum, which absorbs
    # any constant shift, while a skewed latent term of the same standard deviation has a
    # far larger maximum (17.2 vs 5.2 at matched sd in a direct test) and so is re-centred
    # differently. Record the realised scale so the confound travels with the results.
    info['mean_count'] = float(X.mean())
    info['sd_count'] = float(X.std())
    return X, intervene_mask, Z, Um, Ur, Vm, Vr, Beta, info


# ======================================================================================
# Configurations
# ======================================================================================

GUIDE_FRACS   = [0.0, 0.05, 0.10, 0.20, 0.40]
NONLIN_ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.00]
ARMS = ["rcf", "fcf", "nocf"]


def configs_for(mechanism):
    """Configuration names. The zero-severity point is first and always included."""
    cfgs = []
    if mechanism in ("guides", "both"):
        cfgs += [f"guides_{int(round(f * 100)):03d}" for f in GUIDE_FRACS]
    if mechanism in ("nonlinear", "both"):
        cfgs += [f"nonlin_{int(round(al * 100)):03d}" for al in NONLIN_ALPHAS]
    out = []
    for c in cfgs:
        # 'both' would otherwise carry two identical zero-severity points
        if c.endswith("_000") and any(o.endswith("_000") for o in out):
            continue
        out.append(c)
    return out


def knobs_for(cfg, a):
    """Map a configuration name to generator knobs."""
    kind, pct = cfg.rsplit("_", 1)
    v = int(pct) / 100.0
    if kind == "guides":
        return dict(frac_invalid=v, n_recipients=a.n_recipients,
                    invalid_effect=a.invalid_effect,
                    invalid_relevance=a.invalid_relevance, nonlin_alpha=0.0)
    if kind == "nonlin":
        return dict(frac_invalid=0.0, n_recipients=a.n_recipients,
                    invalid_effect=a.invalid_effect, invalid_relevance=1.0,
                    nonlin_alpha=v)
    raise ValueError(f"unknown configuration {cfg}")


def validate_counts(X, cfg, replicate, max_plausible=1e7):
    """Fail at generation, naming the genes, rather than at fitting with a NaN."""
    bad = []
    if (X < 0).any():
        bad.append(f"negative counts in genes {np.unique(np.where(X < 0)[1])[:10].tolist()}")
    if not np.isfinite(X).all():
        bad.append(f"non-finite in genes {np.unique(np.where(~np.isfinite(X))[1])[:10].tolist()}")
    if X.max() > max_plausible:
        hi = np.where(X.max(axis=0) > max_plausible)[0]
        bad.append(f"counts above {max_plausible:.0g} in genes {hi[:10].tolist()} "
                   f"(max {X.max():.3g})")
    if bad:
        raise ValueError(f"[{cfg} repl {replicate}] " + "; ".join(bad))


def stem_for(a, cfg, replicate):
    return (f"n{a.n // 1000}k_p{a.p}_s{a.s0}_{a.graph}_um{a.d}_{cfg}"
            f"_{a.model}_repl{replicate}")


# ======================================================================================
# Stages
# ======================================================================================

def generate(a, replicate):
    os.makedirs(a.datadir, exist_ok=True)
    # one graph and one seed per replicate: every configuration below is a perturbation
    # of the SAME network and the same draw, which is what licenses a paired comparison
    set_random_seed(replicate)
    B = simulate_dag(a.p, a.s0, a.graph)
    W = simulate_parameter(B, w_ranges=((-1.0, -0.5), (0.5, 1.0)))

    for cfg in a.configs:
        k = knobs_for(cfg, a)
        set_random_seed(replicate)              # identical draw across configurations
        out = simulate_NB_sensitivity(
            W=W, theta_m=-1 * np.ones(a.p), theta_r=1 * np.ones(a.p), n=a.n,
            d=a.d, intercept=True, g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
            hard_intervention=(a.itype == "hard"),
            aux_rng=np.random.default_rng(70000 + replicate), **k)
        X, mask, Z, Um, Ur, Vm, Vr, Beta, info = out
        validate_counts(X, cfg, replicate)
        stem = stem_for(a, cfg, replicate)
        np.savez(os.path.join(a.datadir, stem + ".npz"),
                 X=X, W=W, mask=mask, Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr, Beta=Beta,
                 **{f"info_{kk}": vv for kk, vv in info.items()})
        print(f"[gen] repl {replicate} {cfg:<12} mean {X.mean():8.2f}  "
              f"invalid {info['n_invalid_guides']:>3} guides, "
              f"{info['frac_genes_affected'] * 100:4.0f}% genes affected  "
              f"latent_sd {info.get('latent_sd_after', float('nan')):.3f}  "
              f"skew {info.get('latent_skew', 0.0):+.2f}", flush=True)


def fit(a, replicate):
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
        validate_counts(X, cfg, replicate)      # in case of a stale file
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
            for kk, vv in acc.items():
                rows.append({"config": cfg, "method": arm, "measure": kk,
                             "value": float(vv)})
            print(f"[fit] repl {replicate} {cfg:<12} {arm:<5} mcc={acc['mcc']:.3f} "
                  f"shd={acc['shd']:.0f} {secs:.0f}s", flush=True)

        # realised severity, straight from the npz, so the x axis is measured not nominal
        for kk in z.files:
            if kk.startswith("info_"):
                rows.append({"config": cfg, "method": "truth",
                             "measure": kk[5:], "value": float(z[kk])})

    for kk, vv in [("n", a.n), ("p", a.p), ("s0", a.s0), ("d", a.d), ("pen1", a.pen1),
                   ("w_threshold", a.w_threshold), ("n_recipients", a.n_recipients),
                   ("invalid_effect", a.invalid_effect),
                   ("invalid_relevance", a.invalid_relevance),
                   ("nonlin_dispersion", float(a.nonlin_dispersion)),
                   ("itype_is_hard", float(a.itype == "hard")),
                   ("backend_is_cupy", float(a.backend == "cupy")),
                   ("schedule_is_production", float(a.schedule == "production")),
                   ("max_iter", float(sched["max_iter"])),
                   ("epochs", float(sched["epochs"])), ("lr", float(sched["lr"]))]:
        rows.append({"config": "settings", "method": "settings", "measure": kk,
                     "value": float(vv)})

    out = os.path.join(a.resdir, f"{stem_for(a, a.mechanism, replicate)}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[fit] wrote {out}", flush=True)


def selftest(a):
    """The zero-severity reduction, which every curve's reference point depends on."""
    set_random_seed(1)
    B = simulate_dag(20, 60, "ER")
    W = simulate_parameter(B, w_ranges=((-1.0, -0.5), (0.5, 1.0)))
    common = dict(W=W, theta_m=-np.ones(20), theta_r=np.ones(20), n=800, d=5,
                  intercept=True, g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1)
    for itype in ("soft", "hard"):
        set_random_seed(7)
        ref = simulate_NB_tgt(apply_clip=True, hard_intervention=(itype == "hard"), **common)[0]
        set_random_seed(7)
        got = simulate_NB_sensitivity(hard_intervention=(itype == "hard"),
                                      frac_invalid=0.0, nonlin_alpha=0.0,
                                      aux_rng=np.random.default_rng(99), **common)[0]
        assert np.array_equal(ref, got), f"zero-severity differs from published ({itype})"
        print(f"[selftest] {itype}: bit-identical to simulate_NB_tgt")
    for al in (0.0, 0.5, 1.0):
        set_random_seed(7)
        info = simulate_NB_sensitivity(hard_intervention=False, frac_invalid=0.0,
                                       nonlin_alpha=al,
                                       aux_rng=np.random.default_rng(99), **common)[8]
        print(f"[selftest] alpha={al:.2f}  latent_sd={info.get('latent_sd_after', 0):.4f}"
              f"  skew={info.get('latent_skew', 0.0):+.3f}")
    print("[selftest] latent_sd must be constant across alpha; skew must grow")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mechanism", choices=["guides", "nonlinear", "both"],
                    default="both")
    ap.add_argument("--stage", choices=["gen", "fit", "both", "selftest"], default="both")
    ap.add_argument("--replicate", type=int,
                    default=int(os.environ.get("LSB_JOBINDEX", 1)),
                    help="replicate index; defaults to $LSB_JOBINDEX")
    ap.add_argument("--model", default="NB")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--s0", type=int, default=400)
    ap.add_argument("--graph", default="ER")
    ap.add_argument("--d", type=int, default=10, help="number of latent factors")
    ap.add_argument("--itype", choices=["soft", "hard"], default="soft")
    ap.add_argument("--n-recipients", type=int, default=10, dest="n_recipients",
                    help="genes each invalid guide directly affects")
    ap.add_argument("--invalid-effect", type=float, default=3.0, dest="invalid_effect",
                    help="beta ~ U(-x, x) for an invalid guide; 3.0 matches the "
                         "hidden-mediator arm")
    ap.add_argument("--invalid-relevance", type=float, default=1.0,
                    dest="invalid_relevance",
                    help="on-target shift multiplier for invalid guides; 1.0 = "
                         "exclusion-only invalidity, 0.0 = no knockdown either")
    ap.add_argument("--nonlin-dispersion", action="store_true", dest="nonlin_dispersion",
                    help="apply the nonlinearity to the dispersion as well as the mean")
    ap.add_argument("--pen1", type=float, default=1e-4,
                    help="1e-4 is job_simu.py and the Methods value; the other scenario "
                         "drivers use 1e-3")
    ap.add_argument("--w-threshold", type=float, default=0.3, dest="w_threshold")
    ap.add_argument("--backend", choices=["cupy", "numpy"],
                    default=os.environ.get("RICE_BACKEND", "cupy"))
    ap.add_argument("--schedule", choices=["production", "reduced"],
                    default=os.environ.get("RICE_SCHEDULE", "production"))
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset; defaults to the mechanism's sweep")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--datadir",
                    default="data/simu_sens")
    ap.add_argument("--resdir",
                    default="result/simu_sens/scGRAPH/NB")
    a = ap.parse_args()
    a.configs = (a.configs.split(",") if a.configs else configs_for(a.mechanism))
    a.arms = a.arms.split(",")

    if a.stage == "selftest":
        selftest(a)
        return
    print(f"[run] mechanism={a.mechanism} replicate={a.replicate} "
          f"configs={a.configs} arms={a.arms} backend={a.backend} "
          f"schedule={a.schedule}", flush=True)
    if a.stage in ("gen", "both"):
        generate(a, a.replicate)
    if a.stage in ("fit", "both"):
        fit(a, a.replicate)


if __name__ == "__main__":
    main()
