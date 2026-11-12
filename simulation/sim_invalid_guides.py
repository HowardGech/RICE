#!/usr/bin/env python
"""
Generate data with INVALID MATCHED GUIDES at a sweep of prevalences.

A fraction of guides is invalid: guide k acquires a direct effect on genes other than its
intended target, so the exclusion restriction fails for that guide. For gene j the mean's
linear predictor gains

    sum_k Beta[k, j] * D_ik,     Beta[k, j] != 0 only for invalid guides k,

with each invalid guide given exactly `n_recipients` recipients drawn uniformly from the
other genes and effects drawn from U(-invalid_effect, invalid_effect). Per-guide severity is
therefore CONSTANT across the sweep and only prevalence changes, which is what makes the
five points comparable: a sweep in which the guides also got stronger could not say which
of the two moved the result.

This is the assumption the gene-matched (reduced) control function leans on most directly,
since it residualizes gene j on D_j alone and cannot reach an effect carried by D_k.

SETTINGS. Everything other than frac_invalid follows the published configuration, matching
the driver that reads these files: n, p, s0, graph_type, intervention and the number of
latent factors are positional arguments in that driver's order, weights are drawn from
U([-1,-0.5] u [0.5,1]), g = log(1+x), shift = 2, coef = covar = 1, intervene_prob = 0.9,
theta_m = -1, theta_r = +1, and the seed is the array index, as in that driver.

    python sim_invalid_guides.py NB 10000 100 400 ER soft 0
    python sim_invalid_guides.py NB 10000 100 400 ER soft 0 --fracs 0.1,0.3,0.5
    python sim_invalid_guides.py --selftest

Writes one .npz per (frac_invalid, replicate) named

    n{n/1000}k_p{p}_s{s0}_{graph}_cova{um}_inv{100*frac:03d}_repl{job}.npz

with keys X, W, mask, Z, Um, Ur, Vm, Vr, Beta, so the fit driver and the baseline wrappers
read it after a filename change only. Note for any parser: the `inv###` tag sits between the
factor count and `_repl`, so match the setting and the replicate SEPARATELY -- a pattern
requiring `_repl` immediately after the `cova` digits will not match these names.
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


def _equation(X, w, cov_m, cov_r, mask, shift_m, shift_r, cap=6.5, floor=0.5):
    """Inner equation of simulate_NB_tgt, verbatim, including its 6.5 / 0.5 constants.

    Those differ from the module-level helper's 7.5 / -0.5 in sim_base.py;
    matching simulate_NB_tgt keeps these arms on the count scale of the published baseline.
    The elementwise clip is the overflow fix: the cap and the floor are two sequential
    shifts and the second can undo the first, after which exp(m) can leave the integer
    range. It only ever binds where the unpatched code would have overflowed.
    """
    n = X.shape[0]
    cov_m = cov_m.reshape(-1, 1); cov_r = cov_r.reshape(-1, 1)
    X = X.reshape(n, -1); w = w.reshape(-1, 1)
    Xw = (X @ w).reshape(-1, 1)
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


def draw_invalid_guide_matrix(p, frac_invalid, n_recipients, effect, rng, W=None):
    """Beta[k, j] = direct effect of guide k on gene j, nonzero only for invalid guides k.

    Drawn from its own Generator, so frac_invalid = 0 consumes nothing from the global
    stream and reproduces the published generator.
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
                # Fraction of genes receiving at least one invalid-guide effect. This is
                # what the fit actually sees; it saturates well before frac_invalid reaches
                # 1 because recipients are drawn independently across guides.
                frac_genes_affected=float((np.abs(Beta).sum(axis=0) > 0).mean()))
    return Beta, bad, info


def simulate_NB_invalid(W, theta_m, theta_r, n, intercept=True, g=lambda x: x, d=0,
                        shift=1.0, coef=1.0, covar=1.0, intervene_prob=0.9,
                        hard_intervention=True, frac_invalid=0.0, n_recipients=10,
                        invalid_effect=3.0, aux_rng=None):
    """simulate_NB_tgt with a fraction of the guides made invalid.

    frac_invalid = 0 reproduces the published simulate_NB_tgt construction exactly: the
    leak branch is skipped and the auxiliary stream is never drawn from. It is bit-identical
    to unpatched output only where the overflow clip does not bind; see `_equation`.
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

    covar_m = Z @ Um
    covar_r = Z @ Ur
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

    if frac_invalid > 0:
        Beta, bad, info = draw_invalid_guide_matrix(
            p, frac_invalid, n_recipients, invalid_effect, aux_rng, W=W)
        leak_all = intervene_mask.astype(float) @ Beta
    else:
        Beta = np.zeros((p, p)); bad = np.array([], int)
        info = dict(frac_invalid=0.0, n_invalid_guides=0, n_recipients=int(n_recipients),
                    invalid_effect=float(invalid_effect), n_leak=0,
                    n_leak_on_true_edge=0, frac_genes_affected=0.0)
        leak_all = np.zeros((n, p))

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        mask_j = intervene_mask[:, j] if hard_intervention else np.zeros(n)
        X[:, j] = _equation(g(X[:, parents]), W[parents, j],
                            covar_m[:, j] + leak_all[:, j],       # the only change
                            covar_r[:, j], mask_j,
                            shift_m_masked[:, j], shift_r_masked[:, j]).reshape(-1)

    # magnitude of the invalid-guide contribution against the designed perturbation shift,
    # both on the linear-predictor scale, so the sweep can be read in units of the signal
    # Two severity readings, because they answer different questions. leak_to_signal
    # averages over ALL cells and is therefore diluted by the low multiplicity of
    # infection: a cell carries one guide, so only about frac_invalid * n_recipients / p of
    # the (cell, gene) entries are affected at all. mean_abs_effect_where_present is the
    # magnitude WHERE IT OCCURS, which is the number to compare against the perturbation
    # shift when judging whether the violation is strong enough to matter.
    nz = leak_all != 0
    info["leak_to_signal"] = float(np.abs(leak_all).mean() / (np.abs(Vm).mean() + 1e-12))
    info["frac_entries_affected"] = float(nz.mean())
    info["mean_abs_effect_where_present"] = (float(np.abs(leak_all[nz]).mean())
                                             if nz.any() else 0.0)
    info["mean_abs_perturbation_shift"] = float(np.abs(Vm).mean())
    info["invalid_guides"] = bad.tolist()
    return X, intervene_mask, Z, Um, Ur, Vm, Vr, Beta, info


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


def stem_for(a, frac, replicate):
    return (f"n{a.n // 1000}k_p{a.p}_s{a.s0}_{a.graph}_cova{a.um}_"
            f"inv{int(round(frac * 100)):03d}_repl{replicate}")


def generate(a, replicate):
    os.makedirs(a.datadir, exist_ok=True)
    rows = []
    for frac in a.fracs:
        # One graph and one data-RNG start per replicate, shared by every prevalence, so the
        # points are PAIRED within a replicate: any difference between them is the fraction
        # of invalid guides and nothing else. The seed is the array index, as in the fit
        # driver that reads these files.
        set_random_seed(replicate)
        B = simulate_dag(a.p, a.s0, a.graph)
        W = simulate_parameter(B, w_ranges=((-1, -0.5), (0.5, 1.0)))
        X, mask, Z, Um, Ur, Vm, Vr, Beta, info = simulate_NB_invalid(
            W=W, theta_m=-np.ones(a.p), theta_r=np.ones(a.p), n=a.n, d=a.um,
            intercept=True, g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
            intervene_prob=0.9, hard_intervention=(a.intervention == "hard"),
            frac_invalid=frac, n_recipients=a.n_recipients,
            invalid_effect=a.invalid_effect,
            aux_rng=np.random.default_rng(90000 + replicate))
        validate_counts(X, f"frac_invalid={frac} replicate={replicate}")
        stem = stem_for(a, frac, replicate)
        np.savez(os.path.join(a.datadir, stem + ".npz"),
                 X=X, W=W, mask=mask, Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr, Beta=Beta,
                 info=json.dumps(info))
        info.update(replicate=replicate, mean_count=float(X.mean()),
                    frac_zero=float((X == 0).mean()))
        rows.append(info)
        print(f"[gen] {stem}  invalid_guides={info['n_invalid_guides']} "
              f"genes_affected={info['frac_genes_affected']:.2f} "
              f"entries_affected={info['frac_entries_affected']:.3f} "
              f"|effect| where present={info['mean_abs_effect_where_present']:.2f} "
              f"vs shift {info['mean_abs_perturbation_shift']:.2f} "
              f"mean_count={info['mean_count']:.2f}", flush=True)
    with open(os.path.join(a.datadir, f"gen_invalid_repl{replicate}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def selftest():
    """frac_invalid = 0 must reproduce the published generator, and per-guide severity must
    be constant across the sweep while only prevalence moves."""
    from sim_base import simulate_NB_tgt
    global APPLY_CLIP
    p_, n_, d_ = 20, 3000, 0
    common = dict(theta_m=-np.ones(p_), theta_r=np.ones(p_), n=n_, d=d_, intercept=True,
                  g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
                  hard_intervention=False)

    def W_(seed=11):
        set_random_seed(seed)
        B = simulate_dag(p_, 3 * p_, "ER")
        return simulate_parameter(B, w_ranges=((-1, -0.5), (0.5, 1.0)))

    APPLY_CLIP = False
    W = W_(); Xpub = simulate_NB_tgt(W=W, **common)[0]
    W = W_(); Xmine = simulate_NB_invalid(W=W, frac_invalid=0.0,
                                         aux_rng=np.random.default_rng(7), **common)[0]
    same = np.array_equal(Xpub, Xmine)
    APPLY_CLIP = True
    assert same, "frac_invalid=0 differs from simulate_NB_tgt beyond the overflow clip"
    W = W_(); Xclip = simulate_NB_invalid(W=W, frac_invalid=0.0,
                                         aux_rng=np.random.default_rng(7), **common)[0]
    nbad = int((np.abs(Xpub - Xclip).sum(axis=0) > 0).sum())
    print(f"[selftest] frac_invalid=0 equals simulate_NB_tgt bit-for-bit with the overflow "
          f"clip disabled; with it enabled {nbad} of {p_} genes differ")

    # the auxiliary seed must not matter at frac_invalid = 0
    W = W_(); A = simulate_NB_invalid(W=W, frac_invalid=0.0,
                                      aux_rng=np.random.default_rng(1), **common)[0]
    assert np.array_equal(Xclip, A), "frac_invalid=0 depends on the auxiliary seed"

    # prevalence moves, per-guide severity does not
    print(f"[selftest] {'frac':>5} {'guides':>7} {'leaks':>6} {'per-guide':>10} "
          f"{'genes_affected':>15} {'leak/signal':>12}")
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5):
        W = W_()
        X, _, _, _, _, Vm_, _, Beta, info = simulate_NB_invalid(
            W=W, frac_invalid=frac, aux_rng=np.random.default_rng(7), **common)
        per = info["n_leak"] / max(info["n_invalid_guides"], 1)
        assert per == info["n_recipients"], "per-guide recipient count is not constant"
        validate_counts(X, f"selftest frac={frac}")
        print(f"[selftest] {frac:>5.1f} {info['n_invalid_guides']:>7} {info['n_leak']:>6} "
              f"{per:>10.1f} {info['frac_genes_affected']:>15.2f} "
              f"{info['leak_to_signal']:>12.3f}")
    print("[selftest] OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default="NB")
    ap.add_argument("n", nargs="?", type=int, default=10000)
    ap.add_argument("p", nargs="?", type=int, default=100)
    ap.add_argument("s0", nargs="?", type=int, default=400)
    ap.add_argument("graph", nargs="?", default="ER")
    ap.add_argument("intervention", nargs="?", choices=["soft", "hard"], default="soft")
    ap.add_argument("um", nargs="?", type=int, default=0, help="number of latent factors")
    ap.add_argument("--fracs", default=",".join(str(x) for x in FRACS),
                    help="prevalences of invalid guides; 0 is the no-violation reference")
    ap.add_argument("--n-recipients", type=int, default=10, dest="n_recipients",
                    help="genes each invalid guide directly affects")
    ap.add_argument("--invalid-effect", type=float, default=3.0, dest="invalid_effect",
                    help="beta ~ U(-x, x); 3.0 matches the perturbation-shift scale")
    ap.add_argument("--replicate", type=int,
                    default=int(os.environ.get("LSB_JOBINDEX", 1)))
    ap.add_argument("--datadir", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.datadir is None:
        a.datadir = (f"result/"
                     f"simu_{a.intervention}/data/{a.model}")
    a.fracs = [float(x) for x in a.fracs.split(",")]
    generate(a, a.replicate)


if __name__ == "__main__":
    main()
