#!/usr/bin/env python
"""
VIOLATION OF THE INDEPENDENCE ASSUMPTION: effective uptake depends on latent cell state.

Every other run_sensitivity_arms arm leaves independence intact -- the perturbed set is drawn with
`np.random.choice(n, ..., replace=False)`, with no reference to the latent factors, so the
instruments stay valid however large the confounding grows. The other sweeps break exclusion
(off-target activity, hidden mediator, direct contamination) or relevance (efficacy
heterogeneity); cell-level mismatch breaks neither, since the corruption is applied at random
and so the recorded indicator stays independent of the structural errors, making it a
mismeasurement of the instrument rather than a loss of exogeneity.

MECHANISM. Let u_i be the first latent factor, standardised across cells. The cells that
receive ANY perturbation are drawn with probability proportional to exp(a * u_i) rather than
uniformly; which gene a perturbed cell is assigned to is left uniform. A perturbation
therefore registers preferentially in cells in particular latent states, and because u
carries non-zero loadings on every gene it also enters the target equations -- which is what
makes the dependence a violation rather than a nuisance.

Nothing else is changed: the confounding is the published linear, additive kind, so this arm
isolates the independence violation instead of compounding it with the nonlinear and
non-additive mechanisms tested separately.

WHY THE PERTURBED SET RATHER THAN THE GENE. For a binary indicator with rate q and standard
normal u the attainable correlation is bounded by phi(z_q)/sqrt(q(1-q)):

    whether a cell is perturbed at all   q = 0.90    ceiling 0.585
    which gene, among perturbed cells    q = 0.0090  ceiling 0.257

The per-gene route is capped at less than half the reach by the 90 cells per gene, and it
also interacts badly with the assignment loop, which draws genes sequentially from a
shrinking pool (`remaining = np.setdiff1d(remaining, choice)`): under weighted sampling the
first gene would take the most extreme cells and the last whatever is left, making the
realised violation a function of gene index. Selecting the set is gene-symmetric and leaves
that loop untouched.

WHERE THE DISTORTION LANDS. Because 90% of cells are perturbed, the perturbed group barely
moves while the control group is driven far from average -- measured mean u of -1.56 at
a = 2 against 0.18 in the perturbed group. The violation is a CONTROL-GROUP distortion: the
comparison baseline stops being exchangeable with the treated cells.

    python sim_dependent_assignment.py                  # the production sweep
    python sim_dependent_assignment.py --selftest
    python sim_dependent_assignment.py --alphas 0,1 --n 3000 --p 20 --s0 60 --d 6

Writes to result/simu_{intervention}/data/{model}, where the
fit driver already looks, as

    n{n/1000}k_p{p}_s{s0}_{graph}_cova{d}_sel{100*a:03d}_repl{job}.npz

with keys X, W, mask, Z, Um, Ur, Vm, Vr. The `sel###` tag sits between the factor count and
`_repl`, so a parser must match the setting and the replicate SEPARATELY. These files share a
directory with the dense `..._cova{d}_repl{job}.npz` arm, so a glob on `*_cova{d}_*` matches
both; narrow it by `sel` when selecting this arm.
"""
import argparse, json, os, sys
import numpy as np
import igraph as ig
from scipy.stats import nbinom, norm

sys.path.insert(0, os.environ.get("SCGRAPH_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)))))
from sim_base import is_dag
from utils import simulate_dag, simulate_parameter, set_random_seed

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]
APPLY_CLIP = True      # set False only to verify equivalence with the unpatched generator


def corr_ceiling(q):
    """phi(z_q)/sqrt(q(1-q)): the largest |corr| a rate-q binary indicator can have with u."""
    z = norm.ppf(1 - q)
    return float(norm.pdf(z) / np.sqrt(q * (1 - q)))


def _equation(X, w, cov_m, cov_r, mask, shift_m, shift_r, cap=6.5, floor=0.5):
    """Inner equation of simulate_NB_tgt, verbatim, including its 6.5 / 0.5 constants.

    Those differ from the module-level helper's 7.5 / -0.5 in sim_base.py; matching
    simulate_NB_tgt is what keeps this arm on the count scale of the published baseline. The
    elementwise clip is the overflow fix: the cap and the floor are two sequential shifts and
    the second can undo the first, after which exp(m) can leave the integer range. Output is
    therefore NOT bit-identical to unpatched output -- do not mix patched and unpatched runs.
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


def simulate_NB_selection(W, theta_m, theta_r, n, d=6, intercept=True, g=lambda x: x,
                          shift=2.0, coef=1.0, covar=1.0, intervene_prob=0.9,
                          hard_intervention=False, a_select=0.0):
    """simulate_NB_tgt with uptake depending on the first latent factor.

    a_select = 0 reproduces simulate_NB_tgt exactly: the weighted branch is skipped, so the
    random stream is consumed identically and the result is bit-for-bit equal with the
    overflow clip disabled.
    """
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

    # --- the violation -------------------------------------------------------------------
    # u is a real latent factor, so it enters every gene's equation through Um and Ur. The
    # branch matters for reproducibility as well as clarity: the weighted and unweighted
    # calls consume the random stream differently, so a = 0 must take the original path.
    uu = Z[:, 1] if intercept else Z[:, 0]
    uu = (uu - uu.mean()) / (uu.std() + 1e-12)
    n_pert = int(n * intervene_prob)
    if a_select:
        wsel = np.exp(a_select * uu)
        intervene_idx = np.random.choice(n, size=n_pert, replace=False, p=wsel / wsel.sum())
    else:
        intervene_idx = np.random.choice(n, size=n_pert, replace=False)
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
                            mask_j, shift_m_masked[:, j], shift_r_masked[:, j]).reshape(-1)

    any_pert = np.zeros(n); any_pert[intervene_idx] = 1.0
    per_gene = [np.corrcoef(intervene_mask[:, j].astype(float), uu)[0, 1]
                for j in range(p) if intervene_mask[:, j].any()]
    info = dict(
        a_select=float(a_select),
        # Stored so a later comparison can detect patched/unpatched mixing. With the clip
        # ENABLED -- the production default -- a_select=0 reproduces the published
        # generator's construction and random stream but NOT its output: the clip binds
        # wherever the unpatched code would have overflowed. These files are therefore not
        # interchangeable with previously generated unpatched cova files.
        apply_clip=bool(APPLY_CLIP),
        # the severity axis: report this, not a_select, since a is not interpretable
        corr_perturbed_u=float(np.corrcoef(any_pert, uu)[0, 1]),
        corr_ceiling_set=corr_ceiling(intervene_prob),
        # the quantity actually distorted -- the perturbed group is 90% of cells and barely
        # moves, while the control group is driven away from average
        mean_u_control=float(uu[any_pert == 0].mean()),
        mean_u_perturbed=float(uu[any_pert == 1].mean()),
        corr_D_u_per_gene_mean=float(np.mean(per_gene)),
        corr_D_u_per_gene_sd=float(np.std(per_gene)),
        corr_ceiling_per_gene=corr_ceiling(each / n),
        n_control_cells=int((intervene_mask.sum(axis=1) == 0).sum()),
        cells_per_gene_min=int(intervene_mask.sum(axis=0).min()),
        cells_per_gene_max=int(intervene_mask.sum(axis=0).max()))
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


def stem_for(a, alpha, replicate):
    return (f"n{a.n // 1000}k_p{a.p}_s{a.s0}_{a.graph}_cova{a.d}_"
            f"sel{int(round(alpha * 100)):03d}_repl{replicate}")


def generate(a, replicate):
    os.makedirs(a.datadir, exist_ok=True)
    rows = []
    for alpha in a.alphas:
        # One graph and one data-RNG start per replicate, shared by every setting, so the
        # settings are PAIRED within a replicate and the only thing differing between them is
        # the strength of the selection.
        set_random_seed(replicate)
        B = simulate_dag(a.p, a.s0, a.graph)
        W = simulate_parameter(B, w_ranges=((-1, -0.5), (0.5, 1.0)))
        X, mask, Z, Um, Ur, Vm, Vr, info = simulate_NB_selection(
            W=W, theta_m=-np.ones(a.p), theta_r=np.ones(a.p), n=a.n, d=a.d,
            intercept=True, g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
            intervene_prob=0.9, hard_intervention=(a.intervention == "hard"),
            a_select=alpha)
        validate_counts(X, f"a_select={alpha} replicate={replicate}")
        assert info["n_control_cells"] > 0, "no fully unperturbed cell remains"
        assert info["cells_per_gene_min"] == info["cells_per_gene_max"], \
            "cells per gene are no longer balanced"
        stem = stem_for(a, alpha, replicate)
        np.savez(os.path.join(a.datadir, stem + ".npz"),
                 X=X, W=W, mask=mask, Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr,
                 info=json.dumps(info))
        info.update(replicate=replicate, mean_count=float(X.mean()),
                    frac_zero=float((X == 0).mean()))
        rows.append(info)
        print(f"[gen] {stem}  corr(pert,u)={info['corr_perturbed_u']:+.3f} "
              f"(ceiling {info['corr_ceiling_set']:.3f}) "
              f"mean_u ctrl={info['mean_u_control']:+.2f} pert={info['mean_u_perturbed']:+.2f} "
              f"per_gene={info['corr_D_u_per_gene_mean']:+.3f}"
              f"+/-{info['corr_D_u_per_gene_sd']:.3f} "
              f"controls={info['n_control_cells']} cells/gene={info['cells_per_gene_min']} "
              f"mean_count={info['mean_count']:.2f}", flush=True)
    with open(os.path.join(a.datadir, f"gen_selection_repl{replicate}.json"), "w") as fh:
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

    def run(alpha):
        return simulate_NB_selection(W=W_(), a_select=alpha, **common)

    # (a) a = 0 must reproduce the published generator exactly
    from sim_base import simulate_NB_tgt
    APPLY_CLIP = False
    Xpub = simulate_NB_tgt(W=W_(), **common)[0]
    Xmine = run(0.0)[0]
    same = np.array_equal(Xpub, Xmine)
    APPLY_CLIP = True
    assert same, "a_select=0 differs from simulate_NB_tgt beyond the clip"
    Xclip = run(0.0)[0]
    nbad = int((np.abs(Xpub - Xclip).sum(axis=0) > 0).sum())
    print(f"[selftest] a=0 equals simulate_NB_tgt bit-for-bit with the overflow clip "
          f"disabled; with it enabled {nbad} of {p_} genes differ")

    # (b) severity rises with a, stays under the ceiling, and the control group is what moves
    print(f"[selftest] {'a':>5} {'corr(pert,u)':>13} {'ceiling':>8} {'u|ctrl':>8} "
          f"{'u|pert':>8} {'per-gene':>10} {'ctrl n':>7} {'cells/gene':>11}")
    prev = -np.inf
    for al in ALPHAS:
        X, mask, _, _, _, _, _, i = run(al)
        validate_counts(X, f"selftest a={al}")
        assert i["corr_perturbed_u"] <= i["corr_ceiling_set"] + 1e-9, "corr exceeded its bound"
        assert i["n_control_cells"] > 0, "no unperturbed cell"
        assert i["cells_per_gene_min"] == i["cells_per_gene_max"] == int(n_ * 0.9) // p_, \
            "cells per gene changed"
        if al:
            assert i["corr_perturbed_u"] > prev, "severity is not monotone in a"
        prev = i["corr_perturbed_u"]
        print(f"[selftest] {al:>5.2f} {i['corr_perturbed_u']:>13.3f} "
              f"{i['corr_ceiling_set']:>8.3f} {i['mean_u_control']:>8.2f} "
              f"{i['mean_u_perturbed']:>8.2f} {i['corr_D_u_per_gene_mean']:>10.3f} "
              f"{i['n_control_cells']:>7} {i['cells_per_gene_min']:>11}")

    # (c) the control group is the distorted one, not the perturbed group
    i0, i2 = run(0.0)[-1], run(ALPHAS[-1])[-1]
    assert abs(i2["mean_u_control"]) > 5 * abs(i2["mean_u_perturbed"]), \
        "the distortion is not concentrated in the control group"
    print(f"[selftest] at a={ALPHAS[-1]} the control mean moves to "
          f"{i2['mean_u_control']:+.2f} while the perturbed mean sits at "
          f"{i2['mean_u_perturbed']:+.2f} (both {i0['mean_u_control']:+.2f} / "
          f"{i0['mean_u_perturbed']:+.2f} at a=0)")
    print("[selftest] OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphas", default=",".join(str(x) for x in ALPHAS),
                    help="selection strengths. 0 is the no-violation reference and is NOT "
                         "optional: without it a difference across the sweep cannot be told "
                         "from a baseline that was never zero. Beyond 2 the realised "
                         "correlation saturates against the 0.585 ceiling.")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--s0", type=int, default=400)
    ap.add_argument("--graph", default="ER")
    ap.add_argument("--d", type=int, default=6, help="number of latent factors")
    ap.add_argument("--intervention", choices=["soft", "hard"], default="soft")
    ap.add_argument("--model", default="NB")
    ap.add_argument("--replicate", type=int,
                    default=int(os.environ.get("LSB_JOBINDEX", 1)))
    ap.add_argument("--datadir", default=None,
                    help="defaults to the project convention, "
                         "result/simu_{intervention}/data/"
                         "{model}, which is where the fit driver already looks")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    a.alphas = [float(x) for x in a.alphas.split(",")]
    if a.datadir is None:
        a.datadir = (f"result/"
                     f"simu_{a.intervention}/data/{a.model}")
    print(f"[cfg] d={a.d} intervention={a.intervention} alphas={a.alphas} "
          f"apply_clip={APPLY_CLIP} -> {a.datadir}", flush=True)
    if APPLY_CLIP:
        print("[cfg] NOTE the overflow clip is enabled, so the a=0 reference matches the "
              "published generator's construction but not its output; do not compare these "
              "files against unpatched cova runs.", flush=True)
    generate(a, a.replicate)


if __name__ == "__main__":
    main()
