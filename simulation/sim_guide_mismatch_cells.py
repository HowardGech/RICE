#!/usr/bin/env python
"""
Generate data in which A FRACTION OF CELLS HAS A MISMATCH between the perturbation it
actually received and the perturbation indicator recorded for it.

The guides are valid and the biology is clean: every cell is perturbed as the true
assignment says, and the counts are generated from that true assignment. What is corrupted
is the RECORD. For a fraction rho of perturbed cells, the recorded entry is moved to a
uniformly chosen different gene, so the model is handed an indicator that points at the
wrong column for those cells.

    mask_obs[i, k] = 0 ,  mask_obs[i, k'] = 1      for a corrupted cell i, k' != k

This is a property of the indicator, not of the data, so for a given replicate X is
IDENTICAL at every rho -- the arms differ in the recorded assignment and in nothing else.
The generator therefore runs once per replicate and every prevalence reuses its counts,
which makes the comparison exactly paired and costs a fifth of the generation time.

READ THE X AXIS CAREFULLY. rho is the fraction of perturbed CELLS whose record is moved.
Each move creates two errors -- a false positive at the new gene and a false negative at the
true one -- so the fraction of recorded entries that disagree with the truth is about 2*rho.
Both are reported: `rho` and `frac_records_incorrect`. Plot against whichever the text
names, and do not describe rho as the error rate of the indicator.

WHAT IS STORED. `mask` is the RECORDED assignment, so the fit driver reads it unchanged and
fits what an analyst would actually have. `mask_true` is the assignment the data were
generated from. `W` is untouched, so scoring is against the true graph either way.

    python sim_guide_mismatch_cells.py NB 10000 100 400 ER soft 0
    python sim_guide_mismatch_cells.py NB 10000 100 400 ER soft 0 --rhos 0.1,0.3,0.5
    python sim_guide_mismatch_cells.py --selftest

Writes n{n/1000}k_p{p}_s{s0}_{graph}_cova{um}_mis{100*rho:03d}_repl{job}.npz. The `mis###`
tag sits between the factor count and `_repl`, so a parser must match the setting and the
replicate SEPARATELY.
"""
from sim_base import is_dag
import argparse, json, os, sys
import numpy as np
import igraph as ig
from scipy.stats import nbinom

sys.path.insert(0, os.environ.get("SCGRAPH_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)))))
from utils import simulate_dag, simulate_parameter, set_random_seed

RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
APPLY_CLIP = True      # set False only to verify equivalence with the unpatched generator


def _equation(X, w, cov_m, cov_r, mask, shift_m, shift_r, cap=6.5, floor=0.5):
    """Inner equation of simulate_NB_tgt, verbatim, including its 6.5 / 0.5 constants.

    Those differ from the module-level helper's 7.5 / -0.5 in sim_base.py.
    The elementwise clip is the overflow fix: the cap and the floor are two sequential
    shifts and the second can undo the first, after which exp(m) can leave the integer
    range. It binds only where the unpatched code would have overflowed.
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


def simulate_NB_tgt_local(W, theta_m, theta_r, n, intercept=True, g=lambda x: x, d=0,
                          shift=1.0, coef=1.0, covar=1.0, intervene_prob=0.9,
                          hard_intervention=True):
    """The published simulate_NB_tgt, reproduced here so this file is self-contained."""
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

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        mask_j = intervene_mask[:, j] if hard_intervention else np.zeros(n)
        X[:, j] = _equation(g(X[:, parents]), W[parents, j], covar_m[:, j], covar_r[:, j],
                            mask_j, shift_m_masked[:, j], shift_r_masked[:, j]).reshape(-1)

    return X, intervene_mask, Z, Um, Ur, Vm, Vr


def corrupt_guide_assignment(mask, rho, rng=None, min_per_gene=1):
    """Move a fraction rho of perturbation records to a uniformly chosen different gene.

    The cell's ACTUAL perturbation is unchanged -- only the record moves, so this is a
    mismatch between the perturbation a cell received and the indicator recorded for it.
    rho = 0 returns the true mask.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    mask = np.asarray(mask)
    n, p = mask.shape
    mask_obs = mask.astype(float).copy()
    rows, cols = np.nonzero(mask)
    n_pert = len(rows)
    if n_pert == 0:
        raise ValueError("mask contains no perturbations")

    do_swap = rng.random(n_pert) < rho
    n_swap = int(do_swap.sum())
    if n_swap:
        r, c = rows[do_swap], cols[do_swap]
        shift = rng.integers(1, p, size=n_swap)    # 1..p-1 guarantees a different gene
        c_new = (c + shift) % p
        mask_obs[r, c] = 0.0
        mask_obs[r, c_new] = 1.0

    per_gene = mask_obs.sum(axis=0)
    if per_gene.min() < min_per_gene:
        bad = int((per_gene < min_per_gene).sum())
        raise ValueError(
            f"{bad} gene(s) left with fewer than {min_per_gene} recorded perturbations "
            f"after corruption (min {per_gene.min():.0f}); scRICE prep() will reject this. "
            f"Lower rho, raise n, or re-draw the replicate.")
    if not (mask_obs.sum(axis=1) == 0).any():
        raise ValueError("no fully unperturbed cell remains; scRICE prep() requires one")

    info = dict(
        rho=float(rho), n_perturbations=int(n_pert), n_swapped=n_swap,
        realised_rate=float(n_swap / n_pert),
        # Fraction of recorded entries that disagree with the truth. Each move creates a
        # false positive at the new gene AND a false negative at the true one, so this is
        # about twice rho -- it is the error rate of the indicator, rho is not.
        frac_records_incorrect=float(
            (mask_obs.astype(bool) != mask.astype(bool)).sum() / max(mask.sum(), 1)),
        min_recorded_per_gene=float(per_gene.min()),
        n_control_cells=int((mask_obs.sum(axis=1) == 0).sum()))
    return mask_obs, info


def validate_counts(X, where):
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


def stem_for(a, rho, replicate):
    return (f"n{a.n // 1000}k_p{a.p}_s{a.s0}_{a.graph}_cova{a.um}_"
            f"mis{int(round(rho * 100)):03d}_repl{replicate}")


def generate(a, replicate):
    os.makedirs(a.datadir, exist_ok=True)
    # Generate ONCE from the true assignment; the corruption is a property of the record,
    # so every rho shares these counts and the arms are paired by construction. The seed is
    # the array index, matching the fit driver that reads these files.
    set_random_seed(replicate)
    B = simulate_dag(a.p, a.s0, a.graph)
    W = simulate_parameter(B, w_ranges=((-1, -0.5), (0.5, 1.0)))
    X, mask_true, Z, Um, Ur, Vm, Vr = simulate_NB_tgt_local(
        W=W, theta_m=-np.ones(a.p), theta_r=np.ones(a.p), n=a.n, d=a.um,
        intercept=True, g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1,
        intervene_prob=0.9, hard_intervention=(a.intervention == "hard"))
    validate_counts(X, f"replicate={replicate}")

    rows = []
    for rho in a.rhos:
        # own stream, so rho = 0 consumes nothing and the corruption of one rho does not
        # shift the draws of another
        mask_obs, info = corrupt_guide_assignment(
            mask_true, rho, rng=np.random.default_rng(90000 + replicate), min_per_gene=1)
        stem = stem_for(a, rho, replicate)
        np.savez(os.path.join(a.datadir, stem + ".npz"),
                 X=X, W=W, mask=mask_obs, mask_true=mask_true,
                 Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr, info=json.dumps(info))
        info.update(replicate=replicate)
        rows.append(info)
        print(f"[gen] {stem}  swapped={info['n_swapped']}/{info['n_perturbations']} "
              f"(rate {info['realised_rate']:.3f})  "
              f"records_incorrect={info['frac_records_incorrect']:.3f}  "
              f"min_per_gene={info['min_recorded_per_gene']:.0f}  "
              f"controls={info['n_control_cells']}", flush=True)
    with open(os.path.join(a.datadir, f"gen_mismatch_repl{replicate}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def selftest():
    """Invariants of the corruption, and that rho = 0 leaves the record untouched."""
    global APPLY_CLIP
    p_, n_ = 20, 3000
    set_random_seed(11)
    B = simulate_dag(p_, 3 * p_, "ER")
    W = simulate_parameter(B, w_ranges=((-1, -0.5), (0.5, 1.0)))
    X, mask_true, *_ = simulate_NB_tgt_local(
        W=W, theta_m=-np.ones(p_), theta_r=np.ones(p_), n=n_, d=0, intercept=True,
        g=lambda x: np.log(1 + x), shift=2, coef=1, covar=1, hard_intervention=False)
    validate_counts(X, "selftest")

    m0, i0 = corrupt_guide_assignment(mask_true, 0.0, rng=np.random.default_rng(7))
    assert np.array_equal(m0.astype(bool), mask_true.astype(bool)), "rho=0 altered the mask"
    assert i0["frac_records_incorrect"] == 0.0
    print("[selftest] rho=0 leaves the recorded assignment identical to the truth")

    print(f"[selftest] {'rho':>5} {'swapped':>8} {'realised':>9} {'wrong recs':>11} "
          f"{'ratio':>6} {'min/gene':>9} {'controls':>9}")
    for rho in (0.1, 0.2, 0.3, 0.4, 0.5):
        m, i = corrupt_guide_assignment(mask_true, rho, rng=np.random.default_rng(7))
        # a swap moves a record, it does not create or destroy one
        assert m.sum() == mask_true.sum(), "swap changed the number of records"
        # a corrupted cell is still perturbed exactly once, so no cell gains a second entry
        assert m.sum(axis=1).max() == mask_true.sum(axis=1).max(), "a cell gained a record"
        ratio = i["frac_records_incorrect"] / max(i["realised_rate"], 1e-12)
        assert 1.8 < ratio < 2.2, f"wrong-record ratio {ratio:.2f} is not about 2"
        print(f"[selftest] {rho:>5.1f} {i['n_swapped']:>8} {i['realised_rate']:>9.3f} "
              f"{i['frac_records_incorrect']:>11.3f} {ratio:>6.2f} "
              f"{i['min_recorded_per_gene']:>9.0f} {i['n_control_cells']:>9}")
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
    ap.add_argument("--rhos", default=",".join(str(x) for x in RHOS),
                    help="fractions of perturbed CELLS whose record is moved; "
                         "0 is the no-corruption reference")
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
    a.rhos = [float(x) for x in a.rhos.split(",")]
    generate(a, a.replicate)


if __name__ == "__main__":
    main()
