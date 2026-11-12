"""
Guide-assignment error (mismatched guide) for Perturb-seq simulations.

The data are generated from the TRUE assignment by any of the existing generators
in sim_base.py; this module corrupts the RECORDED assignment afterwards,
so no published generator is modified. The model is then fit on the corrupted
mask while metrics are still scored against the true graph.

Two mechanisms, which violate different assumptions:

  'swap'    a perturbed cell's record is moved to a different gene. The cell really
            was perturbed, but at gene m while the record says gene k. This both
            attenuates the first stage for gene k (relevance) and opens a path from
            the recorded D_k to downstream genes that does not pass through x_k
            (exclusion restriction).

  'dropout' a perturbed cell's record is deleted, so the cell is treated as a
            control. Pure attenuation plus contamination of the control pool;
            the exclusion restriction is untouched.

'both' applies swap and dropout at half the given rate each.
"""
import numpy as np


def corrupt_guide_assignment(mask, rho, mode='swap', rng=None, min_per_gene=1):
    """
    Parameters
    ----------
    mask : (n, p) array
        TRUE perturbation indicator used to generate the data.
    rho : float in [0, 1]
        Probability that a perturbed cell's record is corrupted.
    mode : {'swap', 'dropout', 'both'}
    rng : numpy Generator, optional
    min_per_gene : int
        Raise if any gene ends up with fewer than this many recorded
        perturbations; scRICE_CF.prep() requires at least one.

    Returns
    -------
    mask_obs : (n, p) float array   recorded assignment, to be used for fitting
    info : dict                     realised rates and diagnostics
    """
    if rng is None:
        rng = np.random.default_rng()
    if mode not in ('swap', 'dropout', 'both'):
        raise ValueError(f"mode must be swap|dropout|both, got {mode!r}")
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    mask = np.asarray(mask)
    n, p = mask.shape
    mask_obs = mask.astype(float).copy()

    rows, cols = np.nonzero(mask)                 # one entry per (cell, perturbed gene)
    n_pert = len(rows)
    if n_pert == 0:
        raise ValueError("mask contains no perturbations")

    if mode == 'both':
        rho_swap = rho_drop = rho / 2.0
    elif mode == 'swap':
        rho_swap, rho_drop = rho, 0.0
    else:
        rho_swap, rho_drop = 0.0, rho

    draw = rng.random(n_pert)
    do_swap = draw < rho_swap
    do_drop = (draw >= rho_swap) & (draw < rho_swap + rho_drop)

    # dropout: the record is deleted, the cell reads as a control
    mask_obs[rows[do_drop], cols[do_drop]] = 0.0

    # swap: the record is moved to a uniformly chosen different gene
    n_swap = int(do_swap.sum())
    if n_swap:
        r, c = rows[do_swap], cols[do_swap]
        shift = rng.integers(1, p, size=n_swap)   # 1..p-1 guarantees a different gene
        c_new = (c + shift) % p
        mask_obs[r, c] = 0.0
        mask_obs[r, c_new] = 1.0                  # idempotent if the cell already had it

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
        mode=mode, rho=float(rho),
        n_perturbations=int(n_pert),
        n_swapped=n_swap, n_dropped=int(do_drop.sum()),
        realised_rate=float((n_swap + do_drop.sum()) / n_pert),
        # fraction of recorded (cell, gene) entries that are WRONG: swaps land on a gene
        # the cell was not perturbed at, dropouts remove a true one
        frac_records_incorrect=float(
            (mask_obs.astype(bool) != mask.astype(bool)).sum() / max(mask.sum(), 1)),
        min_recorded_per_gene=float(per_gene.min()),
        n_control_cells=int((mask_obs.sum(axis=1) == 0).sum()),
    )
    return mask_obs, info
