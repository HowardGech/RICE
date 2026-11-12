"""Partial knockdown: upstream regulation is retained in perturbed cells in
proportion to a per-gene efficacy e_j in [0, 1]. e_j = 0 is a soft intervention and
e_j = 1 a hard one, so the two published settings are the ends of a continuum.
"""
import numpy as np
import typing
import igraph as ig
from scipy.stats import nbinom


def simulate_NB_efficacy(W: np.ndarray,
                         theta_m: np.ndarray,
                         theta_r: np.ndarray,
                         n: int,
                         xi: typing.Union[float, np.ndarray],
                         intercept: bool = True,
                         g: callable = lambda x: x,
                         d: int = 0,
                         shift: float = 1.0,
                         coef: float = 1.0,
                         covar: float = 1.0,
                         intervene_prob: float = 0.9,
                         ) -> tuple:
    """
    Perturb-seq data with per-gene knockdown efficacy.

    Generalizes simulate_NB_tgt by replacing the binary soft/hard switch with a
    continuous per-gene factor. For gene j and cell i the effective switch is

        Dtilde_ij = 1 - (1 - xi_j) * D_ij

    so that residual parent influence in a perturbed cell is xi_j:

        xi_j = 0  -> identical to simulate_NB_tgt(hard_intervention=True)
        xi_j = 1  -> identical to simulate_NB_tgt(hard_intervention=False)

    xi is RESIDUAL PARENT INFLUENCE, i.e. xi_j = 1 - efficacy_j. A gene knocked
    down with 100% efficacy has xi_j = 0; 50% efficacy gives xi_j = 0.5.

    As in simulate_NB_tgt, the switch multiplies the confounder contribution as
    well as the parent sum (that is what hard_intervention does in the original),
    so the two endpoints are reproduced exactly. The perturbation shift itself is
    applied in full whenever the cell is perturbed, independent of xi.

    Parameters
    ----------
    xi : float or array of length p
        Residual parent influence per gene, in [0, 1]. A scalar is broadcast.
    All other parameters are as in simulate_NB_tgt. Note there is no
    `hard_intervention` flag: xi subsumes it.

    Returns
    -------
    X, intervene_mask, Z, Um, Ur, Vm, Vr, xi_vec
    """
    def _simulate_single_equation(X, w, cov_m, cov_r, mask, shift_m, shift_r):
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
        if np.max(m) > 6.5:
            m -= (np.max(m) - 6.5)
        if np.mean(m) < .5:
            m += (.5 - np.mean(m))
        m = np.exp(m)
        r = np.exp(r)
        m = m.flatten()
        r = r.flatten()
        x = nbinom.rvs(r, r / (m + r))
        return x

    p = W.shape[0]
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    if not G.is_dag():
        raise ValueError('W must be a DAG')
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == p

    xi_vec = np.broadcast_to(np.asarray(xi, dtype=float), (p,)).copy()
    if np.any(xi_vec < 0) or np.any(xi_vec > 1):
        raise ValueError('xi must lie in [0, 1]')

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

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        eff_mask = (1.0 - xi_vec[j]) * intervene_mask[:, j].astype(float)
        X[:, j] = _simulate_single_equation(
            g(X[:, parents]), W[parents, j],
            covar_m[:, j], covar_r[:, j],
            eff_mask,
            shift_m_masked[:, j], shift_r_masked[:, j],
        ).reshape(-1)

    return X, intervene_mask, Z, Um, Ur, Vm, Vr, xi_vec
