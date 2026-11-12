"""Sortability statistics and the ordering baseline, shared by the scripts in this
directory.

These four functions were previously copied into sortability_stats.py,
ordering_baselines.py and sortability_sweep.py, where they had drifted apart in
formatting and in one docstring (one copy described the L1 penalty as chosen by
cross-validation; all three in fact use LassoLarsIC with the BIC criterion, which is
what the reference implementation does). The bodies are the ones the published sweep
ran, so results are unchanged by the consolidation.
"""
import re
import numpy as np


def parse_tag(fname):
    s = SETTING_RE.search(fname)
    r = REPL_RE.search(fname)
    return (int(s.group(1)) if s else -1, int(r.group(1)) if r else -1)


def varsortability(X, W, tol=1e-9):
    """Reisach et al. (2021). W[i, j] != 0 means i -> j. Fraction of directed paths along
    which marginal variance increases, ties counted as one half. 0.5 is chance."""
    E = (W != 0)
    Ek = E.copy()
    var = np.var(X, axis=0, keepdims=True)
    n_paths, n_correct = 0, 0.0
    for _ in range(E.shape[0] - 1):
        if not Ek.any():
            break
        ratio = Ek * var / var.T                 # [i, j] = var_j / var_i on path i -> j
        n_paths += int(Ek.sum())
        n_correct += float((ratio > 1 + tol).sum())
        n_correct += 0.5 * float(((ratio <= 1 + tol) & (ratio > 1 - tol) & Ek).sum())
        Ek = Ek.dot(E)
    return n_correct / n_paths if n_paths else np.nan


def _r2_each_on_rest(X):
    """R^2 of every column regressed on all the others, by least squares."""
    n, p = X.shape
    Xc = X - X.mean(0)
    r2 = np.empty(p)
    for j in range(p):
        others = np.delete(np.arange(p), j)
        A = np.column_stack([Xc[:, others], np.ones(n)])
        coef, *_ = np.linalg.lstsq(A, Xc[:, j], rcond=None)
        tot = float(np.var(Xc[:, j]))
        r2[j] = 1.0 - float(np.var(Xc[:, j] - A @ coef)) / tot if tot > 0 else np.nan
    return r2


def r2sortability(X, W, tol=1e-9):
    """As varsortability, with R^2-on-the-rest in place of marginal variance. Invariant to
    rescaling any single variable, so this is the statistic to quote against a units
    objection."""
    E = (W != 0)
    Ek = E.copy()
    r2 = _r2_each_on_rest(X).reshape(1, -1)
    n_paths, n_correct = 0, 0.0
    for _ in range(E.shape[0] - 1):
        if not Ek.any():
            break
        ratio = Ek * r2 / r2.T
        n_paths += int(Ek.sum())
        n_correct += float((ratio > 1 + tol).sum())
        n_correct += 0.5 * float(((ratio <= 1 + tol) & (ratio > 1 - tol) & Ek).sum())
        Ek = Ek.dot(E)
    return n_correct / n_paths if n_paths else np.nan


def sortnregress(X, order=None):
    """Reisach et al. (2021). Regress each node on its predecessors in `order` under an L1
    penalty chosen by BIC. With order=None the nodes are sorted by marginal variance."""
    from sklearn.linear_model import LassoLarsIC, LinearRegression
    p = X.shape[1]
    order = np.argsort(np.var(X, axis=0)) if order is None else np.asarray(order)
    W = np.zeros((p, p))
    for k in range(1, p):
        target, preds = order[k], order[:k]
        lr = LinearRegression().fit(X[:, preds], X[:, target])
        weight = np.abs(lr.coef_)
        ll = LassoLarsIC(criterion='bic').fit(X[:, preds] * weight, X[:, target])
        W[preds, target] = ll.coef_ * weight
    return W
