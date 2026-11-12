
r"""How closely does each method agree with the vanilla ATE?

Three axes, because "agreement" means different things and the fair comparison
differs by axis. All of them are restricted to pairs where an ATE could actually be
estimated, so no method is charged for pairs with no measurement.

  DIRECT      method edge i->j        vs significant ATE i->j.
              Expected to be INCOMPLETE on purpose: the ATE is a total effect, so a
              mediated pair is a true ATE with no direct edge. Low recall here is the
              mediation argument, not a failure.
  REACHABLE   j is a descendant of i  vs significant ATE i->j.
              The like-for-like comparison: this is what a DAG predicts about totals.
  RANK        |W| (or |ATE| ranking)  vs |ATE| magnitude, overall and per source gene.

  ORIENTATION among pairs where the ATE is significant in exactly ONE direction,
              does the method's edge point the same way? A graph can agree on which
              pairs are connected and still disagree on direction.

Two further columns control for how many pairs a method asserts, since precision and
recall both move with that and a denser graph can buy recall outright:

  precision_at_matched_k  every method is cut to the SAME number of edges -- the
              smallest edge count among them -- taking each method's own top edges by
              |W|. This is the reviewer's matched-edge-count comparison applied to the
              ATE criterion, and it is the one agreement number that is not confounded
              by sparsity.
  auc_edges_vs_ate_t / auc_reach_vs_ate_t  threshold-free: how well membership of the
              edge set (or the descendant set) separates pairs by |t| of the ATE.
              Chance is 0.5 and the value does not depend on the size of the set.

Needs {ctype}_ATE.npz plus W_cpd, IBCD_W, inspre_W, dotears_w.
"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

FDR    = 0.05
GRAPHS = {'RICE': W_cpd, 'DOTEARS': dotears_w, 'IBCD': IBCD_W, 'inspre': inspre_W}

Z = np.load(f'{ctype}_ATE.npz', allow_pickle=True)
ate, q = Z['ate'], Z['q']
# t is what the AUC columns rank on; older npz files may carry only se
tstat = Z['t'] if 't' in Z.files else ate / Z['se']
p = ate.shape[0]
msr = np.isfinite(ate)                       # pairs with an ATE estimate
sig = msr & (q < FDR)


def reachable(W):
    A = (np.asarray(W) != 0); np.fill_diagonal(A, False)
    R = A.copy()
    while True:
        R2 = R | (R @ R)
        if np.array_equal(R2, R):
            return R
        R = R2


def agree(pred, truth, mask):
    """Confusion-based agreement between two boolean relations, over `mask` only."""
    pr, tr = pred[mask], truth[mask]
    tp = int((pr & tr).sum()); fp = int((pr & ~tr).sum())
    fn = int((~pr & tr).sum()); tn = int((~pr & ~tr).sum())
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else np.nan
    orat, pval = (fisher_exact([[tp, fp], [fn, tn]], alternative='greater')
                  if min(tp + fp, tp + fn) > 0 else (np.nan, np.nan))
    return dict(n_pred=tp + fp, tp=tp, fp=fp, fn=fn,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                recall=tp / (tp + fn) if tp + fn else np.nan,
                jaccard=tp / (tp + fp + fn) if (tp + fp + fn) else np.nan,
                mcc=mcc, odds_ratio=orat, p_enrichment=pval)


def auc(score, label, mask):
    """Mann-Whitney AUC of `score` separating label True from False, over `mask`."""
    s, y = np.asarray(score)[mask], np.asarray(label)[mask]
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().values
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# the matched edge count: the smallest number of edges any method reports
_counts = {}
for _n, _W in GRAPHS.items():
    _E = (np.asarray(_W) != 0) & msr
    np.fill_diagonal(_E, False)
    _counts[_n] = int(_E.sum())
MATCHED_K = min(_counts.values())
print('edges per method:', _counts, '-> matched k =', MATCHED_K)

rows = []
for name, W in GRAPHS.items():
    Wm = np.asarray(W, dtype=float)
    E = (Wm != 0); np.fill_diagonal(E, False)
    R = reachable(Wm)

    r_direct = agree(E, sig, msr)
    r_reach = agree(R, sig, msr)

    # rank agreement: over the method's own edges, and over all estimable pairs
    on_edge = E & msr
    sp_edge = (pd.Series(np.abs(Wm[on_edge])).corr(pd.Series(np.abs(ate[on_edge])),
                                                   method='spearman')
               if on_edge.sum() > 10 else np.nan)
    sp_all = pd.Series(np.abs(Wm[msr])).corr(pd.Series(np.abs(ate[msr])), method='spearman')
    per_src = [pd.Series(np.abs(Wm[i][msr[i]])).corr(
                   pd.Series(np.abs(ate[i][msr[i]])), method='spearman')
               for i in range(p) if msr[i].sum() > 10 and np.ptp(np.abs(Wm[i][msr[i]])) > 0]

    # size-controlled: cut every method to the same number of its own top edges
    ii, jj = np.where(on_edge)
    rank = np.argsort(-np.abs(Wm[ii, jj]))[:MATCHED_K]
    prec_matched = float(sig[ii[rank], jj[rank]].mean()) if len(rank) else np.nan

    # orientation: pairs where the ATE is significant one way only
    one_way = sig & ~sig.T
    agreed = int((E & one_way).sum())
    flipped = int((E & one_way.T).sum())
    rows.append(dict(
        dataset=ctype, method=name,
        **{f'direct_{k}': v for k, v in r_direct.items()},
        **{f'reach_{k}': v for k, v in r_reach.items()},
        matched_k=MATCHED_K, precision_at_matched_k=prec_matched,
        auc_edges_vs_ate_t=auc(np.abs(tstat), E, msr),
        auc_reach_vs_ate_t=auc(np.abs(tstat), R, msr),
        spearman_on_edges=sp_edge, spearman_all_pairs=sp_all,
        median_spearman_per_source=float(np.nanmedian(per_src)) if per_src else np.nan,
        orient_agree=agreed, orient_flip=flipped,
        orient_frac_agree=agreed / (agreed + flipped) if agreed + flipped else np.nan))

tab = pd.DataFrame(rows)
tab.to_csv(f'{ctype}_ate_method_agreement.csv', index=False)

print(f'{ctype}: {int(msr.sum()):,} estimable pairs, {int(sig.sum()):,} significant ATEs\n')
print('-- DIRECT edge vs significant ATE (expected to under-recall: ATE is a total effect)')
print(tab[['method', 'direct_n_pred', 'direct_precision', 'direct_recall',
           'direct_mcc', 'direct_odds_ratio']].round(3).to_string(index=False))
print('\n-- REACHABLE (descendant) vs significant ATE: the like-for-like comparison')
print(tab[['method', 'reach_n_pred', 'reach_precision', 'reach_recall',
           'reach_mcc', 'reach_odds_ratio']].round(3).to_string(index=False))
print('\n-- size-controlled agreement (not confounded by how many edges a method reports)')
print(tab[['method', 'matched_k', 'precision_at_matched_k',
           'auc_edges_vs_ate_t', 'auc_reach_vs_ate_t']].round(3).to_string(index=False))
print('\n-- magnitude and orientation')
print(tab[['method', 'spearman_on_edges', 'spearman_all_pairs',
           'median_spearman_per_source', 'orient_agree', 'orient_flip',
           'orient_frac_agree']].round(3).to_string(index=False))
