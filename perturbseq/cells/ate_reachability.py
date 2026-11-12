
r"""RICE vs the vanilla ATE: the ATE measures total effects, which a causal DAG predicts
topologically, but the ATE is not itself a causal graph.

Part 1 and Part 2 make NO functional-form assumption: they use only REACHABILITY in
each estimated DAG (is j a descendant of i?), so nothing depends on the equations
being linear in x. Part 3 adds a magnitude comparison that DOES require a first-order
approximation and is labelled as such.

Needs: {ctype}_ATE.npz (from the ATE cell) and the weight matrices W_cpd, IBCD_W,
inspre_W, dotears_w.
"""
import numpy as np, pandas as pd

FDR      = 0.05
APPROX   = True        # Part 3: first-order total effects. Set False to skip.
GRAPHS   = {'RICE': W_cpd, 'DOTEARS': dotears_w, 'IBCD': IBCD_W, 'inspre': inspre_W}

Z = np.load(f'{ctype}_ATE.npz', allow_pickle=True)
ate, q, tstat = Z['ate'], Z['q'], Z['t']
p = ate.shape[0]
msr = np.isfinite(ate)                      # pairs where an ATE could be estimated
sig = msr & (q < FDR)
print(f'{ctype}: {msr.sum():,} estimable pairs, {sig.sum():,} significant ATEs '
      f'({sig.sum()/msr.sum():.1%} of pairs)')


def reachable(W):
    """Boolean descendant matrix: R[i, j] = a directed path i -> ... -> j exists."""
    A = (np.asarray(W) != 0)
    np.fill_diagonal(A, False)
    R = A.copy()
    while True:                             # transitive closure by repeated squaring
        R2 = R | (R @ R)
        if np.array_equal(R2, R):
            return R
        R = R2


def auc(score, label):
    """Mann-Whitney AUC of `score` separating label True from False."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().values
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# --- Part 1. Does each graph predict WHERE the ATE is significant? ------------------
# Purely topological: a total effect should exist exactly between ancestor-descendant
# pairs. No scale, no link function, no fitted parameters beyond the graph itself.
rows = []
R = {}
for name, W in GRAPHS.items():
    R[name] = reachable(W)
    direct = (np.asarray(W) != 0) & msr
    reach = R[name] & msr
    rows.append(dict(
        dataset=ctype, method=name,
        n_direct_edges=int(direct.sum()), n_reachable_pairs=int(reach.sum()),
        auc_reach_vs_ate_t=auc(np.abs(tstat[msr]), reach[msr]),
        # of the pairs the graph says are connected, how many show a significant ATE
        precision_reachable=float(sig[reach].mean()) if reach.sum() else np.nan,
        # of the significant ATEs, how many the graph connects by SOME path
        recall_reachable=float(reach[sig].mean()) if sig.sum() else np.nan,
        # ... and how many it connects by a DIRECT edge
        recall_direct=float(direct[sig].mean()) if sig.sum() else np.nan))
part1 = pd.DataFrame(rows)

# --- Part 2. Is the significant-ATE relation itself a valid causal graph? -----------
# Three properties a DAG must have and the thresholded ATE does not.
recip = sig & sig.T                                  # i->j and j->i both significant
A = sig.copy(); np.fill_diagonal(A, False)
Rate = reachable(A.astype(float))
n_cyclic_nodes = int(np.diag(Rate).sum())            # genes lying on a directed cycle
# of significant ATEs, how many RICE explains as a PATH rather than an edge
rice_edge = (np.asarray(GRAPHS['RICE']) != 0)
mediated = sig & R['RICE'] & ~rice_edge
part2 = pd.DataFrame([dict(
    dataset=ctype,
    n_significant_ate=int(sig.sum()),
    density_vs_rice=float(sig.sum() / max(rice_edge.sum(), 1)),
    frac_reciprocal=float(recip[sig].mean()) if sig.sum() else np.nan,
    n_genes_on_a_cycle=n_cyclic_nodes, n_genes=p,
    frac_explained_by_rice_path=float((sig & R['RICE'])[sig].mean()) if sig.sum() else np.nan,
    frac_mediated_not_direct=float(mediated[sig].mean()) if sig.sum() else np.nan)])

# --- Part 3. Magnitude agreement (FIRST-ORDER APPROXIMATION -- state this) ----------
# On the analysis scale the structural equation is linear in the transformed predictors,
# so T = (I - W)^-1 - I is the first-order total effect per unit change in the source.
# It is NOT exact for the fitted GLM, and the ATE is the effect of the PERTURBATION, not
# of a unit change in x_i, so each row is compared by RANK rather than by value.
part3 = pd.DataFrame()
if APPROX:
    rows = []
    for name, W in GRAPHS.items():
        Wm = np.asarray(W, dtype=float)
        try:
            T = np.linalg.inv(np.eye(p) - Wm) - np.eye(p)
        except np.linalg.LinAlgError:
            print(f'[warn] {name}: I - W is singular; skipping magnitude comparison')
            continue
        both = msr & np.isfinite(T)
        sp = pd.Series(np.abs(T[both])).corr(pd.Series(np.abs(ate[both])), method='spearman')
        # per-source rank correlation, which removes the per-gene knockdown scale entirely
        # skip sources with no descendants: |T[i]| is then constant and the rank
        # correlation is undefined (pandas warns and returns NaN)
        per_src = [pd.Series(np.abs(T[i][msr[i]])).corr(
                       pd.Series(np.abs(ate[i][msr[i]])), method='spearman')
                   for i in range(p)
                   if msr[i].sum() > 10 and np.ptp(np.abs(T[i][msr[i]])) > 0]
        rows.append(dict(dataset=ctype, method=name, spearman_all=sp,
                         median_spearman_per_source=float(np.nanmedian(per_src)),
                         n_sources=len(per_src)))
    part3 = pd.DataFrame(rows)

for df, nm in ((part1, 'ate_vs_graph_topology'), (part2, 'ate_graph_validity'),
               (part3, 'ate_vs_graph_magnitude')):
    if len(df):
        f = f'{ctype}_{nm}.csv'
        df.to_csv(f, index=False)
        print(f'\n== {nm} ==\n{df.to_string(index=False)}')
