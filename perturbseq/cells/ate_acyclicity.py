
r"""Build a graph from the FDR-significant ATEs and test whether it is acyclic.

Edge i -> j iff the ATE of perturbing i on j is significant at FDR_MAIN. The question
is whether that relation could be a causal DAG at all. Three failure modes are
measured separately, because they say different things:
  reciprocal pairs   i -> j and j -> i both significant: unorientable by ATE alone
  strongly connected components  any cycle, not just length 2
  edges inside non-trivial SCCs  the edges implicated in cycles; some subset of them
                                 must be DELETED to obtain a DAG (an upper bound on the
                                 minimum feedback arc set, not the exact number)
The sweep repeats the test over thresholds, so the result cannot be dismissed as an
artifact of choosing 5%.

Needs {ctype}_ATE.npz. The GRAPHS dict is optional: if the fitted weight matrices are
in the namespace the same statistics are reported for them as a reference.
"""
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.linalg import expm

FDR_MAIN  = 0.05
FDR_SWEEP = [0.001, 0.01, 0.05, 0.10, 0.20]

Z = np.load(f'{ctype}_ATE.npz', allow_pickle=True)
q, ate = Z['q'], Z['ate']
genes = Z['genes'] if 'genes' in Z else np.array([f'g{i}' for i in range(q.shape[0])])
p = q.shape[0]


def graph_stats(A, label):
    """A is a boolean adjacency with A[i, j] = edge i -> j. Diagonal ignored."""
    A = np.asarray(A, dtype=bool).copy()
    np.fill_diagonal(A, False)
    n_edges = int(A.sum())
    recip = A & A.T
    n_recip_pairs = int(recip.sum() // 2)

    n_scc, lab = connected_components(csr_matrix(A), directed=True, connection='strong')
    sizes = np.bincount(lab)
    big = np.where(sizes > 1)[0]                      # non-trivial = contains a cycle
    in_cycle = np.isin(lab, big)
    # edges with both endpoints in the same non-trivial SCC. Directions are fixed by the
    # ATE, so these are the edges involved in cycles: a subset of them must be removed to
    # make the graph acyclic. Upper bound on the minimum feedback arc set.
    same = (lab[:, None] == lab[None, :]) & in_cycle[:, None] & in_cycle[None, :]
    n_edges_in_scc = int((A & same).sum())

    # NOTEARS acyclicity score, 0 exactly when the graph is a DAG
    h = float(np.trace(expm(A.astype(float))) - p)
    return dict(graph=label, n_nodes=p, n_edges=n_edges,
                density=n_edges / (p * (p - 1)),
                is_dag=bool(len(big) == 0),
                n_reciprocal_pairs=n_recip_pairs,
                frac_edges_reciprocal=float(recip.sum() / n_edges) if n_edges else np.nan,
                n_nontrivial_scc=int(len(big)),
                largest_scc=int(sizes[big].max()) if len(big) else 1,
                n_genes_in_a_cycle=int(in_cycle.sum()),
                frac_genes_in_a_cycle=float(in_cycle.mean()),
                n_edges_inside_scc=n_edges_in_scc,
                frac_edges_inside_scc=float(n_edges_in_scc / n_edges) if n_edges else np.nan,
                h_acyclicity=h)


# --- the FDR_MAIN graph, plus the fitted graphs as a reference ----------------------
A_main = np.isfinite(q) & (q < FDR_MAIN)
rows = [graph_stats(A_main, f'ATE q<{FDR_MAIN}')]
for name, W in (globals().get('GRAPHS') or
                {k: v for k, v in (('RICE', globals().get('W_cpd')),
                                   ('DOTEARS', globals().get('dotears_w')),
                                   ('IBCD', globals().get('IBCD_W')),
                                   ('inspre', globals().get('inspre_W')))
                 if v is not None}).items():
    rows.append(graph_stats(np.asarray(W) != 0, name))
main = pd.DataFrame(rows)
main.insert(0, 'dataset', ctype)
main.to_csv(f'{ctype}_ate_graph_acyclicity.csv', index=False)
print(main.to_string(index=False))

# --- does any threshold make it a DAG? ---------------------------------------------
sweep = pd.DataFrame([dict(dataset=ctype, fdr=f,
                           **{k: v for k, v in graph_stats(
                               np.isfinite(q) & (q < f), f'ATE q<{f}').items()
                              if k != 'graph'})
                      for f in FDR_SWEEP])
sweep.to_csv(f'{ctype}_ate_graph_acyclicity_sweep.csv', index=False)
print('\n== threshold sweep ==')
print(sweep[['fdr', 'n_edges', 'is_dag', 'n_reciprocal_pairs', 'largest_scc',
             'frac_genes_in_a_cycle', 'frac_edges_inside_scc']].to_string(index=False))

# --- the edge list itself, for downstream use --------------------------------------
ii, jj = np.where(A_main)
pd.DataFrame({'dataset': ctype, 'source': genes[ii], 'target': genes[jj],
              'source_idx': ii, 'target_idx': jj, 'ate': ate[ii, jj], 'q': q[ii, jj],
              'reciprocal': A_main[jj, ii]}).to_csv(
    f'{ctype}_ate_graph_edges_fdr{FDR_MAIN}.csv', index=False)
print(f"\nwrote {ctype}_ate_graph_edges_fdr{FDR_MAIN}.csv "
      f"({int(A_main.sum()):,} edges, {int((A_main & A_main.T).sum()//2):,} reciprocal pairs)")
