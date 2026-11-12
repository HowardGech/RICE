
r"""{ctype}_{method}_significance_{te,de}.csv for ALL FOUR methods.
Generalizes real_differential_analysis.ipynb cell 68 (= real_soft_summary.py 1134-1173).

Nothing is recomputed: the p-values are read from the npz files cell 67 wrote and the
edge lists from the CSVs cell 31 wrote.

Why both files are needed. Cell 67 saved `np.array(list(p_values_te.values()))`, so the
npz holds VALUES ONLY -- the (i, j) keys are gone. Their order is the order of that
method's *_all_nonzero list, which cell 31 saved as a CSV with source_idx/target_idx.
Reading the two together restores the dict exactly, and a length disagreement between
them is caught here rather than silently mis-pairing every edge.
"""
import os
import numpy as np, pandas as pd

thresholds = [100, 200, 400, 700, 1000]
SIG = 0.01                      # the threshold cell 68 actually used (its comment says .05)

# method -> (edge-list CSV written by cell 31, key inside the pvalues npz, output tag)
FILES = {
    'scRISE':  (f'{ctype}_scRISE_all_nonzero_gene_pairs_normalize.csv', 'RICE',    'scRISE'),
    'IBCD':    (f'{ctype}_IBCD_all_nonzero_gene_pairs.csv',             'IBCD',    'IBCD'),
    'inspre':  (f'{ctype}_inspre_all_nonzero_gene_pairs.csv',           'inspre',  'inspre'),
    'dotears': (f'{ctype}_dotears_all_nonzero_gene_pairs.csv',          'dotears', 'dotears'),
}

PV_TE = np.load(f'{ctype}_pvalues_te.npz', allow_pickle=True)
PV_DE = np.load(f'{ctype}_pvalues_de.npz', allow_pickle=True)

# Optional: IBCD is Bayesian, so order its edges by posterior inclusion probability
# rather than |weight|.  IBCD_pip = pd.read_csv(f'{output_dir}/pip.csv', index_col=0).values
# PIP saturates at 1.0 for many edges, so |weight| breaks the ties.
IBCD_pip = globals().get('IBCD_pip', None)

for method, (edge_csv, npz_key, tag) in FILES.items():
    if not os.path.exists(edge_csv):
        print(f'[skip] {method}: {edge_csv} not found')
        continue
    edges = pd.read_csv(edge_csv)                      # source,target,weight,source_idx,target_idx
    ij = list(zip(edges.source_idx.astype(int), edges.target_idx.astype(int)))

    for tagp, PV in (('te', PV_TE), ('de', PV_DE)):
        pvals = np.asarray(PV[npz_key], dtype=float).ravel()
        if len(pvals) != len(ij):
            raise ValueError(
                f'{ctype}/{method}/{tagp}: {len(pvals)} p-values but {len(ij)} edges in '
                f'{edge_csv}. The npz was written from a DIFFERENT edge list, so position '
                f'cannot be trusted. Regenerate the p-values from the same {method}_W used '
                f'for that CSV, or re-save the edge list from the W the p-values used.')
        pv = dict(zip(ij, pvals))

        # rank within this method's own non-zero edges
        if method == 'IBCD' and IBCD_pip is not None:
            key = lambda t: (IBCD_pip[t[0][0], t[0][1]], abs(t[1]))
            how = 'PIP (|weight| breaks ties)'
        else:
            key = lambda t: (abs(t[1]),)
            how = '|weight|'
        ranked = [p for p, _ in sorted(zip(ij, edges.weight.values), key=key, reverse=True)]

        success, total = [], []
        for k in thresholds:
            take = ranked[:min(k, len(ranked))]        # a method may hold fewer than k
            success.append(sum(1 for p in take if pv[p] < SIG))
            total.append(len(take))
        pd.DataFrame({'success': success, 'total': total}).to_csv(
            f'{ctype}_{tag}_significance_{tagp}.csv', index=False)

    print(f'{method}: {len(ij)} non-zero edges, ranked by {how}'
          + ('' if len(ij) >= max(thresholds) else f'  [TRUNCATED above k={len(ij)}]'))
