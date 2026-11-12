#!/usr/bin/env python
"""
Is the variance ordering ENOUGH to recover the graph?

Varsortability says the causal order is almost readable off the marginal variances. It
does not say the graph is. This script separates the two by running one and the same
edge-selection procedure under three orderings and changing nothing else:

  randomsort      a random permutation              -- floor
  sortnregress    nodes sorted by marginal variance -- what varsortability actually buys
  oracleorder     the TRUE topological order of W   -- upper bound for ANY ordering method

The third is the point. A method cannot extract more from ordering information than it
would get from the correct order handed to it for free, so oracleorder bounds every
order-exploiting approach on these data. If RICE sits above oracleorder, its accuracy
cannot be an artefact of the variance ladder, however high varsortability is -- and the
reason is structural rather than empirical: knowing the order reduces the problem from
p(p-1) candidate edges to p(p-1)/2, which is still a selection problem, and regressing a
gene on its predecessors does nothing about latent confounding, so confounded pairs are
selected whether the order is right or not.

Also reported, because it is the quantity the objection is really about:
  order_agreement  fraction of TRUE edges whose endpoints the ordering places in the
                   correct direction -- i.e. how much of the graph the ordering even
                   permits to be found.

    python ordering_baselines.py path/to/*.npz
    python ordering_baselines.py --rice-csv RESULTDIR path/to/*.npz
    python ordering_baselines.py --selftest

With --rice-csv the script reads the per-replicate result CSVs that the fitting drivers
wrote for the same setting and replicate, and puts RICE on the same table. Note that only
the three baselines are rescored here; RICE's rows carry whatever w_threshold its driver
used, so confirm that matches THRESH below or the comparison is between differently
thresholded edge sets.

SCORING. Baseline edge sets are scored with the project's own measurement(), after the
same threshold-then-project-to-a-DAG step job_simu.py applies to RICE:
    W_est = result(w_threshold=0.3)['W']; W_est = to_dag(W_est); measurement(W_true, W_est)
Scoring them any other way makes the comparison not like-for-like -- in particular SHD
there comes from MetricsDAG, which counts a reversed edge once, whereas a hand-rolled
fp + fn + reversals double-counts it. Set SCGRAPH_DIR to the directory holding utils.py.
If that import fails the fallback scorer runs and every row is tagged
FALLBACK-not-comparable; do not place those numbers next to published ones.

This file is deliberately self-contained -- varsortability, sortnregress, the tag parser
and the scorer are duplicated here rather than imported. sortability_stats.py holds the
same four; if you change one, change both.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sortability import parse_tag, varsortability, r2sortability, sortnregress
import os
import re
import sys

import numpy as np
import pandas as pd

THRESH = 0.3            # must match the w_threshold the fitting driver used for RICE

# The setting and the replicate are matched SEPARATELY, because the generator can write an
# extra tag between them (job_gen_cova.py appends a generator tag such as _hm, giving
# ..._um10_hm_repl1.npz). A single pattern requiring _repl immediately after the digits
# silently fails on those and every file lands in one pooled group reported as setting -1,
# which looks like a result rather than a parse failure.
SETTING_RE = re.compile(r'_(?:um|cova|d)(\d+)(?=_|\.)')
REPL_RE = re.compile(r'repl(\d+)')

try:
    sys.path.insert(0, os.environ.get('SCGRAPH_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
    from utils import measurement, to_dag
    _HAVE_UTILS = True
except Exception:                                    # noqa: BLE001
    _HAVE_UTILS = False


def score_edges(W_hat, W_true, thresh=THRESH):
    if _HAVE_UTILS:
        W_est = W_hat.copy()
        W_est[np.abs(W_est) <= thresh] = 0.0
        W_est = to_dag(W_est)
        acc = measurement(W_true, W_est)
        return dict(ppv=acc['precision'], tpr=acc['recall'],
                    mcc=acc['mcc'], shd=acc['shd'], scorer='utils.measurement')
    P = np.abs(W_hat) > thresh
    T = W_true != 0
    tp = int((P & T).sum()); fp = int((P & ~T).sum()); fn = int((~P & T).sum())
    tn = int((~P & ~T).sum())
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(ppv=tp / (tp + fp) if tp + fp else np.nan,
                tpr=tp / (tp + fn) if tp + fn else np.nan,
                mcc=((tp * tn - fp * fn) / den) if den > 0 else np.nan,
                shd=int(fp + fn), scorer='FALLBACK-not-comparable')


def topological_order(W):
    """Kahn's algorithm on W[i, j] != 0 meaning i -> j. Raises if W is not acyclic."""
    E = (W != 0).astype(int)
    indeg = E.sum(0).astype(float)
    order, avail = [], [int(i) for i in np.flatnonzero(indeg == 0)]
    while avail:
        i = avail.pop(0)
        order.append(i)
        for j in np.flatnonzero(E[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                avail.append(int(j))
    if len(order) != W.shape[0]:
        raise ValueError('W is not acyclic; cannot form a topological order')
    return np.array(order)


def order_agreement(W, order):
    """Fraction of true edges i->j for which `order` puts i before j."""
    pos = np.empty(len(order), dtype=int)
    pos[order] = np.arange(len(order))
    src, dst = np.nonzero(W != 0)
    return float((pos[src] < pos[dst]).mean()) if len(src) else np.nan


def one_file(path, thresh=THRESH, seed=0):
    f = np.load(path, allow_pickle=True)
    X, W = np.asarray(f['X'], dtype=float), np.asarray(f['W'], dtype=float)
    L = np.log1p(X)
    var_order = np.argsort(np.var(L, axis=0))
    true_order = topological_order(W)
    rand_order = np.random.default_rng(seed).permutation(W.shape[0])

    setting, repl = parse_tag(os.path.basename(path))
    common = dict(file=os.path.basename(path), setting=setting, repl=repl,
                  varsortability=varsortability(L, W))
    rows = []
    for name, order in [('randomsort', rand_order),
                        ('sortnregress', var_order),
                        ('oracleorder', true_order)]:
        rec = dict(common, method=name, order_agreement=order_agreement(W, order))
        rec.update(score_edges(sortnregress(L, order=order), W, thresh=thresh))
        rows.append(rec)
    return rows


def attach_rice(t, resultdir):
    """Pull RICE's row for the same setting and replicate from the fitting output."""
    import glob
    out = []
    for f in glob.glob(os.path.join(resultdir, '*.csv')):
        s, r = parse_tag(os.path.basename(f))
        if s == -1:
            continue
        d = pd.read_csv(f)
        if not {'method', 'measure', 'value'} <= set(d.columns):
            continue
        sub = d[d.method.astype(str).str.upper().str.startswith('RICE')]
        piv = sub.pivot_table(index='method', columns='measure', values='value')
        for meth, row in piv.iterrows():
            out.append(dict(file=os.path.basename(f), setting=s, repl=r, method=str(meth),
                            order_agreement=np.nan, varsortability=np.nan,
                            ppv=row.get('precision', np.nan), tpr=row.get('recall', np.nan),
                            mcc=row.get('mcc', np.nan), shd=row.get('shd', np.nan),
                            scorer='from result CSV, threshold set by its own driver'))
    return pd.concat([t, pd.DataFrame(out)], ignore_index=True) if out else t


def selftest():
    """A chain has variance 1, 2, 3, ... along the path, so varsortability is exactly 1 and
    the variance order IS the topological order; standardising sends varsortability to
    0.5 while leaving the true order unchanged."""
    rng = np.random.default_rng(0)
    p, n = 6, 50_000
    W = np.zeros((p, p))
    for j in range(1, p):
        W[j - 1, j] = 1.0
    e = rng.normal(size=(n, p))
    X = np.zeros((n, p))
    X[:, 0] = e[:, 0]
    for j in range(1, p):
        X[:, j] = X[:, j - 1] + e[:, j]
    vs = varsortability(X, W)
    Z = (X - X.mean(0)) / X.std(0)
    top = topological_order(W)
    var_order = np.argsort(np.var(X, axis=0))
    print(f'[selftest] chain varsortability {vs:.4f} (expect 1.0000), '
          f'standardised {varsortability(Z, W):.4f} (expect ~0.5000)')
    print(f'[selftest] topological order {list(top)}, variance order {list(var_order)}, '
          f'order_agreement {order_agreement(W, var_order):.4f} (expect 1.0000)')
    assert abs(vs - 1.0) < 1e-9, vs
    assert abs(varsortability(Z, W) - 0.5) < 0.02
    assert list(top) == list(range(p)) and order_agreement(W, var_order) == 1.0
    print('[selftest] OK | scorer:',
          'utils.measurement' if _HAVE_UTILS else 'FALLBACK (set SCGRAPH_DIR)')


def main(argv):
    resultdir = None
    if '--rice-csv' in argv:
        k = argv.index('--rice-csv')
        resultdir = argv[k + 1]
        argv = argv[:k] + argv[k + 2:]
    paths = [a for a in argv if not a.startswith('--')]
    if '--selftest' in argv or not paths:
        selftest()
        if not paths:
            return
    if not _HAVE_UTILS:
        print('WARNING: utils.measurement not importable, so every score below uses the '
              'fallback and is NOT comparable with the published numbers. Set SCGRAPH_DIR '
              'to the directory holding utils.py and rerun.')
    rows = []
    for p_ in paths:
        rows.extend(one_file(p_))
        print('.', end='', flush=True)
    print()
    t = pd.DataFrame(rows)
    if resultdir:
        t = attach_rice(t, resultdir)
    if (t.setting == -1).any():
        bad = sorted(t.file[t.setting == -1].unique())[:3]
        print(f'WARNING: no setting tag parsed from {int((t.setting == -1).sum())} row(s), '
              f'e.g. {bad} -- pooled under setting -1. Fix SETTING_RE to match your names.')
    t.to_csv('order_sufficiency_per_file.csv', index=False)

    summ = (t.groupby(['setting', 'method'])[['varsortability', 'order_agreement',
                                              'ppv', 'tpr', 'mcc', 'shd']]
            .mean().round(3))
    summ.to_csv('order_sufficiency_summary.csv')
    print(summ.to_string())
    print('\nwrote order_sufficiency_per_file.csv and order_sufficiency_summary.csv')
    print('   scorer:', sorted(t.scorer.dropna().unique()))
    for s_, g_ in t.groupby('setting'):
        orc = g_[g_.method == 'oracleorder'].mcc.mean()
        srt = g_[g_.method == 'sortnregress'].mcc.mean()
        rnd = g_[g_.method == 'randomsort'].mcc.mean()
        print(f'  setting {s_}: MCC random {rnd:.3f} -> variance order {srt:.3f} -> '
              f'TRUE order {orc:.3f}. Any method whose MCC exceeds {orc:.3f} is not '
              f'living off the ordering.')


if __name__ == '__main__':
    main(sys.argv[1:])
