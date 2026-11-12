#!/usr/bin/env python
"""
Sortability statistics for the stored simulation files.

Reports, per .npz and aggregated per confounding setting:

  vs_counts   varsortability on the raw counts
  vs_log1p    varsortability on log(1 + X), the predictor the model actually sees
  vs_std      varsortability on column-standardised log(1 + X)  (sanity: must be ~0.5)
  r2s_log1p   R^2-sortability on log(1 + X), the scale-invariant analogue

VARSORTABILITY (Reisach et al. 2021) is the fraction of DIRECTED PATHS -- edges and
longer paths alike -- along which marginal variance increases, ties counted as one half.
0.5 is chance; 1.0 means sorting variables by variance reproduces the topological order
exactly, so the order is recoverable without any causal reasoning. R^2-SORTABILITY
(Reisach et al. 2023) replaces the variance of each node with the R^2 of that node
regressed on all the others, which is invariant to rescaling any single variable and is
therefore the statistic to quote when the objection is about units.

WHICH CELLS. Varsortability is defined on an observational distribution, but a Perturb-seq
simulation also contains perturbed cells whose shifted means inflate marginal variance.
Both are reported: `cells=all` matches what the methods are given and reproduces the
figures quoted in the response; `cells=control` is the statistic on the observational
distribution alone. They are usually close, but if they diverge quote the one you name.

    python sortability_stats.py path/to/*.npz
    python sortability_stats.py --sortnregress path/to/*.npz

The --sortnregress flag additionally runs the two trivial ordering baselines, which is the
decisive test rather than the statistic: sortnregress orders nodes by marginal variance and
regresses each on its predecessors under an L1 penalty, and randomsort does the same under
a random order. A method that beats sortnregress at high varsortability is not living off
the variance ladder. These need scikit-learn; the statistics above do not.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sortability import parse_tag, varsortability, r2sortability, sortnregress
import os
import re
import sys

import numpy as np
import pandas as pd

# The setting and the replicate are matched SEPARATELY, because the generator can write
# an extra tag between them (job_gen_cova.py appends a generator tag such as _hm, giving
# ..._um10_hm_repl1.npz). A single pattern requiring _repl immediately after the digits
# silently fails on those and every file lands in one pooled group reported as setting -1,
# which looks like a result rather than a parse failure.
SETTING_RE = re.compile(r'_(?:um|cova|d)(\d+)(?=_|\.)')
REPL_RE = re.compile(r'repl(\d+)')


# Baseline edge sets are scored with the PROJECT'S OWN measurement(), after the same
# threshold-then-project-to-a-DAG step job_simu.py applies to RICE:
#     W_est = result(w_threshold=0.3)['W']; W_est = to_dag(W_est); measurement(W_true, W_est)
# Scoring them any other way makes the comparison not like-for-like -- in particular SHD
# there comes from MetricsDAG, which counts a reversed edge once, whereas a hand-rolled
# fp + fn + reversals double-counts it. If utils is not importable the fallback below runs
# and says so, but do not put fallback numbers next to published ones.
try:
    sys.path.insert(0, os.environ.get('SCGRAPH_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
    from utils import measurement, to_dag
    _HAVE_UTILS = True
except Exception:                                    # noqa: BLE001
    _HAVE_UTILS = False


def _metrics(W_hat, W_true, thresh=0.3):
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


def one_file(path, run_baselines=False):
    f = np.load(path, allow_pickle=True)
    X, W = np.asarray(f['X'], dtype=float), np.asarray(f['W'], dtype=float)
    mask = np.asarray(f['mask']) if 'mask' in f.files else None
    rows = []
    for label, idx in [('all', slice(None)),
                       ('control', (np.asarray(mask).sum(1) == 0) if mask is not None else None)]:
        if idx is None:
            continue
        Xs = X[idx]
        if Xs.shape[0] < 50:
            continue
        L = np.log1p(Xs)
        Z = (L - L.mean(0)) / (L.std(0) + 1e-12)
        rec = dict(file=os.path.basename(path), cells=label, n_cells=int(Xs.shape[0]),
                   vs_counts=varsortability(Xs, W),
                   vs_log1p=varsortability(L, W),
                   vs_std=varsortability(Z, W),
                   r2s_log1p=r2sortability(L, W))
        if run_baselines and label == 'all':
            rec.update({f'sortnregress_{k}': v
                        for k, v in _metrics(sortnregress(L), W).items()})
            rng = np.random.default_rng(0)
            rec.update({f'randomsort_{k}': v for k, v in
                        _metrics(sortnregress(L, order=rng.permutation(W.shape[0])), W).items()})
        rows.append(rec)
    return rows


def selftest():
    """A chain with unit weights and unit noise has variance 1, 2, 3, ... along the path,
    so varsortability is exactly 1; standardising must send it to 0.5."""
    rng = np.random.default_rng(0)
    p, n = 6, 50_000
    W = np.zeros((p, p))
    for j in range(1, p):
        W[j - 1, j] = 1.0
    e = rng.normal(size=(n, p))
    Xc = np.zeros((n, p))
    Xc[:, 0] = e[:, 0]
    for j in range(1, p):
        Xc[:, j] = Xc[:, j - 1] + e[:, j]
    vs = varsortability(Xc, W)
    Z = (Xc - Xc.mean(0)) / Xc.std(0)
    vs_z = varsortability(Z, W)
    print(f'[selftest] chain varsortability {vs:.4f} (expect 1.0000), '
          f'standardised {vs_z:.4f} (expect ~0.5000)')
    assert abs(vs - 1.0) < 1e-9, vs
    assert abs(vs_z - 0.5) < 0.02, vs_z
    print('[selftest] OK')


def main(argv):
    run_baselines = '--sortnregress' in argv
    paths = [a for a in argv if not a.startswith('--')]
    if '--selftest' in argv or not paths:
        selftest()
        if not paths:
            return
    if run_baselines and not _HAVE_UTILS:
        print('WARNING: utils.measurement not importable, so baseline PPV/TPR/MCC/SHD use '
              'the fallback scorer and are NOT comparable with the published numbers. '
              'Set SCGRAPH_DIR to the directory holding utils.py and rerun.')
    rows = []
    for p_ in paths:
        rows.extend(one_file(p_, run_baselines=run_baselines))
        print('.', end='', flush=True)
    print()
    t = pd.DataFrame(rows)
    parsed = [parse_tag(f) for f in t.file]
    t['setting'] = [a for a, _ in parsed]
    t['repl'] = [b for _, b in parsed]
    if (t.setting == -1).any():
        bad = sorted(t.file[t.setting == -1].unique())[:3]
        print(f'WARNING: no setting tag parsed from {int((t.setting == -1).sum())} row(s), '
              f'e.g. {bad} -- these are pooled under setting -1 and the per-setting '
              f'breakdown is NOT available for them. Fix SETTING_RE to match your names.')
    t.to_csv('sortability_per_file.csv', index=False)
    # numeric only -- the *_scorer columns are labels and would break the aggregation
    cols = [c for c in t.columns
            if c.startswith(('vs_', 'r2s_', 'sortnregress_', 'randomsort_'))
            and pd.api.types.is_numeric_dtype(t[c])]
    summ = (t.groupby(['cells', 'setting'])[cols]
            .agg(['mean', 'min', 'max']).round(3))
    summ.to_csv('sortability_summary.csv')
    print(f'wrote sortability_per_file.csv ({len(t)} rows) and sortability_summary.csv')
    for c in [c for c in t.columns if c.endswith('_scorer')]:
        print(f'   {c}: {sorted(t[c].dropna().unique())}')
    print(t.groupby(['cells', 'setting'])[['vs_counts', 'vs_log1p', 'vs_std', 'r2s_log1p']]
          .mean().round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1:])
