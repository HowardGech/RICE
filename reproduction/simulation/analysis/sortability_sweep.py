#!/usr/bin/env python
"""
Complete sortability / order-sufficiency result over the stored confounding sweep.

Covers both intervention regimes, every confounding setting and every replicate that
exists, skipping the ones that do not. Nothing is re-simulated and RICE is not refitted --
only the stored .npz files are read, so this can be compared against the RICE numbers you
already have.

WHAT IT COMPUTES, per file

  Sortability statistics (cheap, no scikit-learn needed)
    vs_counts   varsortability on the raw counts
    vs_log1p    varsortability on log(1 + X), the predictor the model is given
    vs_std      varsortability on column-standardised log(1 + X)   -- must come back ~0.5
    r2s_log1p   R^2-sortability on log(1 + X), the scale-invariant analogue
  computed over all cells and over the control cells separately, because varsortability is
  defined on an observational distribution while these files also contain perturbed cells
  whose shifted means inflate marginal variance. They are usually close; quote the one you
  name.

  Order-sufficiency baselines (needs scikit-learn; skip with --skip-baselines)
    randomsort      a random permutation              -- floor
    sortnregress    nodes ordered by marginal variance -- what varsortability actually buys
    oracleorder     the TRUE topological order of W   -- upper bound for ANY ordering method
  One and the same edge-selection procedure under three orderings, nothing else changed.
  oracleorder is the argument: no method can extract more from ordering information than it
  would get from the correct order handed to it for free, so a method scoring above
  oracleorder is not living off the variance ladder, however high varsortability is.
  Also reported is order_agreement, the fraction of true edges whose endpoints an ordering
  places in the correct direction -- how much of the graph the ordering even permits.

USAGE

    python sortability_sweep.py --selftest
    python sortability_sweep.py --jobs 8
    python sortability_sweep.py --skip-baselines --jobs 8
    python sortability_sweep.py --root /other/path --model NB --jobs 8

Run it on a compute node, not a login node -- with baselines this is ~p regressions per
ordering per file. --jobs defaults to 1 for that reason. Rows are appended to the output
CSVs as each file finishes, so an interrupted run keeps what it had.

FILENAME TAGS. The sweep is written with a `_cova{d}_` tag for some settings and `_um{d}_`
for others, so both are tried for every (setting, replicate) and a glob is used as a last
resort. A setting with no file under either spelling is reported as missing rather than
silently dropped.

SCORING. Baseline edge sets are scored with the project's own measurement(), after the
same threshold-then-project-to-a-DAG step job_simu.py applies to RICE:
    W_est[|W_est| <= THRESH] = 0; W_est = to_dag(W_est); measurement(W_true, W_est)
Scoring them any other way makes the comparison not like-for-like -- SHD there comes from
MetricsDAG, which counts a reversed edge once, whereas a hand-rolled fp + fn + reversals
double-counts it. THRESH must match the w_threshold the RICE run used. If utils.py is not
importable the fallback scorer runs, every row is tagged FALLBACK-not-comparable and the
script says so; do not place those numbers next to published ones.

This file is self-contained: every helper is defined here.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sortability import parse_tag, varsortability, r2sortability, sortnregress
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
ROOT = 'result'
SCGRAPH_DIR = os.environ.get('SCGRAPH_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
INTERVENTIONS = ['hard', 'soft']
COVA = [2, 4, 6, 8, 10, 20]
REPL = list(range(1, 21))
MODEL = 'NB'
N, P, S0, GRAPH = 10_000, 100, 400, 'ER'
THRESH = 0.3          # must match the w_threshold used for the RICE results being compared

# ONE SPELLING ONLY, set by --tag (default 'cova'). Accepting several spellings is what
# let `_um20_` files -- which belong to a different experiment -- fill the four cova=20
# grid points that have no `_cova20_` file, so the grid looked complete while two
# generators were being mixed into one curve. A grid point with no file under the chosen
# tag is now reported as missing, which is the correct answer.
TAG = 'cova'
SETTING_RE = re.compile(rf'_{TAG}(\d+)(?=_|\.)')
REPL_RE = re.compile(r'repl(\d+)')


def set_tag(tag):
    global TAG, SETTING_RE
    TAG = tag
    SETTING_RE = re.compile(rf'_{re.escape(tag)}(\d+)(?=_|\.)')

try:
    sys.path.insert(0, SCGRAPH_DIR)
    from utils import measurement, to_dag
    _HAVE_UTILS = True
except Exception:                                    # noqa: BLE001
    _HAVE_UTILS = False


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def resolve_path(root, intervention, model, cova, repl):
    """Exact `_{TAG}{cova}_repl{repl}.npz`, then a glob restricted to the same tag.
    Returns the path or None."""
    d = os.path.join(root, f'simu_{intervention}', 'data', model)
    stem = f'n{N // 1000}k_p{P}_s{S0}_{GRAPH}'
    cand = os.path.join(d, f'{stem}_{TAG}{cova}_repl{repl}.npz')
    if os.path.exists(cand):
        return cand
    # glob only within the chosen spelling, so a file tagged differently cannot stand in
    for cand in sorted(glob.glob(os.path.join(d, f'{stem}_{TAG}{cova}_*repl{repl}.npz'))):
        s, r = parse_tag(os.path.basename(cand))
        if s == cova and r == repl:
            return cand
    return None


def score_edges(W_hat, W_true, thresh=THRESH):
    if _HAVE_UTILS:
        W_est = W_hat.copy()
        W_est[np.abs(W_est) <= thresh] = 0.0
        W_est = to_dag(W_est)
        acc = measurement(W_true, W_est)
        return dict(ppv=acc['precision'], tpr=acc['recall'], mcc=acc['mcc'],
                    shd=acc['shd'], scorer='utils.measurement')
    P_ = np.abs(W_hat) > thresh
    T = W_true != 0
    tp = int((P_ & T).sum()); fp = int((P_ & ~T).sum()); fn = int((~P_ & T).sum())
    tn = int((~P_ & ~T).sum())
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(ppv=tp / (tp + fp) if tp + fp else np.nan,
                tpr=tp / (tp + fn) if tp + fn else np.nan,
                mcc=((tp * tn - fp * fn) / den) if den > 0 else np.nan,
                shd=int(fp + fn), scorer='FALLBACK-not-comparable')


def topological_order(W):
    """Kahn's algorithm on W[i, j] != 0 meaning i -> j."""
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
    pos = np.empty(len(order), dtype=int)
    pos[order] = np.arange(len(order))
    src, dst = np.nonzero(W != 0)
    return float((pos[src] < pos[dst]).mean()) if len(src) else np.nan


# --------------------------------------------------------------------------------------
# per-file work
# --------------------------------------------------------------------------------------
def check_counts(X):
    """Screen the stored count matrix before any arithmetic touches it.

    A NEGATIVE or NaN entry is not a numerical nuisance: log1p is NaN below -1, which then
    propagates into the least-squares fits as `SVD did not converge`. Negative counts come
    from the integer overflow in the published rescale -- the cap and the mean floor are
    applied as two sequential shifts, the second can undo the first, and exp() of the
    result exceeds the integer range, saturating on some platforms and wrapping negative on
    others. Such a replicate must be regenerated (patch_count_rescale.py), not analysed; and if
    it is also present in the fitted results, those carry the same artefact.
    """
    n_nan = int(np.isnan(X).sum())
    n_neg = int((X < 0).sum())
    if n_nan or n_neg:
        finite = X[np.isfinite(X)]
        return False, 'invalid counts (negative or NaN)', dict(
            n_nan=n_nan, n_neg=n_neg,
            min_value=float(finite.min()) if finite.size else float('nan'),
            max_value=float(finite.max()) if finite.size else float('nan'),
            n_genes_affected=int((~np.isfinite(X) | (X < 0)).any(axis=0).sum()))
    return True, '', {}


def process(job):
    """job = (path, intervention, cova, repl, skip_baselines).

    Returns (stat_rows, base_rows, fail_rows). Never raises: a file that cannot be
    processed comes back as a fail row instead, because an exception raised inside a worker
    propagates through fut.result() and takes down the whole sweep.
    """
    path, intervention, cova, repl, skip_baselines = job
    key0 = dict(file=os.path.basename(path), intervention=intervention,
                cova=cova, repl=repl)
    try:
        return _process_inner(job)
    except Exception as exc:                                   # noqa: BLE001
        return [], [], [dict(key0, reason=f'{type(exc).__name__}: {exc}')]


def _process_inner(job):
    path, intervention, cova, repl, skip_baselines = job
    f = np.load(path, allow_pickle=True)
    X = np.asarray(f['X'], dtype=float)
    W = np.asarray(f['W'], dtype=float)
    mask = np.asarray(f['mask']) if 'mask' in f.files else None
    key = dict(file=os.path.basename(path), intervention=intervention,
               cova=cova, repl=repl)

    ok, reason, diag = check_counts(X)
    if not ok:
        return [], [], [dict(key, reason=reason, **diag)]
    if not np.all(np.isfinite(W)):
        return [], [], [dict(key, reason='W contains non-finite entries')]

    stat_rows = []
    populations = [('all', np.ones(X.shape[0], dtype=bool))]
    if mask is not None:
        populations.append(('control', np.asarray(mask).sum(1) == 0))
    for label, sel in populations:
        if sel.sum() < 50:
            continue
        Xs = X[sel]
        L = np.log1p(Xs)
        Z = (L - L.mean(0)) / (L.std(0) + 1e-12)
        stat_rows.append(dict(key, cells=label, n_cells=int(sel.sum()),
                              vs_counts=varsortability(Xs, W),
                              vs_log1p=varsortability(L, W),
                              vs_std=varsortability(Z, W),
                              r2s_log1p=r2sortability(L, W)))

    base_rows = []
    if not skip_baselines:
        L = np.log1p(X)
        orders = [('randomsort', np.random.default_rng(repl).permutation(W.shape[0])),
                  ('sortnregress', np.argsort(np.var(L, axis=0))),
                  ('oracleorder', topological_order(W))]
        for name, order in orders:
            rec = dict(key, method=name, varsortability=varsortability(L, W),
                       order_agreement=order_agreement(W, order))
            rec.update(score_edges(sortnregress(L, order=order), W))
            base_rows.append(rec)
    return stat_rows, base_rows, []


def selftest():
    """A chain has variance 1, 2, 3, ... along the path, so varsortability is exactly 1 and
    the variance order IS the topological order; standardising sends it to 0.5."""
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
    vs, vs_z = varsortability(X, W), varsortability((X - X.mean(0)) / X.std(0), W)
    top, vo = topological_order(W), np.argsort(np.var(X, axis=0))
    print(f'[selftest] varsortability {vs:.4f} (expect 1.0000), '
          f'standardised {vs_z:.4f} (expect ~0.5000), '
          f'r2-sortability {r2sortability(X, W):.4f}')
    print(f'[selftest] topological order {list(top)} == variance order {list(vo)}, '
          f'order_agreement {order_agreement(W, vo):.4f} (expect 1.0000)')
    assert abs(vs - 1.0) < 1e-9 and abs(vs_z - 0.5) < 0.02
    assert list(top) == list(range(p)) and order_agreement(W, vo) == 1.0
    print('[selftest] OK | scorer:',
          'utils.measurement' if _HAVE_UTILS else f'FALLBACK (utils.py not found in {SCGRAPH_DIR})')


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--tag', default=TAG,
                    help="filename tag carrying the setting, e.g. 'cova' or 'um'; "
                         'only this spelling is accepted')
    ap.add_argument('--interventions', default=','.join(INTERVENTIONS))
    ap.add_argument('--cova', default=','.join(map(str, COVA)))
    ap.add_argument('--repl', default=f'1-{REPL[-1]}')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--skip-baselines', action='store_true')
    ap.add_argument('--out-dir', default='.',
                    help='directory for every output CSV (created if absent). Defaults to '
                         'the working directory, which under LSF is wherever bsub was run '
                         'from -- set it explicitly for anything you want to keep.')
    ap.add_argument('--out-prefix', default='sortability')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='resolve the grid and print which file each point mapped to, '
                         'then stop without computing anything')
    a = ap.parse_args()

    set_tag(a.tag)
    selftest()
    if a.selftest:
        return
    if not _HAVE_UTILS and not a.skip_baselines:
        print('WARNING: utils.measurement not importable, so the baseline scores are the '
              'fallback and are NOT comparable with the published numbers. Set SCGRAPH_DIR '
              'and rerun.')

    os.makedirs(a.out_dir, exist_ok=True)
    a.out_prefix = os.path.join(a.out_dir, os.path.basename(a.out_prefix))
    print(f'output directory: {os.path.abspath(a.out_dir)}')

    interventions = a.interventions.split(',')
    cova = [int(c) for c in a.cova.split(',')]
    lo, hi = (a.repl.split('-') + [a.repl])[:2]
    repls = list(range(int(lo), int(hi) + 1))

    jobs, missing = [], []
    for itv in interventions:
        for c in cova:
            for r in repls:
                p_ = resolve_path(a.root, itv, a.model, c, r)
                (jobs if p_ else missing).append(
                    (p_, itv, c, r, a.skip_baselines) if p_ else (itv, c, r))
    # A grid point must map to a file whose OWN tag matches it. If the glob fallback ever
    # returns a file from another setting, two grid points end up claiming one path and the
    # inventory looks like a complete grid -- which is indistinguishable from "all files
    # present" unless it is checked. Fail loudly instead.
    seen = {}
    bad = []
    for p_, itv, c, r, _ in jobs:
        s_tag, r_tag = parse_tag(os.path.basename(p_))
        if (s_tag, r_tag) != (c, r):
            bad.append((itv, c, r, os.path.basename(p_), s_tag, r_tag))
        if p_ in seen:
            bad.append((itv, c, r, os.path.basename(p_), *seen[p_]))
        seen[p_] = (c, r)
    if bad:
        print('ERROR: grid points resolved to files whose tags do not match, or two points '
              'resolved to the same file:')
        for row in bad[:20]:
            print('   ', row)
        raise SystemExit('resolution is unsound -- fix SETTING_RE or the filename stem')

    inv = (pd.DataFrame([(i, c, r) for _, i, c, r, _ in jobs],
                        columns=['intervention', 'cova', 'repl'])
           .groupby(['intervention', 'cova']).size().rename('n_files'))
    print(f'{len(jobs)} files found, {len(missing)} of '
          f'{len(interventions) * len(cova) * len(repls)} grid points missing')
    print(inv.to_string())
    if not jobs:
        print('nothing to do -- check --root and the filename stem')
        return
    pd.DataFrame(missing, columns=['intervention', 'cova', 'repl']).to_csv(
        f'{a.out_prefix}_missing.csv', index=False)

    res = pd.DataFrame([(i, c, r, os.path.basename(p_),
                         'exact' if p_.endswith(f'_{a.tag}{c}_repl{r}.npz') else 'glob')
                        for p_, i, c, r, _ in jobs],
                       columns=['intervention', 'cova', 'repl', 'resolved_file', 'via'])
    res.to_csv(f'{a.out_prefix}_resolved.csv', index=False)
    print('\ntag spelling actually used:')
    print(res.groupby(['intervention', 'cova', 'via']).size().rename('n').to_string())
    if a.dry_run:
        print(f'\ndry run: wrote {a.out_prefix}_resolved.csv and {a.out_prefix}_missing.csv; '
              f'{len(set(p_ for p_, *_ in jobs))} distinct paths for {len(jobs)} grid points')
        return

    stat_csv = f'{a.out_prefix}_per_file.csv'
    base_csv = f'{a.out_prefix}_baselines.csv'
    fail_csv = f'{a.out_prefix}_failed.csv'
    for fn in (stat_csv, base_csv, fail_csv):
        if os.path.exists(fn):
            os.remove(fn)

    def emit(rows, fn):
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(fn, mode='a', header=not os.path.exists(fn), index=False)

    done = 0
    ex = None
    if a.jobs > 1:
        # Some sandboxes and container configurations refuse the semaphores a process pool
        # needs (PermissionError on SC_SEM_NSEMS_MAX). Fall back to serial with a message
        # rather than aborting a run that is otherwise fine.
        try:
            from concurrent.futures import ProcessPoolExecutor
            ex = ProcessPoolExecutor(max_workers=a.jobs)
        except (OSError, NotImplementedError, ValueError) as exc:
            print(f'NOTE: --jobs {a.jobs} unavailable here ({type(exc).__name__}: {exc}); '
                  f'running serially.')
            ex = None
    if ex is not None:
        from concurrent.futures import as_completed
        # cancel_futures on the way out, so an interrupt does not hang in shutdown(wait=True)
        try:
            futs = [ex.submit(process, j) for j in jobs]
            for fut in as_completed(futs):
                try:
                    s_, b_, f_ = fut.result()
                except Exception as exc:              # noqa: BLE001 -- worker died outright
                    s_, b_, f_ = [], [], [dict(file='<unknown>', intervention='', cova=-1,
                                               repl=-1,
                                               reason=f'worker failed: {type(exc).__name__}: {exc}')]
                emit(s_, stat_csv); emit(b_, base_csv); emit(f_, fail_csv)
                done += 1
                print(f'\r{done}/{len(jobs)}', end='', flush=True)
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    else:
        for j in jobs:
            s_, b_, f_ = process(j)
            emit(s_, stat_csv); emit(b_, base_csv); emit(f_, fail_csv)
            done += 1
            print(f'\r{done}/{len(jobs)}', end='', flush=True)
    print()

    if os.path.exists(fail_csv):
        fail = pd.read_csv(fail_csv)
        print(f'\n=== {len(fail)} file(s) SKIPPED ===')
        cols = [c for c in ['intervention', 'cova', 'repl', 'reason', 'n_neg', 'n_nan',
                            'min_value', 'n_genes_affected'] if c in fail.columns]
        print(fail[cols].to_string(index=False))
        if fail.reason.astype(str).str.contains('invalid counts').any():
            print('   Negative or NaN counts are the integer-overflow artefact in the '
                  'rescale, not a numerical nuisance: regenerate those replicates with '
                  'patch_count_rescale.py applied, and check whether the same replicates are in '
                  'the fitted results, which would carry the artefact too.')

    if not os.path.exists(stat_csv):
        print('no file produced any statistics -- nothing to summarise')
        return
    stats = pd.read_csv(stat_csv)
    summ = (stats.groupby(['intervention', 'cells', 'cova'])
            [['vs_counts', 'vs_log1p', 'vs_std', 'r2s_log1p']]
            .agg(['mean', 'std', 'size']).round(3))
    summ.to_csv(f'{a.out_prefix}_summary.csv')
    print('\n=== sortability statistics ===')
    print(stats.groupby(['intervention', 'cells', 'cova'])
          [['vs_counts', 'vs_log1p', 'vs_std', 'r2s_log1p']].mean().round(3).to_string())

    if os.path.exists(base_csv):
        base = pd.read_csv(base_csv)
        bsum = (base.groupby(['intervention', 'cova', 'method'])
                [['order_agreement', 'ppv', 'tpr', 'mcc', 'shd']]
                .agg(['mean', 'std', 'size']).round(3))
        bsum.to_csv(f'{a.out_prefix}_baselines_summary.csv')
        print('\n=== order-sufficiency baselines ===')
        print(base.groupby(['intervention', 'cova', 'method'])
              [['order_agreement', 'ppv', 'tpr', 'mcc', 'shd']].mean().round(3).to_string())
        print('\n=== ceiling for any ordering-based method (mean MCC of oracleorder) ===')
        for (itv, c), g in base.groupby(['intervention', 'cova']):
            o = g[g.method == 'oracleorder'].mcc.mean()
            s = g[g.method == 'sortnregress'].mcc.mean()
            r = g[g.method == 'randomsort'].mcc.mean()
            print(f'  {itv:<5} cova {c:>2}: random {r:.3f} -> variance order {s:.3f} -> '
                  f'TRUE order {o:.3f}  (RICE above {o:.3f} is not using the ordering)')
        print('   scorer:', sorted(base.scorer.dropna().unique()))
    written = [stat_csv, f'{a.out_prefix}_summary.csv', f'{a.out_prefix}_missing.csv',
               f'{a.out_prefix}_resolved.csv']
    if os.path.exists(base_csv):
        written += [base_csv, f'{a.out_prefix}_baselines_summary.csv']
    if os.path.exists(fail_csv):
        written.append(fail_csv)
    print('\nwrote, in ' + os.path.abspath(a.out_dir) + ':')
    for w in written:
        print('   ' + os.path.basename(w))


if __name__ == '__main__':
    main()
