r"""How confounded is this screen, on the scale of the simulated latent-factor sweep?

Uses the objects cell 19 already built: data_X_cont_norm_test, perturbed_rows,
unperturbed_rows. Nothing is recomputed and nothing is imported from project files.

WHAT IT CLAIMS. The confounding-to-perturbation ratio is NOT identifiable in real data
-- shared variation among genes is part unmeasured confounding and part the regulatory
network, and the data alone cannot separate them. So this does not measure the ratio.
It runs one estimator here and compares against the SAME estimator run on the simulated
sweep, and reports which simulated d this screen matches. On the simulation, where the
truth is known, the estimator inflates the true ratio about 2-fold and not by a
constant factor, so its absolute value is not interpretable -- only its position.

  numerator    per-gene SD of the top-K PC reconstruction of CONTROL cells, i.e. shared
               variation that no perturbation explains.
  denominator  per-target |on-target log fold change| = the CRISPRi knockdown. For
               reference, Replogle et al. report a median knockdown of 85.5% (K562) and
               91.6% (RPE1), i.e. |LFC| of 1.93 and 2.48 in natural-log units.
"""
import numpy as np, pandas as pd

K_FACTORS   = 10        # components retained; must match the calibration below
MIN_CELLS   = 10        # minimum cells per target for its LFC to count
COVARIATES  = None      # e.g. total_counts[:, None] to treat depth as nuisance; see note
OUT_PREFIX  = 'confounding'

# Simulated curve. A calibration is only valid at the SHAPE it was generated at -- the
# estimator's floor is set by how much low-rank structure the network alone produces,
# which scales with the gene count. CAL_SHAPE records that shape and the cell refuses to
# interpolate if this dataset differs materially; regenerate with
# estimate_confounding.py --mode calibrate --n <cells> --p <genes> --s0 <edges>.
CAL_SHAPE = dict(n=17383, p=374, s0=1496, scaling='z-scored per gene (cell 4)')
CALIBRATION = pd.DataFrame({
    'd':                  [0,     2,     4,     6,     8,     10,    15,    20   ],
    'true_ratio':         [np.nan, 0.307, 0.453, 0.564, 0.649, 0.727, 0.891, 1.045],
    'est_ratio':          [0.442, 0.659, 0.831, 0.915, 1.088, 1.061, 1.156, 1.201],
    'var_explained_topk': [0.202, 0.354, 0.434, 0.440, 0.486, 0.432, 0.364, 0.301]})

_needed = {'data_X_cont_norm_test': 19, 'perturbed_rows': 19, 'unperturbed_rows': 19}
# dir() must be evaluated HERE, not inside the comprehension: a comprehension has its
# own scope, so dir() called inside it lists the comprehension's locals and would report
# every object as missing even when all are defined.
_defined = set(dir()) | set(globals())
_missing = {k: v for k, v in _needed.items() if k not in _defined}
if _missing:
    raise NameError('missing ' + ', '.join(_missing) +
                    ' -- run notebook cell(s) ' +
                    ', '.join(str(c) for c in sorted(set(_missing.values()))) + ' first')

# --- matrix on a log scale -----------------------------------------------------------
_X = np.asarray(data_X_cont_norm_test, dtype=float)
L = _X if _X.max() < 50 else np.log1p(_X)      # do not log an already-log matrix
print(f'matrix {L.shape}, max {_X.max():.1f} -> treated as '
      f'{"already log-scale" if _X.max() < 50 else "linear, log1p applied"}')

ctrl = np.asarray(unperturbed_rows, dtype=int)
if len(ctrl) < 50:
    raise ValueError(f'only {len(ctrl)} control cells; the numerator needs more')

# --- numerator: shared variation among control cells ---------------------------------
Lc = L[ctrl] - L[ctrl].mean(0)
if COVARIATES is not None:
    C = np.asarray(COVARIATES, dtype=float)[ctrl]
    C = np.hstack([np.ones((len(ctrl), 1)), (C - C.mean(0)) / (C.std(0) + 1e-12)])
    Lc = Lc - C @ np.linalg.lstsq(C, Lc, rcond=None)[0]
k = int(min(K_FACTORS, min(Lc.shape) - 1))
U, s, Vt = np.linalg.svd(Lc, full_matrices=False)
latent_sd = ((U[:, :k] * s[:k]) @ Vt[:k]).std(0)
var_expl = float((s[:k] ** 2).sum() / (s ** 2).sum())

# --- denominator: the on-target knockdown --------------------------------------------
ctrl_mean = L[ctrl].mean(0)
rows = []
for j, idx in perturbed_rows.items():
    j = int(j); idx = np.asarray(idx, dtype=int)
    if j >= L.shape[1] or len(idx) < MIN_CELLS:
        continue
    lfc = abs(L[idx, j].mean() - ctrl_mean[j])
    if lfc > 1e-6:
        rows.append(dict(gene=j, n_cells=len(idx), ontarget_lfc=lfc,
                         latent_sd=latent_sd[j], ratio=latent_sd[j] / lfc))
per_target = pd.DataFrame(rows)
if per_target.empty:
    raise ValueError('no target passed the min_cells / nonzero-LFC filters')
per_target.to_csv(f'{OUT_PREFIX}_per_target.csv', index=False)

# --- report ---------------------------------------------------------------------------
ratio = float(per_target.ratio.median())
print(f'control cells            {len(ctrl)}')
print(f'targets used             {len(per_target)}')
print(f'median on-target |LFC|   {per_target.ontarget_lfc.median():.3f}   '
      f'(Replogle: 1.93 K562, 2.48 RPE1)')
print(f'median latent SD         {np.median(latent_sd):.3f}')
print(f'estimator (not the ratio){ratio:>8.3f}')
print(f'top-{k} PC variance       {var_expl:.3f}')

# Shape guard: a calibration built at a different gene count is not comparable.
_p_obs, _n_obs = L.shape[1], L.shape[0]
if abs(_p_obs - CAL_SHAPE['p']) > 0.05 * CAL_SHAPE['p']:
    raise ValueError(
        f"calibration was generated at p={CAL_SHAPE['p']} genes, n={CAL_SHAPE['n']} cells "
        f"({CAL_SHAPE['scaling']}); this dataset has p={_p_obs}, n={_n_obs}. The "
        f"estimator's floor scales with the gene count, so the lookup would be wrong. "
        f"Regenerate: estimate_confounding.py --mode calibrate "
        f"--n {_n_obs} --p {_p_obs} --s0 {4 * _p_obs}")
if abs(_n_obs - CAL_SHAPE['n']) > 0.25 * CAL_SHAPE['n']:
    print(f"WARNING: calibration n={CAL_SHAPE['n']} vs this dataset n={_n_obs}; "
          f"regenerate if the gap matters")
_c = CALIBRATION.sort_values('est_ratio')
d_hat = float(np.interp(ratio, _c.est_ratio, _c.d))
_s = CALIBRATION[CALIBRATION.d > 0].sort_values('d')
print(f'\n-> matches simulated d ~ {d_hat:.1f}, i.e. a true ratio of about '
      f'{float(np.interp(d_hat, _s.d, _s.true_ratio)):.2f}')
if ratio > CALIBRATION.est_ratio.max():
    print('   NOTE: above the simulated range -- this screen is MORE confounded than the '
          'strongest arm, so d_hat is an extrapolation. Extend the sweep.')
# name the arm actually being compared against, which is the largest d in the table --
# not necessarily d=10 once the sweep is extended
_top = CALIBRATION.loc[CALIBRATION.d.idxmax()]
print(f'   variance check: {var_expl:.3f} here vs {_top.var_explained_topk:.3f} in the '
      f'd={int(_top.d)} arm -> '
      f'{"at least as confounded" if var_expl >= _top.var_explained_topk else "less confounded"}')
