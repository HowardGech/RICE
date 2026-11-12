r"""Realised dependence per selection strength, for the brackets under the x axis.

The generator stores corr_perturbed_u in each .npz's info dict. This reads it AND
recomputes it from the saved mask and Z, then checks the two agree: one is what the
generator recorded at write time, the other is what the arrays actually contain, so a
disagreement means the files do not hold what their diagnostics claim.
"""
import glob, json, os, re
import numpy as np, pandas as pd

DATADIR = 'result/simu_soft/data/NB'
PATTERN = '*_sel*_repl*.npz'
TOL     = 1e-6          # allowed |stored - recomputed|

# The sel### tag sits between the factor count and _repl, so the setting and the replicate
# are matched SEPARATELY; one pattern requiring _repl right after the digits would match
# nothing and pool every file under a single setting.
SEL_RE, REPL_RE = re.compile(r'_sel(\d+)'), re.compile(r'_repl(\d+)')

rows = []
for pth in sorted(glob.glob(os.path.join(DATADIR, PATTERN))):
    b = os.path.basename(pth)
    s, r = SEL_RE.search(b), REPL_RE.search(b)
    if not (s and r):
        print(f'[warn] cannot parse the tag from {b}; skipped')
        continue
    f = np.load(pth, allow_pickle=True)
    info = json.loads(str(f['info'])) if 'info' in f.files else {}
    # the generator prepends an intercept column of ones, so the first latent factor is
    # column 1 -- detect rather than assume, so intercept=False runs still work
    Z = f['Z']
    zz = Z[:, 1 if np.allclose(Z[:, 0], 1.0) else 0].astype(float)
    zz = (zz - zz.mean()) / (zz.std() + 1e-12)
    pert = (f['mask'].sum(axis=1) > 0).astype(float)
    rows.append(dict(kappa=int(s.group(1)) / 100, replicate=int(r.group(1)),
                     stored=float(info.get('corr_perturbed_u', np.nan)),
                     recomputed=float(np.corrcoef(pert, zz)[0, 1]),
                     apply_clip=info.get('apply_clip'), file=b))

sel = pd.DataFrame(rows)
assert len(sel), f'no files matching {PATTERN} under {DATADIR}'

clips = set(sel.apply_clip.dropna())
if len(clips) > 1:
    print(f'[warn] files were generated with DIFFERENT clip settings {clips}; patched and '
          f'unpatched runs are not comparable and must not be pooled')

bad = sel[sel.stored.notna() & ((sel.stored - sel.recomputed).abs() > TOL)]
assert bad.empty, ('stored and recomputed disagree:\n'
                   + bad[['file', 'stored', 'recomputed']].to_string(index=False))

agg = (sel.groupby('kappa')['recomputed']
       .agg(n='size', mean='mean', sd='std', min='min', max='max').round(3))
print(agg.to_string())

# 0.585 = phi(z_q)/sqrt(q(1-q)) at q = 0.9: exact for this design, independent of n and p
CEILING = 0.585
print(f'\nceiling at a 90% perturbation rate: {CEILING} '
      f'(top setting reaches {100*agg["mean"].max()/CEILING:.0f}% of it)')
print('bracket row:', '  '.join(f'{m:.2f}' for m in agg['mean']))
sel.to_csv('selection_corr.csv', index=False)
print(f'\nstored and recomputed agree on all {len(sel)} files; wrote selection_corr.csv')
