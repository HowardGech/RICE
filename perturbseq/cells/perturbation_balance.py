r"""Is perturbation assignment associated with cell-level covariates, and does effective
uptake depend on them? (Reviewer Comment 5.)

EVERYTHING REPORTED HERE IS A MAGNITUDE, NOT A SIGNIFICANCE. With ~17,000 cells a
hypothesis test answers "is the association exactly zero", which it never is, and
returns a small p for an effect of no consequence. The quantities below instead estimate
HOW LARGE the association is, in units that do not grow with the number of cells, and
each is paired with the smallest value this design could resolve.

The naive effect sizes are not usable for this and are deliberately not the headline:
  eta^2 (variance explained) carries a floor of about (G-1)/(n-1) from group count
        alone. Measured on independent data it FALLS from 0.033 to 0.007 as n goes
        2,000 -> 60,000: it tracks the design, not the association.
  max |SMD| grows with the NUMBER of groups, being a maximum over G draws.
  Cramer's V has the same problem: on independent labels it reads 0.44 at n=2,000 and
        0.14 at n=17,383.

What is reported instead:
  A  omega^2   bias-corrected variance of the covariate explained by perturbation
               identity. Estimates the population value; ~0 under independence at any n.
  B  tau/sd    the between-perturbation standard deviation of the covariate, in units
               of the covariate's own SD -- "a typical perturbation group sits tau SDs
               away from average". This is the magnitude behind the SMD table, with the
               sampling noise removed, and is directly comparable to the 0.1 balance
               convention. Its resolution floor is printed beside it.
  C  V_corr    bias-corrected Cramer's V (Bergsma 2013) for assignment against lane.
  D  knockdown magnitude as a function of each covariate: the change in knockdown per
               SD of the covariate, and the spread across terciles as a PERCENTAGE of
               the knockdown itself. Both are already sample-size stable.

Permutation runs are used only to establish each statistic's resolution floor -- the
largest value the design produces when there is nothing there -- not to compute p.

Needs, all from cell 1: adata_real_inc, mask, gene_name_targeting, data_X, Offset.
"""
import numpy as np
import pandas as pd
from scipy import stats

N_PERM  = 200
CONT    = ['UMI_size', 'mitopercent']
OUT_CSV = f'{ctype}_perturbation_balance.csv'

obs    = adata_real_inc.obs
gid    = obs['gene_id'].values.astype(str)
groups = np.unique(gid)
n      = len(gid)
ctrl   = gid == 'non-targeting'
rng    = np.random.default_rng(0)
rows   = []

print(f'{ctype}: {n} cells, {len(groups)} perturbation groups '
      f'({int(ctrl.sum())} non-targeting), median {int(np.median([(gid == g).sum() for g in groups]))} '
      f'cells per group\n')


def _oneway(x, labels):
    gs = [x[labels == g] for g in np.unique(labels)]
    gs = [a for a in gs if len(a) > 1]
    grand, n_, G_ = x.mean(), len(x), len(gs)
    ss_b = sum(len(a) * (a.mean() - grand) ** 2 for a in gs)
    ss_w = sum(((a - a.mean()) ** 2).sum() for a in gs)
    return ss_b, ss_w, G_, n_, [len(a) for a in gs]


def omega2(x, labels):
    """Bias-corrected variance explained; ~0 under independence at any n."""
    ss_b, ss_w, G_, n_, _ = _oneway(x, labels)
    ms_w = ss_w / (n_ - G_)
    return (ss_b - (G_ - 1) * ms_w) / (ss_b + ss_w + ms_w)


def tau_over_sd(x, labels):
    """Between-group SD of x in units of sd(x): the magnitude behind the SMD table."""
    ss_b, ss_w, G_, n_, ng = _oneway(x, labels)
    ms_b, ms_w = ss_b / (G_ - 1), ss_w / (n_ - G_)
    n0 = (n_ - sum(g ** 2 for g in ng) / n_) / (G_ - 1)
    return np.sqrt(max(0.0, (ms_b - ms_w) / n0)) / x.std(ddof=1)


def cramers_v_corrected(a, b):
    """Bergsma (2013) bias-corrected Cramer's V; ~0 under independence at any n."""
    ct = pd.crosstab(a, b).values
    n_, r, c = ct.sum(), *ct.shape
    chi2 = stats.chi2_contingency(ct)[0]
    phi2 = max(0.0, chi2 / n_ - (r - 1) * (c - 1) / (n_ - 1))
    rc = r - (r - 1) ** 2 / (n_ - 1)
    cc = c - (c - 1) ** 2 / (n_ - 1)
    return np.sqrt(phi2 / max(min(rc - 1, cc - 1), 1e-12))


# --- A and B: how far apart are the perturbation groups, really? ---------------------
print('A/B. association between perturbation identity and each continuous covariate')
print('     (omega^2 = variance explained, tau/sd = typical group offset in covariate SDs)')
for cv in CONT:
    x = obs[cv].values.astype(float)
    w2, tau = omega2(x, gid), tau_over_sd(x, gid)
    perm = [(lambda L: (omega2(x, L), tau_over_sd(x, L)))(rng.permutation(gid))
            for _ in range(N_PERM)]
    w2_null = np.array([a for a, _ in perm])
    tau_null = np.array([b for _, b in perm])
    print(f'   {cv}')
    print(f'      omega^2  {w2:+.5f}   resolution floor {np.quantile(w2_null, 0.95):+.5f}')
    print(f'      tau/sd    {tau:.4f}   resolution floor  {np.quantile(tau_null, 0.95):.4f}'
          f'   (balance convention: 0.10)')
    rows.append(dict(check='A_omega2', covariate=cv, magnitude=w2,
                     resolution_floor=np.quantile(w2_null, 0.95), units='variance fraction'))
    rows.append(dict(check='B_tau_over_sd', covariate=cv, magnitude=tau,
                     resolution_floor=np.quantile(tau_null, 0.95), units='covariate SDs'))

# --- C: assignment against lane ------------------------------------------------------
print('\nC. association between perturbation identity and gem_group')
batch = obs['gem_group'].values.astype(str)
v = cramers_v_corrected(gid, batch)
v_null = np.array([cramers_v_corrected(rng.permutation(gid), batch) for _ in range(N_PERM)])
print(f"   corrected Cramer's V {v:.4f}   resolution floor {np.quantile(v_null, 0.95):.4f}"
      f"   ({len(np.unique(batch))} lanes)")
rows.append(dict(check='C_cramers_v_corrected', covariate='gem_group', magnitude=v,
                 resolution_floor=np.quantile(v_null, 0.95), units='0-1 association'))

# --- D: does the knockdown itself depend on the covariate? ---------------------------
# The uptake question, and the one that bears on instrument validity. Within each tercile
# of a covariate the perturbed-minus-control difference in the target gene is computed, so
# the UMI_size normalisation cancels between the two arms. Reported as a slope per
# covariate SD and as a percentage of the knockdown, both stable in n.
print('\nD. knockdown magnitude as a function of each covariate')
logX = np.log1p(np.asarray(data_X) / np.asarray(Offset))
for cv in CONT:
    x = obs[cv].values.astype(float)
    q = np.quantile(x, [1 / 3, 2 / 3])
    terc = np.digitize(x, q)
    xbar = [x[terc == t].mean() for t in range(3)]
    eff = {t: [] for t in range(3)}
    for j in range(len(gene_name_targeting)):
        pj = np.asarray(mask)[:, j]
        if pj.sum() < 30:
            continue
        for t in range(3):
            a, b = pj & (terc == t), ctrl & (terc == t)
            if a.sum() >= 10 and b.sum() >= 10:
                eff[t].append(logX[a, j].mean() - logX[b, j].mean())
    m = [np.mean(eff[t]) if eff[t] else np.nan for t in range(3)]
    overall = np.nanmean(m)
    slope = (m[2] - m[0]) / ((xbar[2] - xbar[0]) / x.std(ddof=1))   # per covariate SD
    rel = abs(m[2] - m[0]) / abs(overall) * 100 if overall else np.nan
    print(f'   {cv:<12} knockdown {overall:+.3f} (log) | low {m[0]:+.3f} mid {m[1]:+.3f} '
          f'high {m[2]:+.3f}')
    print(f'                slope {slope:+.4f} per covariate SD = {rel:.1f}% of the '
          f'knockdown across the outer terciles  ({len(eff[0])} genes)')
    rows.append(dict(check='D_knockdown_slope', covariate=cv, magnitude=slope,
                     resolution_floor=np.nan, units='log knockdown per covariate SD'))
    rows.append(dict(check='D_knockdown_spread_pct', covariate=cv, magnitude=rel,
                     resolution_floor=np.nan, units='% of knockdown'))

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f'\nwrote {OUT_CSV}')
print('Reading it: `magnitude` is the estimated size of the association in the stated '
      '`units`, with sampling noise removed where a correction exists. A magnitude below '
      '`resolution_floor` means the association is smaller than this design can '
      'distinguish from none -- report it as "below the resolution of the data", not as '
      'zero. No p-values are produced: at this number of cells they measure the sample '
      'size, not the association.')
