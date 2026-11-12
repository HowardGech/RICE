#!/usr/bin/env python
"""
Latent-confounder effect sizes for the d = 2..10 sweep, for ALL THREE NB generators that
job_gen_cova.py can produce, because the working copy does not identify which one made
the published figure:

  tgt_log     simulate_NB_tgt, g = log(1+x)                  -- commented out in the driver
  tgt_cubic   simulate_NB_tgt, g = (log(1+x^3))^(1/3)        -- commented out; but
              job_simu_cova.py READS '..._um{um}_cubic_repl{job}.npz', so this tag is what
              the confounder-sweep FITS consume
  hidden_med  simulate_NB_hidden_mediator, g = log(1+x)      -- the ACTIVE line; writes
              '..._um{um}_hm_repl{job}.npz'

All share the same dense latent-factor structure (z ~ N(0, I_d) per cell, loadings
U(-1,1) on every factor-gene pair, separate matrices for mean and dispersion).
hidden_med adds a SECOND, sparse channel on top: for a fraction mediator_density of
(source, target) pairs an unobserved mediator h carries the perturbation D_source to the
target and to further leak genes, with couplings U(-3, 3) in and out.

Settings are those of job_gen_cova.py: n=10000, p=100, s0=400, ER, soft, shift=2,
coef=1, covar=1, intercept=True.
"""
import numpy as np, pandas as pd
from utils import simulate_dag, simulate_parameter, set_random_seed
from sim_base import simulate_NB_tgt, simulate_NB_hidden_mediator

N, P, S0, GRAPH = 10000, 100, 400, 'ER'
DS = [0, 2, 4, 6, 8, 10]
SEED = 0
G_LOG = lambda x: np.log(1 + x)
G_CUBIC = lambda x: (np.log(1 + x ** 3)) ** (1 / 3)

rows = []
for variant in ['tgt_log', 'tgt_cubic', 'hidden_med']:
    for d in DS:
        set_random_seed(SEED)
        B = simulate_dag(P, S0, GRAPH)
        W = simulate_parameter(B, w_ranges=((-1, -0.5), (0.5, 1.0)))
        kw = dict(W=W, theta_m=-np.ones(P), theta_r=np.ones(P), d=d, intercept=True,
                  n=N, shift=2, coef=1, covar=1, hard_intervention=False)
        if variant == 'hidden_med':
            out = simulate_NB_hidden_mediator(g=G_LOG, mediator_density=0.05,
                                              mediator_in_effect=3.0,
                                              mediator_out_effect=3.0, **kw)
            X, mask, Z, Um, Ur, Vm, Vr, leak = out
            n_src = len(leak)
            n_pairs = int(sum(len(v) for v in leak.values()))
        else:
            X, mask, Z, Um, Ur, Vm, Vr = simulate_NB_tgt(
                g=G_LOG if variant == 'tgt_log' else G_CUBIC, **kw)
            n_src, n_pairs = 0, 0

        Zl, Uml, Url = Z[:, 1:], Um[1:], Ur[1:]
        L = np.log1p(X)
        lat = Zl @ Uml
        par = np.column_stack([
            L[:, np.flatnonzero(W[:, j])] @ W[np.flatnonzero(W[:, j]), j]
            if (W[:, j] != 0).any() else np.zeros(N) for j in range(P)])
        sd_lat, sd_par = lat.std(axis=0), par.std(axis=0)
        den = sd_lat ** 2 + sd_par ** 2
        share = np.where(den > 0, sd_lat ** 2 / np.where(den > 0, den, 1), np.nan)
        C = np.corrcoef(L, rowvar=False)
        off = ~((W != 0) | (W != 0).T) & ~np.eye(P, dtype=bool)

        rows.append(dict(
            variant=variant, d=d,
            sd_latent_eta=float(sd_lat.mean()), fold_per_sd=float(np.exp(sd_lat.mean())),
            sd_parent_eta=float(sd_par.mean()), var_share_latent=float(np.nanmean(share)),
            sd_dispersion_eta=float((Zl @ Url).std(axis=0).mean()) if d else 0.0,
            median_abs_cor_null=float(np.median(np.abs(C[off]))),
            loading_mean_abs=float(np.abs(Uml).mean()) if d else 0.0,
            frac_genes_per_factor=float((np.abs(Uml) > 0).mean()) if d else 0.0,
            n_mediator_sources=n_src, n_mediator_pairs=n_pairs,
            frac_pairs_with_mediator=n_pairs / (P * (P - 1)),
            kd_percent_min=float(100 * (1 - np.exp(Vm).max())),
            kd_percent_max=float(100 * (1 - np.exp(Vm).min())),
            cells_per_perturbation=int(mask.sum(axis=0).min()),
            n_control_cells=int((mask.sum(axis=1) == 0).sum())))
        print(rows[-1]['variant'], rows[-1]['d'],
              round(rows[-1]['sd_latent_eta'], 3),
              round(rows[-1]['median_abs_cor_null'], 3),
              rows[-1]['n_mediator_pairs'], flush=True)

tab = pd.DataFrame(rows)
tab.to_csv('confounder_effect_size_theory.csv', index=False)
print('\n', tab[['variant', 'd', 'sd_latent_eta', 'fold_per_sd', 'var_share_latent',
                 'median_abs_cor_null', 'n_mediator_pairs']].round(3).to_string(index=False))
