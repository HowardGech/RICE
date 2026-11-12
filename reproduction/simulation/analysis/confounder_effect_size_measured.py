#!/usr/bin/env python
"""
Effect size of the unmeasured confounders, measured from a stored simulation file.

Nothing is re-simulated: job_gen_cova.py saves Z, Um, Ur, Vm and Vr alongside X, so the
confounder effect sizes are properties of the data on disk.

    python confounder_effect_size_measured.py n10k_p100_s400_ER_um10_nb_repl1.npz

WHAT "EFFECT SIZE" MEANS HERE. The generator is log-linear: for gene j the
negative-binomial mean is exp(eta_j) with

    eta_ij = sum_{k in Pa(j)} w_kj log(1 + x_ik)  +  z_i' u_j  +  v_j 1{i targets j}

so a change of delta on eta MULTIPLIES the expected count by exp(delta). An effect size
is therefore naturally quoted either in eta units (a standard deviation) or, more
interpretably, as the fold change it induces.

THREE NESTED DEFINITIONS, reported separately because they answer different questions.

  1. loading_abs_mean  -- the per-factor, per-gene coefficient u_kj ~ U(-1, 1).
     This is what the generator draws and what a Methods section usually quotes, but it
     is NOT the strength of the confounding: d factors accumulate on every gene, so the
     same loading range produces sqrt(5) times more confounding at d = 10 than at d = 2.
     Quoting only this leaves the reviewer's question unanswered.

  2. sd_latent_eta     -- the standard deviation, across cells, of the TOTAL latent
     contribution z_i' u_j to gene j. This is the per-gene effect size. Because
     z ~ N(0, I), it equals ||u_j||_2 in closed form, and the script checks the empirical
     value against that identity. fold_per_sd = exp(sd_latent_eta) converts it to the
     count scale: the factor by which expected expression changes between a cell at the
     mean of the latent contribution and one a standard deviation above it.

  3. var_share_latent  -- latent variance as a fraction of the variance of the linear
     predictor (latent + parental). This is the one that makes different d comparable,
     because the parental contribution grows with d as well: without it, "SD 1.79" cannot
     be judged large or small.

Two further quantities the reviewer's list asks for:
  frac_genes_affected  -- fraction of genes each factor loads on. Dense by construction
                         (Um is (d, p) with no zeros), so this is 1.0 and should be
                         stated rather than left implicit.
  sd_dispersion_eta    -- the same calculation on Ur, which drives the NB size parameter
                         r = exp(z' u_j^disp). The factors perturb overdispersion too,
                         not only the mean.

WHY A STANDARD DEVIATION AND NOT A MEAN ABSOLUTE VALUE.
z_i' u_j is exactly N(0, ||u_j||^2): z ~ N(0, I) and u_j is a fixed vector, so the
latent contribution is zero-mean Gaussian (measured skew 0.002, excess kurtosis 0.010).
For a zero-mean Gaussian the SD IS the complete description of scale, and the mean
absolute value carries no additional information -- it equals sqrt(2/pi) * sd = 0.798 sd
exactly (measured ratio 0.7977). Three reasons to report the SD:
  * it has the closed form ||u_j||_2, which is what makes the provenance self-check
    (closed_form_max_rel_err) possible;
  * variances add, so it composes with the parental term into var_share_latent and with
    the perturbation into confounding_to_perturbation. A mean absolute value does not
    decompose that way;
  * standardised effect sizes are SD-based by convention, so it is comparable with how
    effect sizes are reported elsewhere.
If a "typical displacement" reading is wanted instead, multiply by 0.798.

AVERAGING ORDER, which changes the headline number. fold_per_sd is exp(mean_j sd_j).
Averaging per gene first gives mean_j exp(sd_j), which is larger because exp is convex
(5.97 vs 6.17 on a d=10 file). A third statement, the AVERAGE multiplicative inflation
E[exp(z'u_j)] = exp(sd_j^2 / 2), gives 5.68. All three are correct about different
quantities; say which one is meant.

READING THE OUTPUT.
  loading_abs_mean        should equal coef/2 (0.5 at coef=1). A sanity check on the
                          file's provenance, NOT an effect size.
  sd_latent_eta           should equal covar*coef*sqrt(d/3) when the factors are standard
                          Gaussian with U(-coef, coef) loadings: 1.826 at d=10, coef=1,
                          covar=1. Agreement confirms the Methods text against the data.
  var_share_latent        a mean of PER-GENE ratios, so it does not equal the ratio of the
                          two mean SDs printed beside it, and root genes push it up.
                          Prefer var_share_latent_parented.
  closed_form_max_rel_err sampling noise alone gives a median of about 0.018 and a 95th
                          percentile of about 0.024 at n=10000, p=100. Values in that
                          range mean the check PASSED; a value several times larger would
                          indicate covar != 1 or a corrupted Z.

ROBUSTNESS NOTE. _simulate_single_equation rescales eta per gene (caps the max at 6.5,
then floors the mean at 0.5). Both are ADDITIVE shifts applied to the whole column, so
they change the intercept of eta but leave every variance, every SD and every variance
share above unchanged. The fold-change interpretation of the LATENT term is therefore
unaffected; only the absolute expression level is.
"""
import sys
import numpy as np
import pandas as pd


def confounder_effect_size_theory(npz_path, g=lambda x: np.log(1 + x)):
    f = np.load(npz_path, allow_pickle=True)
    X, W, Z, Um, Ur = f['X'], f['W'], f['Z'], f['Um'], f['Ur']
    Vm = f['Vm'] if 'Vm' in f.files else None
    p = W.shape[0]

    # job_gen_cova.py runs with intercept=True: Z's first column is ones and Um's first
    # row is theta_m. Strip them -- the intercept is not a confounder.
    has_intercept = Z.shape[1] == Um.shape[0] and Z.shape[1] > 0 and np.allclose(Z[:, 0], 1.0)
    Zl = Z[:, 1:] if has_intercept else Z
    Uml = Um[1:] if has_intercept else Um
    Url = Ur[1:] if has_intercept else Ur
    d = Zl.shape[1]

    lat = Zl @ Uml                                   # latent contribution to eta
    L = g(X)
    par = np.column_stack([
        L[:, np.flatnonzero(W[:, j])] @ W[np.flatnonzero(W[:, j]), j]
        if (W[:, j] != 0).any() else np.zeros(X.shape[0]) for j in range(p)])

    sd_lat = lat.std(axis=0)
    sd_par = par.std(axis=0)
    den = sd_lat ** 2 + sd_par ** 2

    per_gene = pd.DataFrame(dict(
        gene=np.arange(p),
        n_parents=(W != 0).sum(axis=0),
        loading_abs_mean=np.abs(Uml).mean(axis=0) if d else 0.0,
        sd_latent_eta=sd_lat,
        sd_latent_closed_form=np.linalg.norm(Uml, axis=0) if d else np.zeros(p),
        fold_per_sd=np.exp(sd_lat),
        sd_dispersion_eta=(Zl @ Url).std(axis=0) if d else np.zeros(p),
        sd_parent_eta=sd_par,
        var_share_latent=np.where(den > 0, sd_lat ** 2 / np.where(den > 0, den, 1), np.nan)))

    # model-free: dependence induced between genes that are NOT adjacent in the true DAG
    C = np.corrcoef(L, rowvar=False)
    off = ~((W != 0) | (W != 0).T) & ~np.eye(p, dtype=bool)

    summary = dict(
        d=d, n_cells=X.shape[0], n_genes=p, n_edges=int((W != 0).sum()),
        frac_genes_affected=float((np.abs(Uml) > 0).mean()) if d else 0.0,
        loading_abs_mean=float(np.abs(Uml).mean()) if d else 0.0,
        sd_latent_eta=float(sd_lat.mean()),
        fold_per_sd=float(np.exp(sd_lat.mean())),
        sd_dispersion_eta=float((Zl @ Url).std(axis=0).mean()) if d else 0.0,
        sd_parent_eta=float(sd_par.mean()),
        var_share_latent=float(np.nanmean(per_gene.var_share_latent)),
        # Genes with no parents have no parental term, so their share is 1 BY DEFINITION
        # and inflates the mean above. Report this one when comparing latent strength
        # against the regulatory signal; the two differ by several points.
        n_root_genes=int((per_gene.n_parents == 0).sum()),
        var_share_latent_parented=float(
            np.nanmean(per_gene.loc[per_gene.n_parents > 0, 'var_share_latent'])),
        median_abs_cor_nonadjacent=float(np.median(np.abs(C[off]))))

    # ALIGNMENT, the structural counterpart to the magnitude. corr(z'u_j, z'u_k) is
    # exactly cos(u_j, u_k): it is what turns the latent term from noise into bias, and
    # it is what an independent noise term of the SAME per-gene SD does not have.
    # It falls roughly as 1/sqrt(d) while the magnitude grows as sqrt(d), which is why
    # median_abs_cor_nonadjacent is not monotone in d.
    if d:
        Un = Uml / np.linalg.norm(Uml, axis=0)
        cos = Un.T @ Un
        summary['median_abs_cos_loading'] = float(np.median(np.abs(cos[~np.eye(p, dtype=bool)])))

    # Confounding relative to the DESIGNED SIGNAL. Both live on the same eta scale, so
    # their ratio is unit-free and needs no reference to the parental term: it says how
    # far the unobserved factors move a gene's log mean compared with how far the
    # perturbation moves it. This is the instrument-strength framing -- confounding
    # per unit of instrument -- and is the quantity that governs bias in an IV fit.
    if Vm is not None:
        pert = np.abs(np.asarray(Vm).ravel())
        per_gene['perturbation_eta'] = pert
        per_gene['confounding_to_perturbation'] = sd_lat / pert
        summary['perturbation_eta_mean'] = float(pert.mean())
        summary['confounding_to_perturbation'] = float((sd_lat / pert).mean())

    if d:
        # z ~ N(0, 1) => sd(z'u_j) = ||u_j||. Deviation flags a non-unit covar.
        rel = np.abs(sd_lat - np.linalg.norm(Uml, axis=0)) / np.linalg.norm(Uml, axis=0)
        summary['closed_form_max_rel_err'] = float(rel.max())
    return summary, per_gene


if __name__ == '__main__':
    for path in sys.argv[1:]:
        s, pg = confounder_effect_size_theory(path)
        print(f"\n== {path}")
        for k, v in s.items():
            print(f"   {k:<28} {v:.4f}" if isinstance(v, float) else f"   {k:<28} {v}")
        pg.to_csv(path.replace('.npz', '_confounder_effect_size.csv'), index=False)
