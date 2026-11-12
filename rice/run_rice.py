#!/usr/bin/env python
"""Run RICE on your own single-cell perturbation data.

    python run_rice.py --counts counts.csv --perturbations design.csv --out results/

Inputs are given on the command line; parameters live in config.json, so a run is
described by the pair (command line, config file) and nothing is hidden in the code.

INPUTS
  --counts          cells x genes matrix of raw UMI counts (.csv, .tsv, .npy, .npz, .h5ad)
  --perturbations   cells x genes 0/1 matrix: 1 where that gene was targeted in that cell.
                    Omit when --counts is .h5ad and --perturbation-column is given, in
                    which case the matrix is built from that column of adata.obs.
  --covariates      optional cells x k matrix of measured covariates (batch, cell cycle,
                    anything to adjust for). Used in the first stage and in the model.
  --gene-names      optional one name per column of --counts; taken from .h5ad var index
                    when available, otherwise g0, g1, ...

Run with --dry-run to load, check and report the inputs without fitting; this needs no GPU
and is the quickest way to find a shape or coding mistake before queueing a job.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VALID = {
    "preprocessing": {"transform", "size_factor", "standardize"},
    "control_function": {"mode", "include_covariates"},
    "model": {"intervention_type", "add_intercept", "regularizer", "binsum_k", "pen1", "pen2",
              "pen1_c", "pen2_c", "lr", "lr_coef_multiplier", "max_iter", "epochs", "checkpoint",
              "rho_init", "h_tol", "loss_tol", "dtype", "seed", "verbose"},
    "thresholding": {"method", "n_breakpoints", "w_threshold", "to_dag"},
    "output": {"save_weighted_matrix", "save_edge_list", "min_abs_weight_in_edge_list"},
}
CHOICES = {
    "preprocessing.transform": {"log1p", "log1p_offset", "none"},
    "preprocessing.size_factor": {"total_umi", "total_umi_per_gene", "none"},
    "control_function.mode": {"gene_matched", "full", "none"},
    "model.intervention_type": {"soft", "hard"},
    "model.regularizer": {"expm", "binsum"},
    "model.dtype": {"float64", "float32"},
    "thresholding.method": {"changepoint", "fixed", "none"},
}


def load_config(path, overrides):
    """Read config.json, apply --set overrides, and reject unknown or invalid entries.

    Unknown keys are an error rather than a warning: a typo in a parameter name would
    otherwise be silently ignored and the run would use the default, which is the kind of
    thing that is only noticed after the results look odd.
    """
    with open(path) as fh:
        cfg = json.load(fh)
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        parts = dotted.split(".")
        if len(parts) != 2 or parts[0] not in VALID:
            raise SystemExit(f"--set {dotted}: expected one of "
                             f"{', '.join(sorted(VALID))} followed by .key")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass                                   # leave bare strings as strings
        cfg.setdefault(parts[0], {})[parts[1]] = value
    for section, keys in cfg.items():
        if section not in VALID:
            raise SystemExit(f"config: unknown section {section!r}; "
                             f"expected {', '.join(sorted(VALID))}")
        unknown = set(keys) - VALID[section]
        if unknown:
            raise SystemExit(f"config [{section}]: unknown key(s) {', '.join(sorted(unknown))}; "
                             f"valid keys are {', '.join(sorted(VALID[section]))}")
    for dotted, allowed in CHOICES.items():
        section, key = dotted.split(".")
        v = cfg.get(section, {}).get(key)
        if v is not None and v not in allowed:
            raise SystemExit(f"config [{section}] {key}={v!r}: expected one of "
                             f"{', '.join(sorted(allowed))}")
    return cfg


def load_matrix(path, key=None, what="matrix"):
    """Load a 2-D numeric array from .csv/.tsv/.npy/.npz."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        import pandas as pd
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        non_numeric = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
        if non_numeric:                            # a leading cell-id column is common
            df = df.drop(columns=non_numeric)
        return df.to_numpy(dtype=float), list(df.columns)
    if ext == ".npy":
        return np.asarray(np.load(path), dtype=float), None
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        if key is None:
            names = [k for k in z.files if np.asarray(z[k]).ndim == 2]
            if len(names) != 1:
                raise SystemExit(f"{path}: holds {len(names)} two-dimensional arrays "
                                 f"({', '.join(z.files)}); name one with the matching --*-key")
            key = names[0]
        elif key not in z.files:
            raise SystemExit(f"{path}: no array {key!r}; it holds {', '.join(z.files)}")
        return np.asarray(z[key], dtype=float), None
    raise SystemExit(f"{what}: unsupported file type {ext!r}; "
                     "use .csv, .tsv, .npy, .npz, or .h5ad for --counts")


def load_h5ad(path, perturbation_column, control_label):
    """Load counts, gene names and (optionally) the design matrix from an AnnData file."""
    try:
        import anndata as ad
    except ImportError:
        raise SystemExit("reading .h5ad needs anndata: pip install anndata")
    adata = ad.read_h5ad(path)
    X = adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)
    genes = list(map(str, adata.var.index.values))
    mask = None
    if perturbation_column:
        if perturbation_column not in adata.obs.columns:
            raise SystemExit(f"{path}: no obs column {perturbation_column!r}; available: "
                             f"{', '.join(map(str, adata.obs.columns[:20]))}")
        labels = adata.obs[perturbation_column].astype(str).values
        index = {g: i for i, g in enumerate(genes)}
        mask = np.zeros_like(X)
        missing = set()
        for row, lab in enumerate(labels):
            if lab == control_label:
                continue
            for part in str(lab).split(","):        # a cell may carry several guides
                part = part.strip()
                if part in index:
                    mask[row, index[part]] = 1.0
                elif part:
                    missing.add(part)
        if missing:
            print(f"[warn] {len(missing)} perturbation label(s) are not gene columns and were "
                  f"ignored, e.g. {', '.join(sorted(missing)[:5])}")
    return X, mask, genes


def build_predictor(X, cfg, covariates):
    """Transform counts into the regulatory predictors the model conditions on."""
    pre = cfg["preprocessing"]
    if pre["transform"] == "none":
        P = X.astype(float).copy()
    elif pre["transform"] == "log1p":
        P = np.log1p(X)
    else:
        sf = pre["size_factor"]
        if sf == "none":
            P = np.log1p(X)
        else:
            total = X.sum(axis=1, keepdims=True)
            offset = total / X.shape[1] if sf == "total_umi_per_gene" else total
            offset = np.where(offset > 0, offset, 1.0)
            P = np.log1p(X / offset)
    if pre["standardize"]:
        sd = P.std(axis=0, keepdims=True)
        P = (P - P.mean(axis=0, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    return P


def build_control_function(predictor, mask, Z, cfg):
    """First stage: residual of each gene on the perturbation(s) and covariates.

    gene_matched regresses gene j on its OWN indicator only, which is the form the paper
    uses; full regresses on every indicator; none skips the adjustment entirely.
    """
    mode = cfg["control_function"]["mode"]
    if mode == "none":
        return None
    from sklearn.linear_model import LinearRegression
    use_Z = Z if (cfg["control_function"]["include_covariates"] and Z is not None) else None
    res = np.zeros_like(predictor)
    for j in range(predictor.shape[1]):
        design = mask if mode == "full" else mask[:, j].reshape(-1, 1)
        if use_Z is not None:
            design = np.hstack([use_Z, design])
        lr = LinearRegression().fit(design, predictor[:, j])
        res[:, j] = predictor[:, j] - lr.predict(design)
    return res


def threshold_weights(W_raw, cfg, to_dag_fn):
    """Sparsify the fitted weights, then orient to a DAG if asked."""
    th = cfg["thresholding"]
    W = W_raw.copy()
    if th["method"] == "changepoint":
        try:
            import ruptures as rpt
        except ImportError:
            raise SystemExit("thresholding.method='changepoint' needs ruptures "
                             "(pip install ruptures), or set it to 'fixed'")
        weights = np.sort(np.abs(W_raw).flatten())[::-1]
        cut = rpt.BottomUp(model="l1").fit(weights.reshape(-1, 1)).predict(
            n_bkps=int(th["n_breakpoints"]))
        cutoff = weights[cut[0] - 1]
        W[np.abs(W_raw) < cutoff] = 0
    elif th["method"] == "fixed":
        cutoff = float(th["w_threshold"])
        W[np.abs(W_raw) < cutoff] = 0
    else:
        cutoff = 0.0
    if th["to_dag"] and to_dag_fn is not None:
        W = to_dag_fn(W)
    return W, float(cutoff)


def check_inputs(X, mask, Z, genes):
    """Fail early, with the numbers, on the mistakes that are easy to make."""
    problems = []
    if X.ndim != 2:
        problems.append(f"counts must be 2-D, got shape {X.shape}")
    if mask.shape != X.shape:
        problems.append(f"perturbation matrix {mask.shape} does not match counts {X.shape}; "
                        "both must be cells x genes, in the same gene order")
    if not np.isfinite(X).all():
        problems.append(f"counts contain {int((~np.isfinite(X)).sum())} non-finite value(s)")
    if (X < 0).any():
        problems.append("counts contain negative values; RICE models raw UMI counts")
    if not np.isin(mask, (0, 1)).all():
        problems.append("the perturbation matrix must contain only 0 and 1")
    if Z is not None and Z.shape[0] != X.shape[0]:
        problems.append(f"covariates have {Z.shape[0]} rows, counts have {X.shape[0]}")
    if genes is not None and len(genes) != X.shape[1]:
        problems.append(f"{len(genes)} gene names for {X.shape[1]} columns")
    if problems:
        raise SystemExit("input check failed:\n  - " + "\n  - ".join(problems))
    n_pert = int((mask.sum(axis=1) > 0).sum())
    if n_pert == 0:
        raise SystemExit("no cell carries a perturbation; check the design matrix orientation")
    per_gene = mask.sum(axis=0)
    return {"cells": int(X.shape[0]), "genes": int(X.shape[1]),
            "perturbed_cells": n_pert, "control_cells": int(X.shape[0] - n_pert),
            "genes_perturbed": int((per_gene > 0).sum()),
            "median_cells_per_perturbation": float(np.median(per_gene[per_gene > 0]))
            if (per_gene > 0).any() else 0.0,
            "mean_count": float(X.mean()), "max_count": float(X.max())}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Estimate a causal gene network from single-cell perturbation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--counts", required=True, help="cells x genes raw UMI counts")
    ap.add_argument("--perturbations", help="cells x genes 0/1 design matrix")
    ap.add_argument("--covariates", help="optional cells x k covariate matrix")
    ap.add_argument("--gene-names", help="optional file with one gene name per line")
    ap.add_argument("--counts-key", help="array name, when --counts is a .npz")
    ap.add_argument("--perturbations-key", help="array name, when --perturbations is a .npz")
    ap.add_argument("--perturbation-column", help="obs column holding the targeted gene, "
                                                  "when --counts is a .h5ad")
    ap.add_argument("--control-label", default="non-targeting",
                    help="value of that column marking control cells")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--set", action="append", metavar="section.key=value",
                    help="override one config entry, repeatable")
    ap.add_argument("--out", default="rice_output", help="directory for the results")
    ap.add_argument("--prefix", default="rice", help="prefix for the output file names")
    ap.add_argument("--dry-run", action="store_true",
                    help="load and check the inputs, then stop without fitting (no GPU needed)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    genes = None
    if args.counts.lower().endswith(".h5ad"):
        X, mask, genes = load_h5ad(args.counts, args.perturbation_column, args.control_label)
    else:
        X, cols = load_matrix(args.counts, args.counts_key, "counts")
        mask, genes = None, cols
    if args.perturbations:
        mask, _ = load_matrix(args.perturbations, args.perturbations_key, "perturbations")
    if mask is None:
        raise SystemExit("no perturbation design: pass --perturbations, or --counts a .h5ad "
                         "together with --perturbation-column")
    Z = load_matrix(args.covariates, None, "covariates")[0] if args.covariates else None
    if args.gene_names:
        genes = [l.strip() for l in open(args.gene_names) if l.strip()]
    if genes is None or len(genes) != X.shape[1]:
        genes = [f"g{i}" for i in range(X.shape[1])]

    summary = check_inputs(X, mask, Z, genes)
    print("input summary: " + ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in summary.items()))
    predictor = build_predictor(X, cfg, Z)
    res = build_control_function(predictor, mask, Z, cfg)
    print(f"predictors: mean {predictor.mean():.3f}, sd {predictor.std():.3f} | "
          f"control function: {cfg['control_function']['mode']}")
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, args.prefix)
    record = {"inputs": {"counts": args.counts, "perturbations": args.perturbations,
                         "covariates": args.covariates},
              "summary": summary, "config": cfg}
    if args.dry_run:
        with open(stem + "_dryrun.json", "w") as fh:
            json.dump(record, fh, indent=2)
        print(f"dry run complete; inputs are usable. Wrote {stem}_dryrun.json")
        return 0

    sys.path.insert(0, HERE)
    try:
        import cupy as cp
        from scRICE_CF import scRICE_CF
    except ImportError as exc:
        raise SystemExit(f"fitting needs a GPU with CuPy installed ({exc}). "
                         "Use --dry-run to check the inputs without one.")
    from utils import to_dag
    mdl = cfg["model"]
    cp.random.seed(int(mdl["seed"]))
    np.random.seed(int(mdl["seed"]))
    model = scRICE_CF(dtype=cp.float64 if mdl["dtype"] == "float64" else cp.float32)
    model.prep(X=X, predictor=predictor, intervention_effect=mask,
               intervention_type=mdl["intervention_type"], add_intercept=mdl["add_intercept"],
               confounding_res=res, Zm_g=Z, Zr_g=Z)
    t0 = time.time()
    model.fit(regularizer=mdl["regularizer"],
              binsum_k=int(mdl["binsum_k"] or X.shape[1]),
              pen1=mdl["pen1"], pen2=mdl["pen2"], pen1_c=mdl["pen1_c"], pen2_c=mdl["pen2_c"],
              lr=mdl["lr"], lr_coef_multiplier=mdl["lr_coef_multiplier"],
              max_iter=int(mdl["max_iter"]), epochs=int(mdl["epochs"]),
              checkpoint=int(mdl["checkpoint"]), rho_init=mdl["rho_init"],
              h_tol=mdl["h_tol"], loss_tol=mdl["loss_tol"], verbose=mdl["verbose"])
    elapsed = time.time() - t0
    W_raw = model.result(w_threshold=0.0)["W"]
    W_est, cutoff = threshold_weights(W_raw, cfg, to_dag)
    if cfg["output"]["save_weighted_matrix"]:
        np.savez(stem + "_network.npz", W_raw=W_raw, W_est=W_est, genes=np.array(genes))
    if cfg["output"]["save_edge_list"]:
        import pandas as pd
        i, j = np.nonzero(W_est)
        keep = np.abs(W_est[i, j]) >= float(cfg["output"]["min_abs_weight_in_edge_list"])
        pd.DataFrame({"regulator": [genes[a] for a in i[keep]],
                      "target": [genes[b] for b in j[keep]],
                      "weight": W_est[i, j][keep]}).sort_values(
            "weight", key=np.abs, ascending=False).to_csv(stem + "_edges.csv", index=False)
    record.update({"threshold": cutoff, "edges": int((W_est != 0).sum()),
                   "runtime_seconds": round(elapsed, 1)})
    with open(stem + "_run.json", "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"{int((W_est != 0).sum())} edges at |w| >= {cutoff:.4g}; "
          f"fitted in {elapsed:.1f}s. Results written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
