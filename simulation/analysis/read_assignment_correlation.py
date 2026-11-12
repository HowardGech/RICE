#!/usr/bin/env python
"""
Realised dependence per selection strength, for the brackets under the figure's x axis.

    python read_assignment_correlation.py result/simu_soft/data/NB

Reports, for each kappa in the sweep, the mean and standard deviation over replicates of the
realised correlation between perturbation status and the latent state. Two independent
routes, which is the point of the script:

  stored      corr_perturbed_u, written into each .npz's info dict by the generator and also
              collected in gen_selection_repl{j}.json
  recomputed  corr(any perturbation, standardised first latent factor), computed here from
              the stored mask and Z

They are computed from different things -- one is what the generator recorded at write time,
the other is what the saved arrays actually contain -- so a disagreement means the files do
not hold what their diagnostics claim, which is worth catching before the numbers go into a
caption. The script prints both and fails if they differ by more than a tolerance.

Writes selection_corr.csv (per file) and prints the bracket row ready to transcribe.
"""
import argparse, glob, json, os, re, sys
import numpy as np

# The sel### tag sits between the factor count and _repl, so the setting and the replicate
# must be matched SEPARATELY; a single pattern requiring _repl immediately after the digits
# would silently fail and pool every file under one setting.
SEL_RE = re.compile(r"_sel(\d+)")
REPL_RE = re.compile(r"_repl(\d+)")


def parse_tag(path):
    b = os.path.basename(path)
    s, r = SEL_RE.search(b), REPL_RE.search(b)
    return (int(s.group(1)) / 100 if s else None, int(r.group(1)) if r else None)


def recompute(path):
    """corr(perturbed at all, standardised first latent factor) from the saved arrays."""
    f = np.load(path, allow_pickle=True)
    mask, Z = f["mask"], f["Z"]
    # The generator prepends an intercept column of ones to Z, so the first latent factor is
    # column 1. Detect rather than assume, so a run generated with intercept=False still works.
    first = 1 if np.allclose(Z[:, 0], 1.0) else 0
    zz = Z[:, first].astype(float)
    zz = (zz - zz.mean()) / (zz.std() + 1e-12)
    pert = (mask.sum(axis=1) > 0).astype(float)
    if pert.std() == 0:
        return np.nan
    return float(np.corrcoef(pert, zz)[0, 1])


def stored(path):
    f = np.load(path, allow_pickle=True)
    if "info" not in f.files:
        return np.nan, None
    i = json.loads(str(f["info"]))
    return float(i.get("corr_perturbed_u", np.nan)), i.get("apply_clip")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir")
    ap.add_argument("--pattern", default="*_sel*_repl*.npz")
    ap.add_argument("--out-csv", default="selection_corr.csv")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="allowed |stored - recomputed| before the script fails")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(a.datadir, a.pattern)))
    if not paths:
        sys.exit(f"no files matching {a.pattern} under {a.datadir}")

    rows = []
    for pth in paths:
        kappa, repl = parse_tag(pth)
        if kappa is None or repl is None:
            print(f"[warn] could not parse the tag from {os.path.basename(pth)}; skipped")
            continue
        st, clip = stored(pth)
        rows.append(dict(kappa=kappa, replicate=repl, stored=st, recomputed=recompute(pth),
                         apply_clip=clip, file=os.path.basename(pth)))

    import csv
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    bad = [r for r in rows if np.isfinite(r["stored"])
           and abs(r["stored"] - r["recomputed"]) > a.tol]
    clips = {r["apply_clip"] for r in rows}
    if len(clips - {None}) > 1:
        print(f"[warn] files were generated with DIFFERENT clip settings {clips}; patched and "
              f"unpatched runs are not comparable and must not be pooled")

    ks = sorted({r["kappa"] for r in rows})
    print(f"\n{'kappa':>6} {'n':>3} {'mean corr':>10} {'sd':>7} {'min':>7} {'max':>7}")
    means = []
    for k in ks:
        v = np.array([r["recomputed"] for r in rows if r["kappa"] == k], float)
        means.append(v.mean())
        print(f"{k:>6.2f} {len(v):>3} {v.mean():>10.3f} {v.std(ddof=1) if len(v) > 1 else 0:>7.3f} "
              f"{v.min():>7.3f} {v.max():>7.3f}")
    # 0.585 is phi(z_q)/sqrt(q(1-q)) at q = 0.9: exact for this design, independent of n and p
    print(f"\nceiling at a 90% perturbation rate: 0.585 "
          f"(top setting reaches {100*max(means)/0.585:.0f}% of it)")
    print("bracket row:", "  ".join(f"{m:.2f}" for m in means))
    if bad:
        sys.exit(f"\nFAIL: stored and recomputed disagree for {len(bad)} file(s), e.g. "
                 f"{bad[0]['file']}: {bad[0]['stored']:.6f} vs {bad[0]['recomputed']:.6f}")
    print(f"\nstored and recomputed agree on all {len(rows)} files; wrote {a.out_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
