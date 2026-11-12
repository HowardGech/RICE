#!/bin/bash
# Sensitivity of RICE to invalid matched guides and nonlinear confounding.
# One array element per replicate; each element generates every configuration from its
# own graph and seed, then fits all three arms on each. Paired within replicate.
#
#   bsub < job_sensitivity.sh
#
# The array directive below requests ONE element on purpose. No GPU timing exists for this
# driver, so the -W value is a request rather than an estimate: run element 1, read the
# `time` rows it writes, set -W from them, then swap in the [1-20] line below.
#
# CPU fallback: submit to cceb_normal with RICE_BACKEND=numpy. Drop the -gpu line and the
# cuda module, raise -n, and use RICE_SCHEDULE=reduced -- the production optimizer budget
# costs hours per fit on CPU. Numbers from the two schedules are NOT interchangeable, so
# keep every arm of a comparison on one schedule.

#BSUB -J sens[1]
##BSUB -J sens[1-20]    <- uncomment this and comment the line above once -W is calibrated
#BSUB -q dbeigpu
#BSUB -gpu "num=1:gmem=16G"
#BSUB -n 2
#BSUB -R "rusage[mem=16G]"
#BSUB -W 24:00
#BSUB -o out/sens_%I.out
#BSUB -e out/sens_%I.err

set -euo pipefail

module load cuda/12.8

CODE="${SCGRAPH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo simulation/ dir; override with SCGRAPH_DIR
DATA="${DATA:-$CODE/../result/simu_soft/data/NB}"   # override with DATA
RES=result/simu_sens/scGRAPH/NB
PY="${PY:-python}"   # override with PY     # NOT a bare `python`: the compute
                                                  # nodes resolve that to a different
                                                  # interpreter with no cupy

mkdir -p "$DATA" "$RES" out
cd "$CODE"

# Preflight: fail here with a clear message rather than inside the fit loop.
"$PY" - <<'PREFLIGHT'
import importlib, sys
missing = [m for m in ("numpy", "pandas", "igraph", "scipy", "sklearn", "cupy")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"missing modules: {missing}")
import cupy as cp
cp.zeros(1).sum()                      # touches the device; fails now if none is visible
print(f"preflight ok: cupy {cp.__version__}, device "
      f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
PREFLIGHT

# The zero-severity reduction to the published generator, checked once per element. If
# this fails the sweep has no valid reference point and nothing below is worth running.
"$PY" run_sensitivity_arms.py --stage selftest

echo "=== element ${LSB_JOBINDEX} starting $(date) ==="

"$PY" run_sensitivity_arms.py \
    --mechanism both \
    --stage both \
    --replicate "${LSB_JOBINDEX}" \
    --n 10000 --p 100 --s0 400 --graph ER --d 10 \
    --itype soft \
    --n-recipients 10 \
    --invalid-effect 3.0 \
    --invalid-relevance 1.0 \
    --pen1 1e-4 \
    --w-threshold 0.3 \
    --backend cupy \
    --schedule production \
    --datadir "$DATA" \
    --resdir "$RES"

echo "=== element ${LSB_JOBINDEX} complete $(date) ==="
