#!/bin/bash
# Control-function comparison: generation AND fitting in one array element, one .py file.
#
#   mkdir -p out
#   bsub < job_cfcompare_single.sh          # poll: bjobs -A <jobid>
#
# Each element is self-contained: it generates its own four datasets from its own DAG and
# then fits all three arms on them, so there is no cross-job dependency and no window in
# which a half-written .npz can be read by a fit job. Generation is CPU-only and costs
# under a minute at p=100, so the GPU sits idle for that long per element -- cheaper than
# managing a two-stage dependency.
#
#BSUB -J cfc[1-30]
#BSUB -o out/cfc_%I.out
#BSUB -e out/cfc_%I.err
#BSUB -q dbeigpu
#BSUB -gpu "num=1:gmem=16G"
#BSUB -n 2
#BSUB -R "span[hosts=1] rusage[mem=32000]"
#BSUB -W 24:00

cd "$CODE" || exit 1
module load cuda/12.8
PY="${PY:-python}"   # override with PY

# a bare `python` on these nodes resolves to a different stack; fail loudly if the
# interpreter or its GPU library is not the expected one
$PY - <<'PRE' || exit 1
import sys
assert __import__("importlib").util.find_spec("cupy"), "CuPy not importable in " + sys.executable
import numpy, cupy
print('preflight ok', sys.executable, 'numpy', numpy.__version__,
      'cupy', cupy.__version__)
PRE

DATADIR=result/simu_cfcompare/data/NB
RESDIR=result/simu_cfcompare/scGRAPH/NB

# --replicate defaults to $LSB_JOBINDEX, so the array index selects the DAG and the
# data-RNG start. pen1=1e-4 is the Methods value; job_simu_comp_cf.py used 1e-3 -- use
# ONE value across every arm and configuration or the curves are not comparable.
#
# -W IS A PLACEHOLDER: 12 fits per element (4 configurations x 3 arms) at the production
# optimizer budget, and no GPU timing exists for this code. Submit as `#BSUB -J cfc[1]`
# first, read the `time` rows from the CSV, then widen to [1-30]. If one element will not
# fit the window, split by configuration with --configs (keeping `none` in every split,
# since it is the reference for the difference-in-differences).
$PY compare_control_functions.py \
    --stage both \
    --n 10000 --p 100 --s0 400 --graph ER --d 10 \
    --dens 0.05 --pen1 1e-4 --w-threshold 0.3 \
    --backend cupy --schedule production \
    --datadir "${DATADIR}" --resdir "${RESDIR}" || exit 1

echo "=== cfc element ${LSB_JOBINDEX} complete ==="
