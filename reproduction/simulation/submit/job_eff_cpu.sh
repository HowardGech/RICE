#!/bin/bash
# CPU fallback for the efficacy fits, for when dbei-ai1 is closed_Adm and the
# GPU queue has no host to place a job on. scRICE_eff runs the identical maths
# under NumPy; only the array backend differs.
#
# Cores: -n 8 here, NOT the -n 2 used for dbeigpu arrays. That convention exists
# because dbeigpu is CPU-poor relative to its GPUs; cceb_normal is not, and this
# workload is BLAS-bound, so cores buy near-linear speedup.
#
# Cost basis (all measured under NumPy; NONE measured on GPU)
#
#   (a) Per-epoch rate at PRODUCTION shape (n=10000, p=100, s0=400), 10 cores,
#       max_iter=1 so every nominal epoch really executes: 85 ms/epoch.
#       => UPPER BOUNDS, assuming the full nominal epoch count is reached:
#            production budget (1e6 epochs)  <= 23.6 h per fit
#            reduced budget    (9e4 epochs)  <=  2.1 h per fit
#
#   (b) What the two early-stopping breaks actually save, A/B measured on
#       identical data at REDUCED shape (n=3000, p=20, s0=80, d=3):
#            loss_tol=1e-6 vs loss_tol=0   ->  0.46x (reduced), 0.53x (production)
#       Separately, the h_tol=1e-2 OUTER break exits well before max_iter: the
#       full production budget finished in 138 s at this shape, far short of its
#       1e6 nominal epochs.
#
#   The bounds in (a) are at production shape and are sound. The factors in (b)
#   are at p=20 and DO NOT transfer reliably to p=100 -- convergence depends on
#   problem shape. So size -W on (a), not on (a) scaled by (b), and replace both
#   with the `time` rows of the first element that completes at real scale.
#
#   bsub < job_eff_cpu.sh      # then poll: bjobs -A <jobid>
# Start with `#BSUB -J effcpu[1]` to calibrate, then widen to [1-20].
#BSUB -J effcpu[1-20]
#BSUB -o out/effcpu_%I.out
#BSUB -e out/effcpu_%I.err
#BSUB -q cceb_normal
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=24000]"
#BSUB -W 24:00

cd "$CODE" || exit 1
PY="${PY:-python}"   # override with PY

export RICE_BACKEND=numpy
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

hostname
nproc

$PY - <<'PRE' || exit 1
import sys, numpy
assert __import__("importlib").util.find_spec("cupy"), "CuPy not importable in " + sys.executable
print('preflight ok', sys.executable, 'numpy', numpy.__version__)
PRE

model=NB
n=10000
p=100
s=400
graph_type=ER
pen1=1e-4
xi_scope=both

# Reduced optimizer schedule. Upper bound 2.1 h per fit => <= 6.3 h per dataset
# (three arms) => <= 38 h for six centers, which EXCEEDS the -W 24:00 below.
# Early stopping will bring that down -- measured 0.46x at reduced shape, and the
# h_tol outer break cuts more -- but neither factor is established at p=100, so
# the bound is what -W must respect until one element has actually finished.
# Start with ONE element to measure, then either raise -W, cut the centers per
# element, or split the array over centers as well as replicates.
# The published budget (max_iter=100/epochs=1e4) has an upper bound of 23.6 h per
# fit and will not fit six centers in any reasonable window on CPU.
# NOTE the reduced schedule also changes lr (1e-3 vs 1e-4), so its results are
# not interchangeable with production-budget results.
export RICE_SCHEDULE=${RICE_SCHEDULE:-reduced}

for center in 0.0 0.2 0.4 0.6 0.8 1.0
do
  echo "=== cpu fit center=${center} repl=${LSB_JOBINDEX} ==="
  $PY fit_efficacy.py ${model} ${n} ${p} ${s} ${graph_type} ${center} ${pen1} ${xi_scope} || exit 1
done
echo "=== effcpu element ${LSB_JOBINDEX} complete ==="
