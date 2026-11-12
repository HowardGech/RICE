#!/bin/bash
# CPU fallback for when dbei-ai1 is unavail/closed_Adm. Identical maths under numpy.
# -n 8 rather than the -n 2 used for dbeigpu arrays: cceb_normal is not CPU-poor and this
# workload is BLAS-bound, so cores buy near-linear speedup.
#   bsub < job_cfcompare_cpu.sh
#BSUB -J cfcpu[1]
#BSUB -o out/cfcpu_%I.out
#BSUB -e out/cfcpu_%I.err
#BSUB -q cceb_normal
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=32000]"
#BSUB -W 24:00

cd "$CODE" || exit 1
PY="${PY:-python}"   # override with PY

export RICE_BACKEND=numpy
export RICE_SCHEDULE=reduced
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Cost basis: 85 ms per optimizer epoch measured under numpy at n=10000/p=100/s0=400 on
# 10 cores with early stopping DISABLED -- an UPPER BOUND of 2.1 h per fit on the reduced
# budget. This element runs 12 fits (4 configs x 3 arms), i.e. <= 25.2 h worst case, which
# does NOT fit -W 24:00. Either raise -W, or cut to two configurations per element
# (6 fits, <= 12.6 h). Read the `time` rows before widening -- the bound ignores early
# stopping and is loose.
$PY fit_cf_comparison.py NB 10000 100 400 ER 10 1e-4 || exit 1
echo "=== cfcpu element ${LSB_JOBINDEX} complete ==="
