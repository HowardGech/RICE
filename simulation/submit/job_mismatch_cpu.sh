#!/bin/bash
# CPU fallback, for when dbei-ai1 is unavail/closed_Adm. Identical maths under numpy.
# -n 8 rather than the -n 2 used for dbeigpu arrays: that convention exists because
# dbeigpu is CPU-poor relative to its GPUs; cceb_normal is not, and this workload is
# BLAS-bound so cores buy near-linear speedup.
#   bsub < job_mismatch_cpu.sh
#BSUB -J mmcpu[1]
#BSUB -o out/mmcpu_%I.out
#BSUB -e out/mmcpu_%I.err
#BSUB -q cceb_normal
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=24000]"
#BSUB -W 24:00

cd "$CODE" || exit 1
PY="${PY:-python}"   # override with PY

export RICE_BACKEND=numpy
export RICE_SCHEDULE=reduced
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Cost basis: 85 ms per optimizer epoch measured under numpy at n=10000/p=100/s0=400 on
# 10 cores with early stopping DISABLED, so 2.1 h per fit is an UPPER BOUND for the
# reduced budget and 23.6 h for the production budget. 15 fits per element therefore has
# an upper bound far beyond -W 24:00 -- start at [1] with a single rho, read the `time`
# rows, and size the array from that.
model=NB; n=10000; p=100; s=400; graph_type=ER; mode=swap; d=10; itype=soft; pen1=1e-4

for rho in 0.0 0.20
do
  echo "=== cpu fit rho=${rho} repl=${LSB_JOBINDEX} ==="
  $PY fit_mismatch.py ${model} ${n} ${p} ${s} ${graph_type} ${rho} ${mode} ${d} ${itype} ${pen1} || exit 1
done
echo "=== mmcpu element ${LSB_JOBINDEX} complete ==="
