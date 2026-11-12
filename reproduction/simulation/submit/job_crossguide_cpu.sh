#!/bin/bash
# CPU fallback for when dbei-ai1 is unavail/closed_Adm. Identical maths under numpy.
# -n 8 rather than the -n 2 used for dbeigpu arrays: cceb_normal is not CPU-poor and
# this workload is BLAS-bound, so cores buy near-linear speedup.
#   bsub < job_crossguide_cpu.sh
#BSUB -J xgcpu[1]
#BSUB -o out/xgcpu_%I.out
#BSUB -e out/xgcpu_%I.err
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
# 10 cores with early stopping DISABLED -- so <= 2.1 h per fit on the reduced budget and
# <= 23.6 h on production, as UPPER BOUNDS. Two densities x 3 arms = 6 fits, i.e. <= 12.6 h
# worst case, which fits inside -W 24:00. Adding a third density (9 fits, <= 18.9 h) still
# fits; a fourth (12 fits, <= 25.2 h) does not. Read the `time` rows of the first element
# before widening, since the bound ignores early stopping and is loose.
model=NB; n=10000; p=100; s=400; graph_type=ER
leak_effect=3.0; d=10; itype=soft; pen1=1e-4

for density in 0.0 0.05
do
  echo "=== cpu fit density=${density} repl=${LSB_JOBINDEX} ==="
  $PY fit_crossguide.py ${model} ${n} ${p} ${s} ${graph_type} ${density} \
      ${leak_effect} ${d} ${itype} ${pen1} || exit 1
done
echo "=== xgcpu element ${LSB_JOBINDEX} complete ==="
