#!/bin/bash
# Stage 1 - data generation. Pure numpy, so the CPU queue; never occupy a GPU slot.
# Arrays reject -K: submit detached, then poll `bjobs -A <jobid>`.
#   bsub < generate_mismatch.sh
#BSUB -J mmgen[1-20]
#BSUB -o out/mmgen_%I.out
#BSUB -e out/mmgen_%I.err
#BSUB -q cceb_normal
#BSUB -n 2
#BSUB -R "span[hosts=1] rusage[mem=16000]"
#BSUB -W 4:00

cd "$CODE" || exit 1
PY="${PY:-python}"   # override with PY

# a bare `python` on cceb_normal resolves to /appl/python-3.12, a different numpy
# than earlier results used; fail loudly rather than generating with another stack
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
mode=swap          # swap | dropout | both
d=10               # unmeasured confounders; calibrated (d=3 near ceiling,
                   # d=20 inverts the arm ordering). d=0 makes the CF comparison vacuous.
itype=soft         # soft | hard -- misassignment bites far harder under hard,
                   # where the mask gates the parental sum; worth running both

# rho = 0 is the uncorrupted reference; the DAG, weights and counts are identical
# across rho for a given array index, so differences are attributable to corruption alone
for rho in 0.0 0.05 0.10 0.20 0.30
do
  echo "=== gen rho=${rho} mode=${mode} repl=${LSB_JOBINDEX} ==="
  $PY generate_mismatch.py ${model} ${n} ${p} ${s} ${graph_type} ${rho} ${mode} ${d} ${itype} || exit 1
done
echo "=== mmgen element ${LSB_JOBINDEX} complete ==="
