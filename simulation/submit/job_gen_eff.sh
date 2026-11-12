#!/bin/bash
# Stage 1 - data generation. Pure numpy, so CPU queue (never occupy a GPU slot).
# Arrays reject -K: submit detached, then poll `bjobs -A <jobid>`.
#   bsub < generate_efficacy.sh
#BSUB -J effgen[1-20]
#BSUB -o out/effgen_%I.out
#BSUB -e out/effgen_%I.err
#BSUB -q cceb_normal
#BSUB -n 2
#BSUB -R "span[hosts=1] rusage[mem=16000]"
#BSUB -W 4:00

cd "$CODE" || exit 1
PY="${PY:-python}"   # override with PY

# preflight: a bare `python` on cceb_normal resolves to /appl/python-3.12, a
# different numpy than prior results used. Fail loudly rather than silently
# generating data with a different stack.
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
half_width=0.1

for center in 0.0 0.2 0.4 0.6 0.8 1.0
do
  echo "=== gen center=${center} repl=${LSB_JOBINDEX} ==="
  $PY generate_efficacy.py ${model} ${n} ${p} ${s} ${graph_type} ${center} ${half_width} || exit 1
done
echo "=== effgen element ${LSB_JOBINDEX} complete ==="
