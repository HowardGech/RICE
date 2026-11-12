#!/bin/bash
# Stage 1 - data generation. Pure numpy, so the CPU queue; never occupy a GPU slot.
#   bsub < generate_crossguide.sh      # then poll: bjobs -A <jobid>
#BSUB -J xggen[1-20]
#BSUB -o out/xggen_%I.out
#BSUB -e out/xggen_%I.err
#BSUB -q cceb_normal
#BSUB -n 2
#BSUB -R "span[hosts=1] rusage[mem=16000]"
#BSUB -W 4:00

cd "$CODE" || exit 1
PY="${PY:-python}"   # override with PY

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
leak_effect=3.0    # beta ~ U(-3, 3), matching the hidden-mediator arm
d=10               # d=0 makes the CF comparison vacuous; 10 is the published upper end
itype=soft         # soft | hard

# density * (p-1) leaking guides land on EACH gene, so at p=100 these are
# 0, 1, 2, 5 and 10 sources per gene. Plot against the `leak_to_signal` row
# in the CSV (median leak SD / parental-signal SD), not against density,
# which is not comparable across p.
for density in 0.0 0.01 0.02 0.05 0.10
do
  echo "=== gen density=${density} repl=${LSB_JOBINDEX} ==="
  $PY generate_crossguide.py ${model} ${n} ${p} ${s} ${graph_type} ${density} \
      ${leak_effect} ${d} ${itype} || exit 1
done
echo "=== xggen element ${LSB_JOBINDEX} complete ==="
