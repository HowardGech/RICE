#!/bin/bash
# Stage 2 - three fits per dataset (hard / soft / adaptive) on the GPU queue.
# Request GPU by memory, not whole cards; keep CPU small on dbeigpu.
# Arrays reject -K: submit detached, then poll `bjobs -A <jobid>`.
#   bsub < job_eff.sh
#BSUB -J eff[1-20]
#BSUB -o out/eff_%I.out
#BSUB -e out/eff_%I.err
#BSUB -q dbeigpu
#BSUB -gpu "num=1:gmem=16G"
#BSUB -n 2
#BSUB -R "span[hosts=1] rusage[mem=16000]"
#BSUB -W 24:00

cd "$CODE" || exit 1
module load cuda/12.8
PY="${PY:-python}"   # override with PY

hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY - <<'PRE' || exit 1
import sys, cupy
assert __import__("importlib").util.find_spec("cupy"), "CuPy not importable in " + sys.executable
assert cupy.cuda.is_available(), 'no CUDA device visible'
print('preflight ok', sys.executable, 'cupy', cupy.__version__,
      cupy.cuda.runtime.getDeviceProperties(0)['name'].decode())
PRE

model=NB
n=10000
p=100
s=400
graph_type=ER
# pen1: 1e-4 matches job_simu.py and the Methods lambda1.
# Use 1e-3 to match job_simu_comp_cf/cova/missing/highMOI/cpd instead.
pen1=1e-4
# xi_scope: what xi multiplies in the adaptive arm.
#   both   = parent sum AND control-function residual sum (strict Eq.3 generalization)
#   parent = parent sum only, CF term left unmasked (removes the upward bias in xi-hat)
xi_scope=both

for center in 0.0 0.2 0.4 0.6 0.8 1.0
do
  echo "=== fit center=${center} repl=${LSB_JOBINDEX} ==="
  $PY fit_efficacy.py ${model} ${n} ${p} ${s} ${graph_type} ${center} ${pen1} ${xi_scope} || exit 1
done
echo "=== eff element ${LSB_JOBINDEX} complete ==="
