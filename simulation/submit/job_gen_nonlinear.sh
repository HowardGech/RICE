#!/bin/bash
#BSUB -J nlgen[1-20]
#BSUB -q cceb_normal
#BSUB -n 2
#BSUB -R "rusage[mem=16G]"
#BSUB -W 8:00
#BSUB -o log/nlgen_%J_%I.out
#BSUB -e log/nlgen_%J_%I.err
#
# Generation only -- no GPU needed, so this belongs on cceb_normal rather than dbeigpu.
# One array element per replicate; each element writes every (form, alpha) for that
# replicate from ONE graph and ONE data-RNG start, so the severities are paired within a
# replicate. Submit with:   bsub < job_gen_nonlinear.sh
#
# Run element 1 alone first and read its [gen] lines: sd_before and sd_after must agree to
# four decimals at every alpha. They are equal by construction, so a disagreement means the
# mixing branch was edited, not that the setting is more severe.

set -euo pipefail

CODE="${SCGRAPH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo simulation/ dir; override with SCGRAPH_DIR
DATA="${DATA:-$CODE/../result/simu_soft/data/NB}"   # override with DATA
mkdir -p "$DATA" log

# The bare interpreter on the CPU nodes resolves to a different stack than the one the
# project uses; name it explicitly and fail loudly rather than generating with the wrong one.
PY="${PY:-python}"   # override with PY
[ -x "$PY" ] || { echo "interpreter not found: $PY" >&2; exit 1; }
"$PY" -c "import numpy, scipy, igraph" || { echo "missing dependencies in $PY" >&2; exit 1; }

cd "$CODE"
export SCGRAPH_DIR="$CODE"

for FORM in saturating interaction; do
  echo "=== form=${FORM} replicate=${LSB_JOBINDEX} ==="
  "$PY" sim_nonlinear_confounding.py \
      --form "${FORM}" \
      --replicate "${LSB_JOBINDEX}" \
      --alphas 0,0.25,0.5,0.75,1.0 \
      --n 10000 --p 100 --s0 400 --d 10 --graph ER \
      --intervention soft \
      --datadir "$DATA" || exit 1
done
echo "=== nlgen element ${LSB_JOBINDEX} complete ==="
