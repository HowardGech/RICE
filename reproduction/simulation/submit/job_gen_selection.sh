#!/bin/bash
#BSUB -J selgen[1-20]
#BSUB -q cceb_normal
#BSUB -n 2
#BSUB -R "rusage[mem=16G]"
#BSUB -W 4:00
#BSUB -o log/selgen_%J_%I.out
#BSUB -e log/selgen_%J_%I.err
#
# Generation only -- no GPU, so cceb_normal.   bsub < job_gen_selection.sh
#
# One array element per replicate; each writes all five selection strengths for that
# replicate from ONE graph and ONE data-RNG start, so the settings are paired within the
# replicate and the only thing differing between them is the strength of the selection.
#
# Output lands in the project convention, simu_${INTERVENTION}/data/${MODEL}, ALONGSIDE the
# dense ..._cova6_repl{j}.npz arm. The sel### tag keeps the names distinct, but a glob on
# '*_cova6_*' now matches both -- narrow it by 'sel' when selecting this arm.
#
# Before widening to [1-20], run element 1 alone and read its [gen] lines:
#   corr(pert,u)   the severity axis. Expect about 0.01 / 0.19 / 0.34 / 0.45 / 0.54 against
#                  a ceiling of 0.585; it saturates by a = 2, which is why the sweep stops
#                  there. Report THIS, not a_select, which is not interpretable.
#   mean_u ctrl    the quantity actually distorted. The perturbed group is 90% of cells and
#                  barely moves; the control group is driven to about -1.6 SD. The violation
#                  is a control-group distortion, so say so rather than calling it imbalance.
#   controls       must stay > 0; the generator asserts it
#   cells/gene     must be identical across settings; the generator asserts min == max
#
# Expect EVERY method to degrade here: the estimand is no longer identified, so this arm
# measures how fast the bias grows rather than ranking methods.

set -euo pipefail

CODE="${SCGRAPH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo simulation/ dir; override with SCGRAPH_DIR
INTERVENTION=soft
MODEL=NB
mkdir -p "result/simu_${INTERVENTION}/data/${MODEL}" \
         log

# The bare interpreter on the CPU nodes resolves to a different stack than the project's;
# name it explicitly and fail loudly rather than generating with the wrong one.
PY="${PY:-python}"   # override with PY
[ -x "$PY" ] || { echo "interpreter not found: $PY" >&2; exit 1; }
"$PY" -c "import numpy, scipy, igraph" || { echo "missing dependencies in $PY" >&2; exit 1; }

cd "$CODE"
export SCGRAPH_DIR="$CODE"

"$PY" sim_dependent_assignment.py \
    --replicate "${LSB_JOBINDEX}" \
    --alphas 0,0.25,0.5,1.0,2.0 \
    --n 10000 --p 100 --s0 400 --graph ER --d 6 \
    --intervention "${INTERVENTION}" --model "${MODEL}" || exit 1

echo "=== selgen element ${LSB_JOBINDEX} complete ==="
