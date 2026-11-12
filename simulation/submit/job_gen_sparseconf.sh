#!/bin/bash
#BSUB -J confgen[1-20]
#BSUB -q cceb_normal
#BSUB -n 2
#BSUB -R "rusage[mem=16G]"
#BSUB -W 4:00
#BSUB -o log/confgen_%J_%I.out
#BSUB -e log/confgen_%J_%I.err
#
# Generation only -- no GPU, so cceb_normal.   bsub < job_gen_sparseconf.sh
#
# One array element per replicate; each writes all six fractions for that replicate from ONE
# graph and ONE data-RNG start, so the settings are paired within the replicate. The seed is
# the array index, matching the fit driver that reads these files.
#
# Output lands in the project convention, simu_${INTERVENTION}/data/${MODEL}, ALONGSIDE the
# pre-existing dense ..._cova6_repl{j}.npz arm. The conf### tag keeps the names distinct, but
# a glob on '*_cova6_*' now matches both -- narrow it by 'conf' when selecting this arm.
#
# Before widening to [1-20], run element 1 alone and read its [gen] lines:
#   latent_sd unconf   must be 0.000 at every fraction, or the zeroing is not working
#   gain               must sit inside [0.500, 1.500] at gamma = 0.5
#   conf/pert          about 1.6 at conf_scale 3 and d = 6; this is the magnitude claim
#   mean_count         rises with the fraction by construction. Higher depth helps recovery
#                      while confounding hurts, so a degradation across the sweep is
#                      conservative, but a null result would be uninterpretable.

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

"$PY" sim_sparse_confounding.py \
    --replicate "${LSB_JOBINDEX}" \
    --fracs 0,0.1,0.2,0.3,0.4,0.5 \
    --gamma 0.5 --alpha 0.5 --conf-scale 3.0 \
    --n 10000 --p 100 --s0 400 --graph ER --d 6 \
    --intervention "${INTERVENTION}" --model "${MODEL}" || exit 1

echo "=== confgen element ${LSB_JOBINDEX} complete ==="
