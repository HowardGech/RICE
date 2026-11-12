#!/bin/bash
#BSUB -J ivgen[1-20]
#BSUB -q cceb_normal
#BSUB -n 2
#BSUB -R "rusage[mem=16G]"
#BSUB -W 8:00
#BSUB -o log/ivgen_%J_%I.out
#BSUB -e log/ivgen_%J_%I.err
#
# Generation only -- no GPU, so cceb_normal rather than dbeigpu.  bsub < job_gen_invalid.sh
#
# One array element per replicate. Each element writes all six prevalences for that
# replicate from ONE graph and ONE data-RNG start, so the points are paired within the
# replicate, and the seed is the array index to match the fit driver.
#
# Run element 1 alone first and read its [gen] lines. The number to check is
# entries_affected: with n_recipients=10 and p=100 it is small even at frac 0.5, because a
# cell carries one guide. If the sweep shows no effect, raise --invalid-effect rather than
# the prevalence.

set -euo pipefail

CODE="${SCGRAPH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo simulation/ dir; override with SCGRAPH_DIR
DATA="${DATA:-$CODE/../result/simu_soft/data/NB}"   # override with DATA
mkdir -p "$DATA" log

# The bare interpreter on the CPU nodes resolves to a different stack than the project's;
# name it explicitly and fail loudly rather than generating with the wrong one.
PY="${PY:-python}"   # override with PY
[ -x "$PY" ] || { echo "interpreter not found: $PY" >&2; exit 1; }
"$PY" -c "import numpy, scipy, igraph" || { echo "missing dependencies in $PY" >&2; exit 1; }

cd "$CODE"
export SCGRAPH_DIR="$CODE"

"$PY" sim_invalid_guides.py NB 10000 100 400 ER soft 0 \
    --replicate "${LSB_JOBINDEX}" \
    --fracs 0,0.1,0.2,0.3,0.4,0.5 \
    --datadir "$DATA" || exit 1

echo "=== ivgen element ${LSB_JOBINDEX} complete ==="
