#!/bin/bash
#BSUB -J misgen[1-20]
#BSUB -q cceb_normal
#BSUB -n 2
#BSUB -R "rusage[mem=16G]"
#BSUB -W 4:00
#BSUB -o log/misgen_%J_%I.out
#BSUB -e log/misgen_%J_%I.err
#
# Generation only -- no GPU, so cceb_normal.    bsub < job_gen_mismatch_cells.sh
#
# One array element per replicate. The counts are generated ONCE from the true assignment
# and every rho reuses them, so the element is about as cheap as a single generation and
# the arms are paired by construction.
#
# Read the [gen] lines of element 1 before launching the fits. Two numbers matter:
#   records_incorrect  about 2 * rho, since each moved record is a false positive at the new
#                      gene and a false negative at the true one. Decide now which of the
#                      two the figure's x axis will be; they are not interchangeable.
#   min_per_gene       recorded perturbations for the thinnest gene. scRICE prep() needs at
#                      least one; the generator raises rather than writing a file that the
#                      fit would reject hours later.

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

"$PY" sim_guide_mismatch_cells.py NB 10000 100 400 ER soft 0 \
    --replicate "${LSB_JOBINDEX}" \
    --rhos 0,0.1,0.2,0.3,0.4,0.5 \
    --datadir "$DATA" || exit 1

echo "=== misgen element ${LSB_JOBINDEX} complete ==="
