#!/bin/bash
# Stage 2 - fits. 4 configurations x 3 arms = 12 fits per element.
#   bsub < job_cfcompare.sh                          # poll: bjobs -A <jobid>
#BSUB -J cffit[1-20]
#BSUB -o out/cffit_%I.out
#BSUB -e out/cffit_%I.err
#BSUB -q dbeigpu
#BSUB -gpu "num=1:gmem=16G"
#BSUB -n 2
#BSUB -R "span[hosts=1] rusage[mem=16000]"
#BSUB -W 24:00

cd "$CODE" || exit 1
module load cuda/12.8
PY="${PY:-python}"   # override with PY

export RICE_BACKEND=cupy
export RICE_SCHEDULE=production

# -W IS A PLACEHOLDER: 12 fits per element and no GPU timing exists for this code.
# Submit as `#BSUB -J cffit[1]` first, read the `time` rows from the CSV, then widen to
# [1-20]. If one element will not fit the window, pass a single config per element
# instead of looping all four inside the driver.
$PY fit_cf_comparison.py NB 10000 100 400 ER 10 1e-4 || exit 1
echo "=== cffit element ${LSB_JOBINDEX} complete ==="
