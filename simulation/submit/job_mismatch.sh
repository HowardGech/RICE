#!/bin/bash
# Stage 2 - fits. Three control-function variants per dataset on the GPU queue.
#   bsub < job_mismatch.sh        # then poll: bjobs -A <jobid>
#BSUB -J mmfit[1-20]
#BSUB -o out/mmfit_%I.out
#BSUB -e out/mmfit_%I.err
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

model=NB
n=10000
p=100
s=400
graph_type=ER
mode=swap
d=10               # unmeasured confounders; calibrated (d=3 near ceiling,
                   # d=20 inverts the arm ordering). d=0 makes the CF comparison vacuous.
itype=soft         # soft | hard -- misassignment bites far harder under hard,
                   # where the mask gates the parental sum; worth running both
# pen1: 1e-4 matches job_simu.py and the Methods lambda1. Use 1e-3 to compare against
# job_simu_comp_cf.py / cova / missing / highMOI instead -- not both.
pen1=1e-4

# -W is a placeholder until one element reports. Each element is 5 rho x 3 arms = 15
# fits; check the `time` rows of the first CSV and resize before widening the array.
for rho in 0.0 0.05 0.10 0.20 0.30
do
  echo "=== fit rho=${rho} mode=${mode} repl=${LSB_JOBINDEX} ==="
  $PY fit_mismatch.py ${model} ${n} ${p} ${s} ${graph_type} ${rho} ${mode} ${d} ${itype} ${pen1} || exit 1
done
echo "=== mmfit element ${LSB_JOBINDEX} complete ==="
