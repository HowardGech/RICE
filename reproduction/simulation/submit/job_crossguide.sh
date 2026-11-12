#!/bin/bash
# Stage 2 - fits. Three control-function variants per dataset on the GPU queue.
#   bsub < job_crossguide.sh          # then poll: bjobs -A <jobid>
#BSUB -J xgfit[1-20]
#BSUB -o out/xgfit_%I.out
#BSUB -e out/xgfit_%I.err
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

model=NB; n=10000; p=100; s=400; graph_type=ER
leak_effect=3.0; d=10; itype=soft
# pen1: 1e-4 matches job_simu.py and the Methods lambda1. The comparison this arm is
# meant to sit beside (job_simu_comp_cf.py, the hidden-mediator result) uses 1e-3 --
# pick one and use it for both, or the curves are not comparable.
pen1=1e-4

# -W is a placeholder. Each element is 5 densities x 3 arms = 15 fits and no GPU timing
# exists for this code; run the array as [1] first and size it from the `time` rows.
# density * (p-1) leaking guides land on EACH gene, so at p=100 these are
# 0, 1, 2, 5 and 10 sources per gene. Plot against the `leak_to_signal` row
# in the CSV (median leak SD / parental-signal SD), not against density,
# which is not comparable across p.
for density in 0.0 0.01 0.02 0.05 0.10
do
  echo "=== fit density=${density} repl=${LSB_JOBINDEX} ==="
  $PY fit_crossguide.py ${model} ${n} ${p} ${s} ${graph_type} ${density} \
      ${leak_effect} ${d} ${itype} ${pen1} || exit 1
done
echo "=== xgfit element ${LSB_JOBINDEX} complete ==="
