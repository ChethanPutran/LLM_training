
#!/bin/bash
for STAGE in 0 1 2 3
do
    echo "Starting Experiment for ZeRO Stage $STAGE..."
    srun -N 4 --ntasks=4 --partition=ds256 \
         --reservation=team08_20260408_0000 \
         --output="/scratch/chethan1/SSDS/llm_training/stage_${STAGE}_output.log" \
         --error="/scratch/chethan1/SSDS/llm_training/stage_${STAGE}_error.log" \
         -t 03:59:00 \
         /apps/run_wrapper.sh a2p2_v0.1.py $STAGE
done