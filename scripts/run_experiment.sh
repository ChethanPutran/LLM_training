#!/bin/bash
#SBATCH --job-name=ds256_a2_p3
#SBATCH --nodes=4            
#SBATCH --ntasks=4
#SBATCH --partition=ds256
#SBATCH --time=04:00:00     
#SBATCH --output=multi_stage_%j.log

source ~/.bashrc

# Hard deadline for the slot
SLOT_END_TIME="04:00:00"
END_SECONDS=$(date -d "$(date +%Y-%m-%d) $SLOT_END_TIME" +%s)

RES_NAME=$(/apps/myslot.sh | grep "Reservation:" | awk '{print $2}')

for STAGE in 0 1 2 3
do
    # 1. Recalculate remaining time EVERY loop iteration
    CURRENT_SECONDS=$(date +%s)
    TOTAL_REMAINING=$((END_SECONDS - CURRENT_SECONDS))

    # 2. Safety Buffer: stop if we have less than 15 mins left
    if [ $TOTAL_REMAINING -lt 900 ]; then
        echo "Insufficient time remaining ($TOTAL_REMAINING sec) for Stage $STAGE. Exiting."
        break
    fi

    # 3. Time Allocation Strategy
    # We want to give this stage either 1 hour OR the remaining time divided by stages left.
    # Let's be simple: Give it a max of 1 hour (01:00:00) so we don't hog the slot if it hangs.
    STAGE_TIME_SEC=3600 
    if [ $STAGE_TIME_SEC -gt $TOTAL_REMAINING ]; then
        STAGE_TIME_SEC=$((TOTAL_REMAINING - 60)) # Use remaining minus 1 min buffer
    fi

    FMT_TIME=$(printf '%02d:%02d:%02d\n' $(($STAGE_TIME_SEC/3600)) $(($STAGE_TIME_SEC%3600/60)) $(($STAGE_TIME_SEC%60)))

    echo "----------------------------------------------------"
    echo "STARTING STAGE $STAGE at $(date +%H:%M:%S)"
    echo "Allocated Time for this stage: $FMT_TIME"
    echo "----------------------------------------------------"

    # 4. Execute srun
    srun -N 4 \
        --ntasks=4 \
        --partition=ds256 \
        --reservation=$RES_NAME \
        -t $FMT_TIME \
        /apps/run_wrapper.sh a2p2_v0.1.py $STAGE

done

echo "All scheduled stages completed at $(date)"