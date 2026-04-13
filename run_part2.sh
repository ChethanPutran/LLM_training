#!/bin/bash

# Hard deadline for the slot
SLOT_END_TIME="03:50:00"
END_SECONDS=$(date -d "$(date +%Y-%m-%d) $SLOT_END_TIME" +%s)

# STAGES=(0 1 2 3)
STAGES=(0)
TOTAL_STAGES=${#STAGES[@]}

# Silence NVIDIA communication logs
export NCCL_DEBUG=WARN

# Silence DeepSpeed's internal monitoring noise
export DEEPSPEED_LOG_LEVEL=warning

# Silence Triton cache messages
export TRITON_INTERPRET=0


export OUTPUT_DIR="/scratch/chethan1/SSDS/llm_training/outputs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "/scratch/chethan1/SSDS/llm_training/logs"

for i in "${!STAGES[@]}"
do
    STAGE=${STAGES[$i]}
    STAGE_DIR="$OUTPUT_DIR/stage_${STAGE}"
    mkdir -p "$STAGE_DIR"

    OUTPUTFILE="$STAGE_DIR/test_output.txt"
    STAGES_LEFT=$((TOTAL_STAGES - i))

    # 1. Recalculate remaining time EVERY loop iteration
    CURRENT_SECONDS=$(date +%s)
    TOTAL_REMAINING=$((END_SECONDS - CURRENT_SECONDS))

    if [ $TOTAL_REMAINING -lt 120 ]; then
        echo "Insufficient time remaining ($TOTAL_REMAINING sec) for Stage $STAGE. Exiting."
        break
    fi

    TIME_SEC=$(($TOTAL_REMAINING - 5))

    # Format seconds to HH:MM:SS for SLURM
    FMT_TIME=$(printf '%02d:%02d:%02d\n' $(($TIME_SEC/3600)) $(($TIME_SEC%3600/60)) $(($TIME_SEC%60)))

    echo "----------------------------------------------------"
    echo "STARTING STAGE $STAGE at $(date +%H:%M:%S)"
    echo "Stages remaining: $STAGES_LEFT"
    echo "Allocated Time for this stage: $FMT_TIME"
    echo "----------------------------------------------------"

    export NCCL_DEBUG=INFO

    # 4. Execute srun
    srun -N 4 \
        --ntasks=4 \
        --partition=ds256 \
        --reservation=team08_20260414_0000\
        --output="/scratch/chethan1/SSDS/llm_training/stage_${STAGE}_output.log" \
        --error="/scratch/chethan1/SSDS/llm_training/stage_${STAGE}_error.log" \
        -t $FMT_TIME \
        /apps/run_wrapper.sh -u a2p2_v0.1.py --stage $STAGE --outputfile $OUTPUTFILE

    echo "--- Finished ZeRO Stage $STAGE ---"

done

echo "All scheduled stages completed at $(date)"