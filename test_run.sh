#!/bin/bash
#SBATCH --job-name=testing_file
#SBATCH --nodes=2            
#SBATCH --ntasks=2
#SBATCH --partition=debug
#SBATCH --time=00:05:00  
#SBATCH --output=/scratch/chethan1/SSDS/llm_training/logs/multi_stage_%j.log
#SBATCH --error=/scratch/chethan1/SSDS/llm_training/logs/multi_stage_%j.log


export OUTPUT_DIR="/scratch/chethan1/SSDS/llm_training/outputs"
STAGE=0
STAGE_DIR="$OUTPUT_DIR/stage_${STAGE}"
OUTPUTFILE="$STAGE_DIR/test_output.txt"

# Create directories before running srun
mkdir -p "$OUTPUT_DIR"
mkdir -p "$STAGE_DIR"
mkdir -p "/scratch/chethan1/SSDS/llm_training/logs"

# RUN COMMAND: Ensure NO spaces after the backslashes
srun -N 2 --ntasks=2 \
    --job-name=pp \
    --output="/scratch/chethan1/SSDS/llm_training/stage_${STAGE}_output.log" \
    --error="/scratch/chethan1/SSDS/llm_training/stage_${STAGE}_error.log" \
    /apps/run_wrapper.sh test.py --stage $STAGE --outputfile $OUTPUTFILE

srun -N 4 \
        --ntasks=4\
        --partition=ds256 \
        --qos=ds256_qos \
        --reservation=team08_20260414_0000 \
        -t 03:30:00 \
        /apps/run_wrapper.sh <your_script.py>
