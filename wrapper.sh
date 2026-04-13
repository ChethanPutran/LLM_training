#!/bin/bash
source /mnt/data/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/miniconda3/envs/$USER
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=WARN
export DEEPSPEED_LOG_LEVEL=WARN   # Silences DeepSpeed info logs
export PYTHONWARNINGS="ignore"    # Optional: Silences Python-level warnings
mkdir -p /tmp/triton_$USER
export TRITON_CACHE_DIR=/tmp/triton_$USER

# Unique port per job, same across all tasks in the job
export MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 10000) ))
export MASTER_ADDR=$(scontrol show job $SLURM_JOB_ID | grep -o 'NodeList=[^ ]*' | cut -d= -f2 | head -1)

python -u "$@"