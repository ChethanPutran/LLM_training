#!/bin/bash  

# 1. Load the variables from the .env file
if [ -f .env ]; then
    export $(echo $(cat .env | sed 's/#.*//g' | xargs) | envsubst)
else
    echo ".env file not found!"
    exit 1
fi

REMOTE_USER="chethan1"
REMOTE_HOST="10.24.1.10"
REMOTE_DIR="/scratch/chethan1/SSDS/scalability_study/outputs/"

# Update these variables in your local script
REMOTE_FILE="/scratch/chethan1/SSDS/llm_training/stage_3_results.log"
LOCAL_DIR="/home/chethan/Desktop/IISC/courses/sem2/SSML/asssignment/llm_training"

# Revised rsync command for just ONE file
sshpass -p "$REMOTE_PASS" rsync -avz --progress -e ssh \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FILE}" "$LOCAL_DIR"