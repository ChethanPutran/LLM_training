#!/bin/bash

# 1. Load the variables from the .env file
if [ -f .env ]; then
    export $(echo $(cat .env | sed 's/#.*//g' | xargs) | envsubst)
else
    echo ".env file not found!"
    exit 1
fi

# Configuration
LOCAL_DIR="/home/chethan/Desktop/IISC/courses/sem2/SSML/asssignment/llm_training/"
REMOTE_USER="chethan1"
REMOTE_HOST="10.24.1.10"
REMOTE_DIR="/scratch/chethan1/SSDS/llm_training/"

echo "------------------------------------------------"
echo "Sending files to IISc Cluster (excluding outputs/, .env, and .git)..."
echo "------------------------------------------------"

# Rsync with exclusions
sshpass -p "$REMOTE_PASS" rsync -avz --progress -e ssh \
    --exclude='outputs/' \
    --exclude='.env' \
    --exclude='.git/' \
    --exclude='gitignore' \
    --exclude='tables/' \
    --exclude='plots/' \
    --exclude='data/' \
    "$LOCAL_DIR" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "Successfully synced to cluster."
else
    echo "Error: Transfer failed. Check your VPN/SSH connection."
fi
