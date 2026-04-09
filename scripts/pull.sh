#!/bin/bash  

# 1. Load the variables from the .env file
if [ -f .env ]; then
    export $(echo $(cat .env | sed 's/#.*//g' | xargs) | envsubst)
else
    echo ".env file not found!"
    exit 1
fi

# Configuration
LOCAL_DIR="/home/chethan/Desktop/IISC/courses/sem2/SSML/asssignment/scalability_study/outputs/"
REMOTE_USER="chethan1"
REMOTE_HOST="10.24.1.10"
REMOTE_DIR="/scratch/chethan1/SSDS/scalability_study/outputs/"

echo "------------------------------------------------"
echo "Fetching only .log and .csv files from IISc Cluster..."
echo "------------------------------------------------"

# Rsync command to pull only .log and .csv files
sshpass -p "$REMOTE_PASS" rsync -avz --progress -e ssh \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_DIR"

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "Successfully synced .log and .csv files from cluster."
else
    echo "Error: Transfer failed. Check your VPN/SSH connection."
fi