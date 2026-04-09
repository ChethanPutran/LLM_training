#!/bin/bash
#SBATCH --job-name=ds256_a2_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ds256
#SBATCH --qos=ds256_qos
#SBATCH --time=01:00:00
#SBATCH --output=job_%j.log

# Load environment
source ~/.bashrc

echo "Starting job at $(date)"

# Get reservation dynamically
RESERVATION=$(/apps/myslot.sh | grep "Reservation:" | awk '{print $2}')

if [ -z "$RESERVATION" ]; then
    echo "No active reservation found. Exiting."
    /apps/myslot.sh
    exit 1
fi

echo "Using reservation: $RESERVATION"

# Run your script
srun --reservation=$RESERVATION /apps/run_wrapper.sh a2p1_v0.1.py

echo "Job finished at $(date)"