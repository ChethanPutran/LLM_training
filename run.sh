# # Activate your Conda environment
# conda activate chethan1

# # Enable the newer compiler 
# scl enable gcc-toolset-13 bash

# # 3. Clear the bad DeepSpeed cache from previous failed attempts
# rm -rf ~/.cache/torch_extensions

#!/bin/bash
for STAGE in 0 1 2 3
do
    echo "Starting Experiment for ZeRO Stage $STAGE..."
    srun -N 4 --ntasks=4 --partition=ds256 \
         --reservation=team08_20260408_0000 \
         -t 01:00:00 \
         /apps/run_wrapper.sh a2p2_v0.1.py $STAGE
done