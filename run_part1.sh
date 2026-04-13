
#!/bin/bash
echo "Starting Experiment..."

srun -N 1 \
        --ntasks=1\
        --partition=ds256 \
        --reservation=team08_20260414_0000 \
        -t 01:00:00 \
        /apps/run_wrapper.sh a2p1_v0.1.py 

echo "Experiment Completed."