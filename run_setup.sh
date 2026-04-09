# Set Up the Directory Structure
mkdir -p results/stage0 results/stage1 results/stage2 results/stage3


# 3. Step: Run the Experiments (The "SLURM" Way)
# You will submit 4 separate jobs (or run them sequentially).
# In each command, you must point the TensorBoard or CSV output to the corresponding folder.
# The trick: You don't need to change your Python code. You can override the output path using 
# the DeepSpeed configuration's job_name or by passing a variable to your script.

# In your SLURM script
deepspeed --num_nodes 4 --num_gpus 1 train.py \
    --deepspeed_config ds_stage2.json \
    --output_dir ./results/stage2/

# 4. Step: Automatically Naming Logs in Config
# To make plotting easier, use the job_name field in your ds_config.json for each run: