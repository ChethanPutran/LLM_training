'''
## DS256 - Scalable Systems for Data Science | Jan 2026
# Assignment 2: LLM Distributed Training with DeepSpeed
# Part 2: Distributed Training & Evaluation

#### Posted on: 03/04/2026
#### Deadline: 17/04/2026

### Common Instructions
----
* You must ONLY edit regions that are clearly marked for modification.
* **DO NOT MODIFY other regions**. This can cause the evaluation script to break and you **WILL** be penalized or get **ZERO** points.
* You MUST NOT use the string **###!@** anywhere in your code or comments. We will be using this special substring for pattern matching during evaluations.
* You may declare **all** your valid imports and user-defined functions in the regions provided.

### Academic Integrity
----
The assignment must be completed by yourself and your teammate without any assistance from others or from online sources, ChatGPT, Copilot, etc.
If **any cases of plagiarism are detected, you may get a failing grade in the course and/or be reported to the Institute for further action**.

### Submission Guidelines (Part 2)
----
1. This script assumes the dataset was fully preprocessed by Part 1 and lives in your `BASE_DIR`. (See Part 1 script for more details) Launch configuration provided at the end of the script.

### Pipeline Steps (Part 2)
| Step | Name | Description |
|-------|------|-------------|
| 3 | Distributed Training | Train GPT-2 Mini from scratch using DeepSpeed and log benchmarks |
| 4 | Evaluation | Evaluate trained checkpoints on test dataset (Perplexity & Sample generation) |
'''

# ──────────────────────────────────────────────────────
# START: DO NOT MODIFY THIS SECTION (IMPORT STATEMENTS)
# ──────────────────────────────────────────────────────
import os
import subprocess
import time
import datetime
import random
import json
import numpy as np
import torch
import torch.distributed as dist
import deepspeed
import socket
import math
# ──────────────────────────────────────────────────────
# END: DO NOT MODIFY THIS SECTION (IMPORT STATEMENTS)
# ──────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────
# START: OTHER IMPORT STATEMENTS (ALLOWED TO MODIFY)
# ──────────────────────────────────────────────────────

# Add your import statements here

# ──────────────────────────────────────────────────────
# END: OTHER IMPORT STATEMENTS (ALLOWED TO MODIFY)
# ──────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────
# START: DO NOT MODIFY THIS SECTION (Shared dataset path)
# ──────────────────────────────────────────────────────

# Shared read-only directory containing the dataset and configs
shared_dir = "/mnt/data/ds256_2026/as2"

final_test_dataset = os.path.join(shared_dir, "test_dataset")

# Model path
GPT2_CONFIG_DIR = os.path.join(shared_dir, "model_config")

# ──────────────────────────────────────────────────────
# END: DO NOT MODIFY THIS SECTION (Shared dataset path)
# ──────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────
# START: MODIFY THE PATH (Local output directories, ALLOWED TO MODIFY)
# ──────────────────────────────────────────────────────

# Local output directories (relative to working directory)
BASE_DIR = os.path.abspath("/scratch/<your_username>/")

out_train_dataset = os.path.join(BASE_DIR, "train_dataset")
tokenized_train_dataset = os.path.join(out_train_dataset, "tokenized")
final_train_dataset = os.path.join(out_train_dataset, "final")

os.makedirs(out_train_dataset, exist_ok=True)
os.makedirs(tokenized_train_dataset, exist_ok=True)
os.makedirs(final_train_dataset, exist_ok=True)

DS_CONFIG_PATH = os.path.join(BASE_DIR, "ds_config.json")
# ──────────────────────────────────────────────────────
# END: MODIFY THE PATH (Local output directories, ALLOWED TO MODIFY)
# ──────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────
# START: HELPER FUNCTIONS AND CLASSES ( DO NOT MODIFY)
# ──────────────────────────────────────────────────────
def set_seed(seed):
    """Ensure deterministic initialization and sampling."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# ──────────────────────────────────────────────────────
# END: HELPER FUNCTIONS AND CLASSES ( DO NOT MODIFY)
# ──────────────────────────────────────────────────────

# Hyperparameters
block_size = 512


# ──────────────────────────────────────────────────────
# Step 3 - Distributed Training
# ──────────────────────────────────────────────────────
'''
## STEP 3: Distributed Training with DeepSpeed
----

### Objective
Train a GPT-2 mini model from scratch using DeepSpeed with a SLURM-based distributed initialization.

### Guidelines/Suggestions
1. Load the preprocessed training data (`final_train_dataset`) and initialize the model using `AutoModelForCausalLM.from_config`.
2. Initialize DeepSpeed at `DS_CONFIG_PATH`.
3. Write a training loop. Ensure you handle DeepSpeed's `backward()` and `step()`.
4. You may log metrics including VRAM utilization, GPU utilization, and Throughput (tokens/sec) periodically.
5. You may save DeepSpeed checkpoints periodically to ensure you have checkpointed models in case of terminated runs, and at the end of the epoch.
6. Export the final model in HuggingFace format using `save_pretrained`.

### DeepSpeed ZeRO Optimization Analysis (Mandatory)
----

#### Objective
Evaluate the impact of DeepSpeed ZeRO optimization stages on distributed training
performance, memory usage, and communication overhead.

#### Requirements
You MUST run training with the following ZeRO configurations:
- ZeRO Stage 0 (baseline, no optimization)
- ZeRO Stage 1
- ZeRO Stage 2
- ZeRO Stage 3
You MAY explore other Parallelism Techniques (e.g. Pipeline Parallelism, etc.), if you want to.

You MAY record the following metrics:
1. Total Training Time
2. Compute Time (forward + backward pass)
3. Communication Time (gradient synchronization / parameter sharding overhead)
4. Throughput (tokens processed per second)
5. GPU Memory Utilization (VRAM usage)

#### Instrumentation Suggestions
- Use DeepSpeed logs and profiling tools wherever possible.
- You MAY additionally use:
    * torch.cuda.Event for compute timing
    * DeepSpeed engine logs for communication timing
    * nvidia-smi / NVML for GPU memory utilization
- Clearly document how each metric is measured and any approximations used in your report.

#### Report Minimum Requirements
Your report MUST include:

1. Comparison of all ZeRO stages across all collected metrics.
2. Detailed analysis addressing the following:
   - How does communication overhead change across ZeRO stages?
   - What are the trade-offs between memory savings and runtime?
   - At which ZeRO stage do diminishing returns or performance degradation appear?
   - Which ZeRO stage performs best for your setup, and why?

#### Notes
- Superficial or qualitative observations without supporting data may be penalized.
'''

#######################################
###!@3 START ANSWER STEP 3

def step_3_training():
    """
    Train GPT-2-mini model from scratch using DeepSpeed with SLURM-based distributed initialization.
    """

    # ── Output directories ──
    now_dt = datetime.datetime.now()
    date_str = now_dt.strftime("%Y-%m-%d")
    time_str = now_dt.strftime("%H-%M-%S")
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints", date_str, time_str)
    hf_output_dir  = os.path.join(BASE_DIR, "gpt2_trained", date_str, time_str)

    ## start your edits here  =================


    ## end your edits here  =================

    return checkpoint_dir

###!@3 END ANSWER STEP 3
    

# ──────────────────────────────────────────────
# Step 4 - Evaluation
# ──────────────────────────────────────────────
'''
## STEP 4: Model and Checkpoint Evaluation
----

### Objective
Evaluate the trained model and multiple checkpoints to track progress and quality.

### Guidelines
1. Identify the saved checkpoints in `checkpoint_dir` and select a subset to evaluate.
2. For each selected checkpoint:
   - Calculate Perplexity on the test dataset using a sliding window approach with `stride=512`.
   - Generate a short text sample (e.g., 50 new tokens) to qualitatively assess the model using a dummy prompt.
3. Partition checkpoints across ranks to speed up evaluation.
4. Aggregate results from all ranks and save them to a JSON file (`perplexity_results.json`) in the `checkpoint_dir`.
    Format : 
    [{
        "checkpoint": "step_1000",
        "perplexity": 10.0,
        "sample":{
            "prompt": "Once upon a time",
            "response": "Hello, my name is ..."
        }
    },
    ...]
'''

#######################################
###!@4 START ANSWER STEP 4

def step_4_evaluation(checkpoint_dir):
    """
    Evaluate perplexity and generate samples for multiple checkpoints.
    """
    print(f">>> Starting Step 4: Evaluation on {checkpoint_dir}...")
    
    ## start your edits here  =================


    ## end your edits here  =================

    return 

###!@4 END ANSWER STEP 4
    

# ─────────────────────────────────────────────────────────────────
# START: Main pipeline (Distributed Training & Eval) (DO NOT MODIFY)
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ──────────────────────────────────────────────────────
    # START: SLURM ENVIRONMENT VARIABLES (DO NOT MODIFY)
    # ──────────────────────────────────────────────────────
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

    master = subprocess.getoutput(
        "scontrol show hostnames $SLURM_NODELIST | head -n 1"
    )
    print("MASTER_ADDR:", master)
    os.environ["MASTER_ADDR"] = master
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend="nccl")
    print(f"INIT DONE: {socket.gethostname()} rank {dist.get_rank()}")
    # ──────────────────────────────────────────────────────
    # END: SLURM ENVIRONMENT VARIABLES (DO NOT MODIFY)
    # ──────────────────────────────────────────────────────

    t_start = time.time()
    print(">>> Starting ML Assignment 2 Pipeline (Part 2)...")

    # Verify that dataset exists
    if not os.path.exists(final_train_dataset):
        raise FileNotFoundError(f"Missing fully preprocessed dataset at {final_train_dataset}. Did you run the Prep script first?")

    # Step 3: Train GPT-2-Small
    checkpoint_dir = step_3_training()

    # Step 4: Evaluate model and checkpoints
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        step_4_evaluation(checkpoint_dir)

    print(f"Part 2 Complete! Execution Time (this rank): {time.time() - t_start:.2f}s")


    dist.destroy_process_group()
    print(f"DESTROY DONE: {socket.gethostname()} rank {dist.get_rank()}")

    
# ─────────────────────────────────────────────────────────────────
# END: Main pipeline (Distributed Training & Eval) (DO NOT MODIFY)
# ─────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────
# GUIDELINES TO RUN THE CODE
# ──────────────────────────────────────────────────────
'''
SLURM & DeepSpeed Execution Details:
The cluster is configured to run distributed PyTorch jobs via SLURM. 

1. Run the `/apps/myslot.sh` script to find out when your next slot is. Run command: `/apps/myslot.sh`
Example Outputs:

✅ Your slot is ACTIVE now!
   Reservation: team12_20260403_0345

   Run your job with:
   srun -N <num_gpus> --ntasks=<num_gpus> --partition=ds256 --qos=ds256_qos --reservation=team12_20260403_0345 -t <HH:MM:SS> /apps/run_wrapper.sh <your_script.py>

2. In case your slot is not active, you will get the following output. Please wait for your slot to be active.
⏳ Your slot is NOT active yet.
   Next reservation: team03_20260404_1145
   Starts at:        2026-04-04T11:45:00

3. If your slot is active, note down the reservation name.
4. Run the following command to run your script. Replace <your_reservation_name> with the reservation name you noted down. 
   Replace <time> with the time you want to run your script for. Yoor job won't launch if the time exceeds your slot.
   The job will automatically terminate if the time exceeds your slot. All slots are 4 hours long.
   Do not change the partition and qos.
   You can run the job for a maximum of 4 hours.
   Replace <your_script.py> with the path to your script.
   This script should be run in a distributed mode since it performs distributed training and evaluation.
   Example Command:

    srun -N <num_gpus> \
        --ntasks=<num_gpus>\
        --partition=ds256 \
        --qos=ds256_qos \
        --reservation=<your_reservation_name> \
        -t <time> \
        /apps/run_wrapper.sh <your_script.py>
'''