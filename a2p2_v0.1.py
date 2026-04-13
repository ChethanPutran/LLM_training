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
import sys
import gc
from typing import Union
from transformers import GPT2LMHeadModel
import pynvml  # For hardware-level VRAM
from datasets import DatasetDict, load_from_disk, Dataset as HFDataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, DataCollator
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int)
parser.add_argument("--outputfile", type=str)
args = parser.parse_args()

STAGE = args.stage
OUTPUTFILE = args.outputfile

os.makedirs(f"/scratch/chethan1/SSDS/llm_training/outputs/{STAGE}", exist_ok=True)

BASE_DIR = os.path.abspath(f"/scratch/chethan1/SSDS/llm_training/outputs/stage_{STAGE}")
DATA_DIR = os.path.abspath(f"/scratch/chethan1/SSDS/llm_training/outputs/")
SCRATCH_DIR =  os.path.abspath("/scratch/chethan1/SSDS/llm_training/results")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)


out_train_dataset = os.path.join(DATA_DIR, "train_dataset")
final_train_dataset = os.path.join(out_train_dataset, "final")

# DeepSpeed temp directory for compiled extensions 
os.environ.setdefault("DEEPSPEED_USE_SOFT_ADAM", "1") 

# This is where DeepSpeed will store its compiled CUDA extensions.
os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_ext_{os.environ.get('USER', 'user')}")

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


# DeepSpeed config builder
def build_deepspeed_config(stage: int, micro_batch_size: int, gradient_accumulation: int) -> dict:
    """
    Build a DeepSpeed configuration dictionary based on the specified ZeRO stage and training parameters.
    """
    
    # Hyper parameters are from standard GPT-2 training configs
    ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps":gradient_accumulation,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas":   [0.9, 0.95],
            "eps":  1e-8,
            "weight_decay": 0.1,
        },
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr":   0,
            "warmup_max_lr":   5e-5,
            "warmup_num_steps": 100,
            "total_num_steps":  1000,
        },
    },
    "gradient_clipping":    1.0,
    "fp16":                 {"enabled": True, "loss_scale": 0},
    "steps_per_print":      1,

    # This will log the time taken for each step, including a breakdown of forward,
    #  backward, and step times.
    "wall_clock_breakdown": True, # enables detailed timing breakdown in DeepSpeed logs

    # This will profile the FLOPs at the specified step, giving us an estimate of 
    # the model's computational load and efficiency in TFLOPS.
    "flops_profiler": {
            "enabled": True,
            "output_file": f"{SCRATCH_DIR}/reflops_profile_{stage}.txt",
            # profiles FLOPs at step 10 for an estimate of model FLOPs.
            "profile_step": 10,
            "module_depth": 0,  # Set to 0 for overall model FLOPs, or higher for layer-wise breakdown
            "detailed": False, # Set to True for a more detailed breakdown of FLOPs by module
            "top_modules": 1,
        },
    # This will log the communication patterns, message sizes, and bandwidth during training, 
    # which is crucial for analyzing the communication overhead in ZeRO stages.
    "comms_logger": { 
        "enabled": True,
        "verbose": False,
        "prof_all": True
    },
    # Visual report (graphs instead of text tables)
    "monitor": {
    "enabled": True,
    "tag": "gpt2_zero_analysis",
    # This will create TensorBoard logs that we can use to visualize training metrics over time,
    #  including throughput and VRAM usage.
    "tensorboard": { 
        "enabled": True,
        "output_path": f"{SCRATCH_DIR}/tensorboard_logs",
        "job_name": f"gpt2_zero_stage_{stage}"
    },
    # This will create a CSV file with step-wise metrics that we can use to plot 
    # graphs of throughput and VRAM usage over time.
    "csv_monitor": { 
        "enabled": True,
        "output_file": f"{SCRATCH_DIR}/metrics_stage_{stage}.csv"
    }
    }
    ,
        "runtime": {
                "timers": {
                    "enabled": True
                }
            },
    # This will give us a detailed breakdown of time spent in different parts of the training loop, 
    # which is crucial for analyzing the communication overhead in ZeRO stages.
    "timers":{ 
        "enabled": True
        },
    }
   
    if stage > 0:
        ds_config["zero_optimization"] = {
            "stage":                 stage,
            "allgather_partitions":  stage >= 2,
            "allgather_bucket_size": 5e8 if stage >= 2 else 0,
            "overlap_comm":          stage >= 2,
            "reduce_scatter":        stage >= 2,
            "reduce_bucket_size":    5e8 if stage >= 2 else 0,
            "contiguous_gradients":  stage >= 2,
            "round_robin_gradients": stage == 3,
        }
    return ds_config



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
    print(f"\n{'='*80}")
    print(f"Training ZeRO Stage {STAGE}")
    print(f"{'='*80}")

    deepspeed.init_distributed()
 
    # Derive world size from the process group
    WORLD_SIZE = dist.get_world_size()
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    TOTAL_NUM_WEIGHT_UPDATE_STEPS = 100 
    GRADIENT_ACCUMULATION = 8
    TOTAL_STEPS = TOTAL_NUM_WEIGHT_UPDATE_STEPS * GRADIENT_ACCUMULATION
    MICRO_BATCH_SIZE = 4
    CHECKPOINT_FOR_STEP = 10
    WARMUP_STEPS = 0
    REPORT_SUMMARY_STEP = 10
    PROFILE_PER_STEP = 5
    PRINT_PROFILE = False
    TRAIN_BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION * WORLD_SIZE
 
    torch.cuda.set_device(LOCAL_RANK)
    print(f"World Size: {WORLD_SIZE}, Local Rank: {LOCAL_RANK}, Train Batch Size: {TRAIN_BATCH_SIZE} tokens/step")

    # Initialize NVML for precise NVIDIA stats
    pynvml.nvmlInit()

    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    print("Generating DeepSpeed configuration...")
    # DeepSpeed engine
    ds_config  = build_deepspeed_config(STAGE, MICRO_BATCH_SIZE, GRADIENT_ACCUMULATION)

    
    print("DEBUG:Loading dataset and creating DataLoader...")

    # Load your dataset
    raw_dataset: HFDataset = load_from_disk(final_train_dataset) # type: ignore

    # Convert to PyTorch tensors for DeepSpeed. 
    # This will allow us to use the dataset directly with the DeepSpeed engine, 
    # which expects PyTorch tensors for efficient data loading and processing.
    dataset: Dataset = raw_dataset.with_format("torch") # type: ignore

    # Define the Sampler
    # 'rank' and 'num_replicas' are automatically handled if you've initialized distributed
    sampler = DistributedSampler(
        dataset, 
        num_replicas=dist.get_world_size(), # Should be 4
        rank=dist.get_rank(),               # 0, 1, 2, or 3
        shuffle=True,                       # Recommended for pre-training
        seed=42
    )

    # Create the DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=ds_config['train_micro_batch_size_per_gpu'],
        sampler=sampler, 
        num_workers=4,
        pin_memory=True,
        shuffle=False, # shuffle must be False when using a sampler
    )


    # Model
    print("DEBUG:Initializing model...")

    model_config = AutoConfig.from_pretrained(GPT2_CONFIG_DIR)
    model = AutoModelForCausalLM.from_config(model_config)

    config_path = os.path.join(BASE_DIR, "deepspeed_config.json")

    print(f"DEBUG:Saving DeepSpeed config to {config_path}...")
    with open(config_path, "w") as f:
        json.dump(ds_config, f, indent=4)

    print("DEBUG:Initializing DeepSpeed engine...")

    # Initialize DeepSpeed engine with the model and config. 
    # This will handle model partitioning, optimizer setup, and mixed precision.
    deep_speed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    print("DEBUG:Initializing Profiler...")

    # Initialize the Profiler
    # This will track TFLOPS and Latency breakdown
    prof = FlopsProfiler(model)
    
    
    # Training loop 
    print("DEBUG:Starting training...")

    # Setup timing events for measuring step time (including communication in ZeRO stages)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # *************************************** THE LLM TRAINING LOOP ****************************************************
    # Load from checkpoint if available to resume training. 
    # This will ensure we can continue training from the last saved state in case of interruptions,
    # and also allows us to have a checkpointed model for each ZeRO stage for later evaluation.
    # By passing tag=None, DeepSpeed automatically reads the 'latest' file
    load_path, client_state = deep_speed_engine.load_checkpoint(
        checkpoint_dir, 
        tag=None 
    )

    if load_path is not None:
        # DeepSpeed successfully found and loaded a checkpoint
        # We retrieve the step count we saved manually in the 'client_state' dict
        current_step = client_state['step']
        print(f"DEBUG:RANK {dist.get_rank()}: Successfully resumed from {load_path} at step {current_step}")
    else:
        # No checkpoint found, starting from scratch
        current_step = 0
        print(f"DEBUG:RANK {dist.get_rank()}: No checkpoint found, starting from step 0")

    # Start total training timer
    total_start_time = time.perf_counter()

    # Total parameters (e.g., 51,210,000)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Assuming FP16 or BF16 (2 bytes per parameter)
    bytes_per_param = 2

    # Initialize data iterator once
    data_iter = iter(train_dataloader)

    total_tokens = 0  # To track total tokens processed for throughput calculation

    global_rank = dist.get_rank()

    print(f"DEBUG: GLOBAL RANK {global_rank} LOCAL RANK {LOCAL_RANK} - Starting training loop from step {current_step} to {TOTAL_STEPS} with ZeRO Stage {STAGE}")

    RANK_OUTPUTFILE = f"{OUTPUTFILE.replace('.txt', '')}_rank_{global_rank}.txt"

    for step in range(current_step, TOTAL_STEPS):
        # Re-set the seed for the sampler in distributed training to ensure each node gets different data per step
        # It ensures the data is reshuffled differently for each step
        if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(step)
        try:
            batch = next(data_iter)
        except StopIteration:
            # Sync ranks before resetting to prevent one rank from racing ahead
            dist.barrier(device_ids=[torch.cuda.current_device()]) 
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        # Warmup Phase: We can skip profiling and logging during the warmup 
        # phase to avoid noisy data.
        is_profiling_step = (step >= WARMUP_STEPS and (step % PROFILE_PER_STEP == 0))
        
        # Synchronization Barrier: Before the profiling step, we add a barrier
        # to ensure all nodes start the measured window at the same time. This
        # is crucial for accurate timing and fair comparison across ZeRO stages
        torch.distributed.barrier()

        # Start profiling at the designated step. This will capture the forward, 
        # backward, and step times, as well as compute the TFLOPS
        if is_profiling_step:
            prof.start_profile()

        # Record start event before the forward pass to capture total step time 
        # including communication in ZeRO stages.
        start_event.record(torch.cuda.current_stream())

        # Forward pass with DeepSpeed engine. 
        # DeepSpeed will handle the .to(device) and distributed data partitioning for us
        inputs = {k: v.to(deep_speed_engine.device, non_blocking=True) for k, v in batch.items()} # Move batch to the correct GPU

        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()
     
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        # In ZeRO-3, the forward pass will include hidden communication overhead as parameters
        # are fetched from other nodes.
        outputs = deep_speed_engine(**inputs)
        loss = outputs.loss

        # Backward pass with DeepSpeed engine.
        # DeepSpeed manages the ZeRO gradient sharding/reduction here
        deep_speed_engine.backward(loss)

        # Optimizer step with DeepSpeed engine. 
        # This will handle gradient clipping, weight updates, and LR scheduling
        deep_speed_engine.step()

        # Record end event after step to capture total time including communication in ZeRO stages.
        end_event.record(torch.cuda.current_stream())

        # Block CPU until GPU finishes work to ensure accurate timing
        torch.cuda.synchronize() 
                    
        # Stopping the profiler at the end of the step will give us the breakdown
        #  of forward/backward/step times, as well as TFLOPS.
        if is_profiling_step:
            prof.stop_profile()

            torch.cuda.synchronize()

            try:
                # Pull the key metrics from the profiler for reporting
                flops = prof.get_total_flops()
                macs = prof.get_total_macs()
                params = prof.get_total_params()

                if PRINT_PROFILE:
                    prof.print_model_profile(profile_step=step)

            except Exception as e:
                # print(f"DEBUG:RANK {dist.get_rank()}: Profiler not ready at step {step}")
                # print(f"DEBUG:Error retrieving profiler metrics: {e}")
                # Fallback if the first profiling step is still warm
                flops, macs, params = 0, 0, 0
                # print("DEBUG:Profiler attributes not yet ready.")
            # End the profiler to reset for the next measured step.
            # This is important because the profiler accumulates data, and we want to isolate each profiling step.
            prof.end_profile() 

            # Capture the true VRAM usage using NVML, which includes all CUDA context memory, not just PyTorch tensors.
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Get the allocated memory from PyTorch (this is what the DS logs will report)
            allocated_Gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_Gb = torch.cuda.memory_reserved() / (1024 ** 3)

            # Get the true VRAM usage from NVML (this is the "hardware reality" that includes
            # all memory used by the process, including fragmentation and non-PyTorch allocations)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            peak_vram_Gb = mem_info.used / (1024 ** 3)
            utilization_pct = util.gpu

                    
            # Total step time including communication overhead in ZeRO stages
            # This is the most critical metric for the report, as it captures the real-world
            #  impact of ZeRO's communication overhead on training speed
            step_time = start_event.elapsed_time(end_event)
            
            
            # Throughput calculation
            tokens_per_step = MICRO_BATCH_SIZE * WORLD_SIZE * block_size 
            total_tokens += tokens_per_step
            tokens_per_second = tokens_per_step / (step_time / 1000)

            

            if global_rank == 0:
                # Retrieve standard and DeepSpeed-specific metrics
                # --- 1. Data Prep & Calculations ---
                lr = deep_speed_engine.get_lr()[0]
                grad_norm = deep_speed_engine.get_global_grad_norm()

                # --- 1. Data Prep & Calculations ---
                lr = deep_speed_engine.get_lr()[0]
                grad_norm = deep_speed_engine.get_global_grad_norm()

                # Safeguard against None values for logging
                safe_lr = lr if lr is not None else 0.0
                safe_grad_norm = grad_norm if grad_norm is not None else 0.0
                
                # Raw Latencies (ms)
                fwd_t = deep_speed_engine.timers('forward').elapsed(reset=False) * 1000
                bwd_t = deep_speed_engine.timers('backward').elapsed(reset=False) * 1000
                ag_t  = deep_speed_engine.timers('allgather').elapsed(reset=False) * 1000
                rs_t  = deep_speed_engine.timers('reduce_scatter').elapsed(reset=False) * 1000
                
                compute_t = fwd_t + bwd_t
                comm_t    = ag_t + rs_t
                total_t   = compute_t + comm_t
                
                # --- 2. Efficiency Ratios ---
                # Comm Tax: % of step spent on networking
                comm_tax_pct = (comm_t / total_t * 100) if total_t > 0 else 0
                # Comp-to-Comm: Efficiency ratio (>1 is compute-bound, <1 is IO-bound)
                comp_comm_ratio = (compute_t / comm_t) if comm_t > 0 else 0
                # Memory Efficiency: % of reserved VRAM actually utilized
                mem_eff_pct = (allocated_Gb / peak_vram_Gb * 100) if peak_vram_Gb > 0 else 0
                # MFU: Model FLOPs Utilization (Ratio of actual vs theoretical peak)
                # Note: Replace 312 with your GPU's peak TFLOPS (e.g., A100=312, V100=125)
                gpu_peak_tflops = 312 
                mfu_pct = ((flops / 1e12) / gpu_peak_tflops * 100) if gpu_peak_tflops > 0 else 0

                # --- 3. Communication Volume ---
                multiplier = 3 if STAGE == 3 else 2
                step_vol_gb = (total_params * bytes_per_param * multiplier) / (1024**3)


                # --- THE COMPREHENSIVE SCRAPPABLE BLOCK ---
                print(f"METRIC_START STAGE: {STAGE} | STEP: {step} {'='*25}", flush=True)
                
                # Progress & Training Health
                print(f"METRIC_PROGRESS: loss={loss.item():.4f}, lr={safe_lr:.8f}, grad_norm={safe_grad_norm:.4f}", flush=True)
                
                # Speed & Hardware
                print(f"METRIC_SPEED: step_time_ms={step_time:.2f}, throughput_tps={tokens_per_second:.2f}, util_pct={utilization_pct}, flops_t={flops/1e12:.2f} TFLOPS, macs={macs/1e9:.2f} GMACs, params={params/1e6:.2f} M params, temp_c={temp_c}C ", flush=True)
                
                # Memory Metrics
                print(f"METRIC_VRAM: alloc_gb={allocated_Gb:.2f}, peak_gb={peak_vram_Gb:.2f}, reserved_gb={reserved_Gb:.2f}, mem_eff_pct={mem_eff_pct:.2f}", flush=True)
                
                # Latency Breakdown
                print(f"METRIC_LATENCY: compute_ms={compute_t:.2f}, fwd_ms={fwd_t:.2f}, bwd_ms={bwd_t:.2f}, comm_ms={comm_t:.2f}, ag_ms={ag_t:.2f}, rs_ms={rs_t:.2f}", flush=True)
                
                # Analysis Ratios (The "Insight" Data)
                print(f"METRIC_RATIOS: comm_tax_pct={comm_tax_pct:.2f}, compute_to_comm={comp_comm_ratio:.2f}, mfu_pct={mfu_pct:.2f}, vol_gb={step_vol_gb:.4f}", flush=True)
                
                print(f"METRIC_END {'='*10}\n",flush=True)


                print(f"Logging output to {RANK_OUTPUTFILE}")
                with open(RANK_OUTPUTFILE, "a") as f:
                    f.write(f"Rank {global_rank} completed step {step} on {deep_speed_engine.device}\n") 
                                  # --- THE COMPREHENSIVE SCRAPPABLE BLOCK ---
                    f.write(f"METRIC_START STAGE: {STAGE} | STEP: {step} {'='*25}",)
                    
                    # Progress & Training Health
                    f.write(f"METRIC_PROGRESS: loss={loss.item():.4f}, lr={safe_lr:.8f}, grad_norm={safe_grad_norm:.4f}")
                    
                    # Speed & Hardware
                    f.write(f"METRIC_SPEED: step_time_ms={step_time:.2f}, throughput_tps={tokens_per_second:.2f}, util_pct={utilization_pct}, flops_t={flops/1e12:.2f} TFLOPS, macs={macs/1e9:.2f} GMACs, params={params/1e6:.2f} M params, temp_c={temp_c}C ")
                    
                    # Memory Metrics
                    f.write(f"METRIC_VRAM: alloc_gb={allocated_Gb:.2f}, peak_gb={peak_vram_Gb:.2f}, reserved_gb={reserved_Gb:.2f}, mem_eff_pct={mem_eff_pct:.2f}")
                    
                    # Latency Breakdown
                    f.write(f"METRIC_LATENCY: compute_ms={compute_t:.2f}, fwd_ms={fwd_t:.2f}, bwd_ms={bwd_t:.2f}, comm_ms={comm_t:.2f}, ag_ms={ag_t:.2f}, rs_ms={rs_t:.2f}")
                    
                    # Analysis Ratios (The "Insight" Data)
                    f.write(f"METRIC_RATIOS: comm_tax_pct={comm_tax_pct:.2f}, compute_to_comm={comp_comm_ratio:.2f}, mfu_pct={mfu_pct:.2f}, vol_gb={step_vol_gb:.4f}")
                    
                    f.write(f"METRIC_END {'='*10}\n")


        # Periodic checkpointing. 
        # This saves the model state, optimizer state, and LR scheduler state in a format that can be resumed by DeepSpeed.
        if step % CHECKPOINT_FOR_STEP == 0:
            # We pass 'step' in a dict so we can retrieve it during load
            deep_speed_engine.save_checkpoint(
                checkpoint_dir, 
                tag=f"step_stage{STAGE}_step{step}_checkpoint",
                client_state={'step': step} 
            )

        # if step % 10 == 0:
        #     # Flush the DeepSpeed monitor to ensure all metrics are written to disk. 
        #     # This is important for real-time monitoring and for ensuring we have 
        #     # up-to-date logs for analysis.
        #     if hasattr(deep_speed_engine, 'monitor') and deep_speed_engine.monitor is not None:
        #         # Replace the crashing loop with this:
        #         # print(f"DEBUG:RANK {dist.get_rank()}: Flushing monitor at step {step}")
        #         deep_speed_engine.monitor.write_events()
        #     else:
        #         # print(f"DEBUG:RANK {dist.get_rank()}: Monitor not available to flush at step {step}")
        #         pass

        

        # Periodic summary logging. 
        # This will print the communication summary from DeepSpeed's comms logger, 
        # which includes message sizes and bandwidth.
        if dist.get_rank() == 0 and step % REPORT_SUMMARY_STEP == 0: 
            # DeepSpeed internal communication breakdown (All-Reduce vs All-Gather)
            deepspeed.comm.log_summary()
        
           
    # End total training timer
    total_training_time = time.perf_counter() - total_start_time

    if dist.get_rank() == 0:
        total_time = time.time() - total_start_time
        print(f"\n{'#'*20} FINAL STAGE {STAGE} REPORT {'#'*20}")
        print(f"Total Training Time: {total_training_time:.2f} seconds")
        print(f"Average Throughput: {total_tokens / total_time:.2f} tokens/sec")
        print(f"{'#'*50}")

    # Save model
    if STAGE == 3:
        # In ZeRO-3, the model parameters are fully partitioned across the nodes, 
        # so we need to gather them into a single state dict before saving in HuggingFace format.
        # This gathers the shards into a single FP32 state dict
        state_dict = deep_speed_engine.save_16bit_model(hf_output_dir)
        if dist.get_rank() == 0:
            model.save_pretrained(hf_output_dir, state_dict=state_dict)
    else:
        # In ZeRO-0, ZeRO-1, and ZeRO-2, the model parameters are either fully replicated or partially
        #  partitioned but still accessible on each node, so we can save directly using the DeepSpeed engine's module.
        if dist.get_rank() == 0:
            deep_speed_engine.module.save_pretrained(hf_output_dir)
            print(f"DEBUG:Model saved to {hf_output_dir}")

    # Cleanup before next stage
    # Barrier ensures all ranks finish before teardown.
    dist.barrier()
    del deep_speed_engine, model

    # Empty CUDA cache and synchronize to ensure all GPU memory is freed before the next stage starts. 
    # This is important to avoid out-of-memory errors in subsequent stages, especially when using ZeRO-3 which can have higher memory overhead.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"\nZeRO Stage {STAGE} completed. Results saved to {BASE_DIR}. Total Training Time: {total_training_time:.2f} seconds.")

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

def _concat_input_ids(test_dataset) -> torch.Tensor:
    """
    Concatenate the 'input_ids' from the test dataset into a single 1-D tensor.

    Args:
        test_dataset: A HuggingFace Dataset object that contains a column named 'input_ids', 
                      where each entry is a list or tensor of token IDs.
    """
    rows = test_dataset["input_ids"]
    parts = []
    for row in rows:
        if isinstance(row, torch.Tensor):
            parts.append(row.reshape(-1).long())
        else:
            parts.append(torch.tensor(row, dtype=torch.long))
    return torch.cat(parts)       
 

def _compute_perplexity(
    model: torch.nn.Module,
    encodings: torch.Tensor,
    stride: int = 512,
    max_length: int = 512,
) -> float:
    """
    Sliding-window perplexity on a pre-tokenised 1-D token tensor.

    Args:
        model: The language model to evaluate.
        encodings: A 1-D tensor of token IDs representing the test dataset.
        stride: The step size for the sliding window. Determines how much we move forward for each evaluation step.
        max_length: The maximum sequence length that the model can process. This determines the size of the sliding window.
    """
    sequence_length = encodings.size(0)
    device  = next(model.parameters()).device
    average_loss  = 0.0
    total_target_tokens = 0
 
    with torch.no_grad():
        for i in range(0, sequence_length, stride):
            # For each window, we determine the actual target tokens that will be scored, 
            # which is the last `trg_len` tokens of the window
            begin_location = max(i + stride - max_length, 0)
            end_location = min(sequence_length, i + stride)

            # We score only the last `current_length` tokens of the window, 
            # which is the overlap region that moves forward with each stride.
            current_length = end_location - i                     

            # Prepare input IDs for the current window
            # Extract the slice
            chunk = encodings[begin_location:end_location]

            # Ensure it's a tensor, then unsqueeze and move to device
            input_ids = torch.tensor(chunk).unsqueeze(0).to(device)

            # Label preparation: We want to compute the loss only on the last `trg_len` tokens of the window
            target_ids = input_ids.clone()

            # Mask out the context tokens (the first part of the window) by setting them to -100, 
            # which is the ignore index in PyTorch's CrossEntropyLoss.
            target_ids[:, :-current_length] = -100 
 
            outputs = model(input_ids, labels=target_ids)

            # We move the NLL sum to CPU immediately to avoid accumulating large tensors on GPU,
            # which can lead to OOM errors on large test sets. This way, we keep only the final
            # scalar NLL sums on CPU, which is memory efficient.
            cur_loss = outputs.loss.detach().cpu() * current_length

            average_loss += cur_loss.item() 

            # We accumulate the total number of target tokens across all windows to get the correct denominator for perplexity
            total_target_tokens += current_length

    average_loss = average_loss / total_target_tokens

    # Compute perplexity: exp(average_loss) gives us the perplexity, which is the exponentiation
    #  of the average negative log-likelihood per token.
    perplexity = np.exp(average_loss)

    return float(perplexity)
 
 
def _generate_sample(
    model: GPT2LMHeadModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
        ) 

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def step_4_evaluation(checkpoint_dir):
    """
    Evaluate perplexity and generate samples for multiple checkpoints.
    """
    print(f">>> Starting Step 4: Evaluation on {checkpoint_dir}...")
    
    ## start your edits here  =================

    #######################################
    if not dist.is_initialized():
        raise RuntimeError(
            "step_4_evaluation requires an initialised process group. "
            "Call deepspeed.init_distributed() before this function."
        )
 
    rank = dist.get_rank()
    world_size = dist.get_world_size()


    STRIDE = 512
    MAX_LENGTH = 512
    TEST_PROMPT = "Once upon a time"
    RESULTS_FILE = os.path.join(checkpoint_dir, "perplexity_results.json")
 
    # Discover checkpoints in the directory.
    all_checkpoints = sorted(
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d))
    )

    if not all_checkpoints:
        if rank == 0:
            print("No checkpoints found — nothing to evaluate.")
        return
 
    # Partition evenly across ranks (round-robin).
    my_checkpoints = [ all_checkpoints[i] for i in range(len(all_checkpoints)) if i % world_size == rank ]
 
    if rank == 0:
        print(f"Total checkpoints: {len(all_checkpoints)}")

    print(f"Rank {rank} evaluating: {my_checkpoints}")
 
    # Shared resources (loaded once per rank)
    MODEL_CONFIG = AutoConfig.from_pretrained(GPT2_CONFIG_DIR)
    tokenizer = AutoTokenizer.from_pretrained(GPT2_CONFIG_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    test_dataset = load_from_disk(final_test_dataset)

    # FIX 3: robust concatenation
    encodings = _concat_input_ids(test_dataset)
 
    device = torch.device(f"cuda:{torch.cuda.current_device()}"
                          if torch.cuda.is_available() else "cpu")
 
    # Evaluate each assigned checkpoint
    local_results: list[dict] = []
 
    for checkpoint_tag in my_checkpoints:
        print(f"[Rank {rank}] Evaluating checkpoint: {checkpoint_tag}")

        # Load the model state dict from the DeepSpeed checkpoint. 
        # We need to gather the full model state on this rank because DeepSpeed checkpoints 
        # are sharded across ranks, and for evaluation we need the complete model parameters 
        # to compute perplexity and generate samples.
        state_dict = get_fp32_state_dict_from_zero_checkpoint(
            checkpoint_dir, checkpoint_tag
        )

        # Initialize the model and load the state dict. 
        # We use the same config for all checkpoints since they are from the same training run.
        model = AutoModelForCausalLM.from_config(MODEL_CONFIG)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Use half-precision for evaluation to save 50% VRAM
        model.half()
 
        # Compute perplexity on the test dataset using the sliding window approach.
        perplexity = _compute_perplexity(model, encodings, STRIDE, MAX_LENGTH)
 
        # Generate a sample response from the model using a test prompt to qualitatively assess its generation quality.
        generated_response = _generate_sample(model, tokenizer, TEST_PROMPT)

        # Store the results in a structured format for later aggregation. 
        # Each entry includes the checkpoint tag, computed perplexity, and the generated sample for the test prompt.
        local_results.append({
            "checkpoint": checkpoint_tag,
            "perplexity": perplexity,
            "sample": {
                "prompt":   TEST_PROMPT,
                "response": generated_response,
            },
        })
 
        # Free GPU memory before loading the next checkpoint.
        del model, state_dict
        torch.cuda.empty_cache()

        # Force Python garbage collection for safety
        gc.collect()
 
    # Aggregate across all ranks
    all_gathered: list[list[dict] | None] = [None] * world_size

    # This will gather the local_results list from each rank into the all_gathered list on every rank.
    dist.all_gather_object(all_gathered, local_results)
    
    # Only rank 0 will process the gathered results and save to a JSON file. 
    # This avoids redundant file writing and ensures a single source of truth for the evaluation results.
    if rank == 0:
        flat_results: list[dict] = []
        for res_list in all_gathered:
            if res_list:
                flat_results.extend(res_list)
 
        # Sort for deterministic output regardless of rank assignment.
        flat_results.sort(key=lambda r: r["checkpoint"])

        # Save the aggregated results to a JSON file for later analysis and reporting.
        with open(RESULTS_FILE, "w") as f:
            json.dump(flat_results, f, indent=4)
 
        print(f"\nDEBUG: Evaluation complete. Results saved to {RESULTS_FILE}")

        for r in flat_results:
            print(f"  {r['checkpoint']}: PPL = {r['perplexity']:.2f}")

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