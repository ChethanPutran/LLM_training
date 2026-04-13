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

def build_deepspeed_config(stage: int, micro_batch_size: int, gradient_accumulation: int, total_num_steps: int) -> dict:
    """
    Build a DeepSpeed configuration dictionary based on the specified ZeRO stage and training parameters.
    """
    
    # Hyper parameters are from standard GPT-2 training configs
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-5,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 1e-6,
                "warmup_max_lr": 5e-5,
                "warmup_num_steps": 10,  # 10% of your 100 update steps
                "total_num_steps": total_num_steps, 
            },
        },
        "gradient_clipping": 1.0,
        "fp16": {"enabled": True, "loss_scale": 0},
        "steps_per_print": 1,
        
        # This will log the time taken for each step, including a breakdown of forward,
        # backward, and step times.
        "wall_clock_breakdown": True,
        
        # This will profile the FLOPs at the specified step
        "flops_profiler": {
            "enabled": True,
            "output_file": f"{SCRATCH_DIR}/reflops_profile_{stage}.txt",
            "profile_step": 1,
            "module_depth": 0,
            "detailed": True,
            "top_modules": 1,
        },
        # This will log the communication patterns
        "comms_logger": { 
            "enabled": True,
            "verbose": False,
            "prof_all": True
        },
        
        # Visual report
        "monitor": {
            "enabled": True,
            "tag": "gpt2_zero_analysis",
            "tensorboard": { 
                "enabled": True,
                "output_path": f"{SCRATCH_DIR}/tensorboard_logs",
                "job_name": f"gpt2_zero_stage_{stage}"
            },
            "csv_monitor": { 
                "enabled": True,
                "output_file": f"{SCRATCH_DIR}/metrics_stage_{stage}.csv"
            }
        },
        
        # Timers for detailed breakdown
        "timers": { 
            "enabled": True
        },
    }
    
    # Add ZeRO optimization if stage > 0
    if stage > 0:
        ds_config["zero_optimization"] = {
            "stage": stage,
            "allgather_partitions": stage >= 2,
            "allgather_bucket_size": 5e8 if stage >= 2 else 0,
            "overlap_comm": stage >= 2,
            "reduce_scatter": stage >= 2,
            "reduce_bucket_size": 5e8 if stage >= 2 else 0,
            "contiguous_gradients": stage >= 2,
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
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints", date_str, time_str)

    # date_str = "2026-04-14"
    # time_str = "02-17-44"

    # return "/scratch/chethan1/SSDS/llm_training/outputs/stage_0/checkpoints/2026-04-14/02-17-44"
    # if STAGE == 3:
    #     return os.path.join(BASE_DIR, date_str, time_str)
    # return os.path.join(BASE_DIR, "gpt2_trained", date_str, time_str)
    print(f"\n{'='*80}")
    print(f"Training ZeRO Stage {STAGE}")
    print(f"{'='*80}")
    
    global_rank = dist.get_rank()
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(LOCAL_RANK)
    WORLD_SIZE = dist.get_world_size()

    # # Initialize distributed training with proper device mapping
    # LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    # WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    # global_rank = int(os.environ.get("RANK", 0))

    
    # Training parameters
    TOTAL_NUM_WEIGHT_UPDATE_STEPS = 100 
    GRADIENT_ACCUMULATION = 8
    MICRO_BATCH_SIZE = 4
    CHECKPOINT_FOR_STEP = 10
    WARMUP_STEPS = 0
    REPORT_SUMMARY_STEP = 10
    PROFILE_PER_STEP = 5
    PRINT_PROFILE = False
    
    
    # Initialize DeepSpeed distributed backend with device ID
    deepspeed.init_distributed(dist_backend="nccl")

    # Re-get ranks after initialization
    global_rank = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    
    print(f"World Size: {WORLD_SIZE}, Local Rank: {LOCAL_RANK}")

    # Initialize NVML
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    print("Generating DeepSpeed configuration...")
    ds_config = build_deepspeed_config(STAGE, MICRO_BATCH_SIZE, GRADIENT_ACCUMULATION, TOTAL_NUM_WEIGHT_UPDATE_STEPS)
    
    print("DEBUG: Loading dataset and creating DataLoader...")
    raw_dataset = load_from_disk(final_train_dataset)
    dataset = raw_dataset.with_format("torch")

    sampler = DistributedSampler(
        dataset, 
        num_replicas=WORLD_SIZE,
        rank=global_rank,
        shuffle=True,
        seed=42
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH_SIZE,
        sampler=sampler, 
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    print("DEBUG: Initializing model...")
    model_config = AutoConfig.from_pretrained(GPT2_CONFIG_DIR)
    model = AutoModelForCausalLM.from_config(model_config)

    print("DEBUG: Initializing DeepSpeed engine...")
    deep_speed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    print("DEBUG: Starting training...")
    
    # Setup timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Training state
    step = 0  # This is the number of weight updates completed
    micro_step = 0
    total_tokens = 0
    total_training_time = 0
    total_start_time = time.perf_counter()

    # Get total parameters for calculations
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param = 2  # FP16
    
    
    print(f"DEBUG: Starting training loop - Target weight updates: {TOTAL_NUM_WEIGHT_UPDATE_STEPS}")
    
    # Training loop
    data_iter = iter(train_dataloader)
    epoch = 0
    
    while step < TOTAL_NUM_WEIGHT_UPDATE_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        
        # Move batch to device
        inputs = {k: v.to(deep_speed_engine.device, non_blocking=True) for k, v in batch.items()}
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        # Forward pass
        outputs = deep_speed_engine(**inputs)
        loss = outputs.loss
        
        # Backward pass
        deep_speed_engine.backward(loss)
        
        micro_step += 1
        
        # Check if we should update weights
        if micro_step % GRADIENT_ACCUMULATION == 0:
            # Start timing for this weight update
            torch.cuda.synchronize(device=LOCAL_RANK)
            start_event.record(torch.cuda.current_stream())
            
            # Optimizer step
            deep_speed_engine.step()
            
            torch.cuda.synchronize(device=LOCAL_RANK)
            end_event.record(torch.cuda.current_stream())
            torch.cuda.synchronize(device=LOCAL_RANK)
            
            # Calculate step time
            step_time = start_event.elapsed_time(end_event)
            
            # Calculate throughput
            tokens_per_step = MICRO_BATCH_SIZE * WORLD_SIZE * GRADIENT_ACCUMULATION * block_size
            total_tokens += tokens_per_step
            tokens_per_second = tokens_per_step / (step_time / 1000)
            
            # Get metrics only on rank 0 and at profiling steps
            if global_rank == 0 and (step % PROFILE_PER_STEP == 0 or step == 0):

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                allocated_Gb = torch.cuda.memory_allocated(device=LOCAL_RANK) / (1024 ** 3)
                reserved_Gb = torch.cuda.memory_reserved(device=LOCAL_RANK) / (1024 ** 3)
                peak_vram_Gb = mem_info.used / (1024 ** 3)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Memory efficiency
                mem_eff_pct = (allocated_Gb / peak_vram_Gb * 100) if peak_vram_Gb > 0 else 0
                
                # Get learning rate and gradient norm
                lr = deep_speed_engine.get_lr()[0] if deep_speed_engine.get_lr() else 5e-5
                
                # Get gradient norm safely
                try:
                    print("DEBUG: Attempting to get global gradient norm...")
                    grad_norm = deep_speed_engine.get_global_grad_norm()
                    if grad_norm is None:
                        grad_norm = 0.0
                    print(f"DEBUG: Gradient norm obtained: {grad_norm}")
                except:
                    grad_norm = 0.0

                # Calculate approximate FLOPs (simple estimation)
                # For GPT-2: ~6 * params * tokens per step
                estimated_flops = 6 * total_params * tokens_per_step
                gpu_peak_tflops = 312  # A100 FP16 peak
                mfu_pct = ((estimated_flops / 1e12) / gpu_peak_tflops * 100) if gpu_peak_tflops > 0 else 0
                
                # Communication volume (theoretical)
                multiplier = 3 if STAGE == 3 else 2
                step_vol_gb = (total_params * bytes_per_param * multiplier) / (1024**3)

                # Try to get DeepSpeed timers (communication breakdown)
                try:
                    print("DEBUG: Attempting to get DeepSpeed timers for communication breakdown...")
                    if hasattr(deep_speed_engine, 'timers'):
                        timer_names = deep_speed_engine.timers.timer_names if hasattr(deep_speed_engine.timers, 'timer_names') else []
                        
                        fwd_t = deep_speed_engine.timers('forward').elapsed(reset=False) * 1000 if 'forward' in timer_names else 0
                        bwd_t = deep_speed_engine.timers('backward').elapsed(reset=False) * 1000 if 'backward' in timer_names else 0
                        ag_t = deep_speed_engine.timers('allgather').elapsed(reset=False) * 1000 if 'allgather' in timer_names else 0
                        rs_t = deep_speed_engine.timers('reduce_scatter').elapsed(reset=False) * 1000 if 'reduce_scatter' in timer_names else 0
                        step_t = deep_speed_engine.timers('step').elapsed(reset=False) * 1000
                        
                        # In Stage 0, DeepSpeed doesn't always split out 'comm' 
                        # separately from 'backward'. We can estimate it as:
                        compute_t = fwd_t + bwd_t
                        comm_t = ag_t + rs_t
                        comm_t = max(0, step_t - compute_t) 
                        total_t = compute_t + comm_t
                    
                        comm_tax_pct = (comm_t / step_t * 100) if step_t > 0 else 0
                        comp_comm_ratio = (compute_t / comm_t) if comm_t > 0 else 0
                        
                        # Reset timers periodically so the moving average stays fresh
                        if step % 20 == 0:
                            deep_speed_engine.timers.log(['forward', 'backward', 'step'])

                    else:
                        fwd_t = bwd_t = ag_t = rs_t = compute_t = comm_t = 0
                        comm_tax_pct = comp_comm_ratio = 0
                    print(f"DEBUG: Timers obtained - Forward: {fwd_t:.2f} ms, Backward: {bwd_t:.2f} ms, Allgather: {ag_t:.2f} ms, ReduceScatter: {rs_t:.2f} ms")
                except Exception as e:
                    fwd_t = bwd_t = ag_t = rs_t = compute_t = comm_t = 0
                    comm_tax_pct = comp_comm_ratio = 0

                   # Comprehensive output with all metrics
                output = (
                    f"METRIC_START STAGE: {STAGE} | STEP: {step} {'='*25}\n"
                    f"METRIC_PROGRESS: loss={loss.item():.6f}, lr={lr:.8f}, grad_norm={grad_norm:.6f}\n"
                    f"METRIC_SPEED: step_time_ms={step_time:.2f}, throughput_tps={tokens_per_second:.2f}, util_pct={util.gpu}, temp_c={temp_c}C\n"
                    f"METRIC_VRAM: alloc_gb={allocated_Gb:.2f}, peak_gb={peak_vram_Gb:.2f}, reserved_gb={reserved_Gb:.2f}, mem_eff_pct={mem_eff_pct:.2f}\n"
                    f"METRIC_LATENCY: compute_ms={compute_t:.2f}, fwd_ms={fwd_t:.2f}, bwd_ms={bwd_t:.2f}, comm_ms={comm_t:.2f}, ag_ms={ag_t:.2f}, rs_ms={rs_t:.2f}\n"
                    f"METRIC_RATIOS: comm_tax_pct={comm_tax_pct:.2f}, compute_to_comm={comp_comm_ratio:.2f}, mfu_pct={mfu_pct:.2f}, vol_gb={step_vol_gb:.4f}\n"
                    f"METRIC_END {'='*10}\n"
                )
                print(output, flush=True)
                
                # Save to file
                os.makedirs(os.path.dirname(OUTPUTFILE), exist_ok=True)
                with open(OUTPUTFILE, "a") as f:
                    f.write(output)
            
            # # Periodic communication summary (every 10 steps)
            # if global_rank == 0 and step > 0 and step % REPORT_SUMMARY_STEP == 0:
            #     try:
            #         print(f"\n--- Communication Summary at Step {step} ---")
            #         deepspeed.comm.log_summary()
            #         print("--- End Communication Summary ---\n")
            #     except:
            #         pass

            # Checkpointing
            if step > 0 and step % CHECKPOINT_FOR_STEP == 0:
                deep_speed_engine.save_checkpoint(
                    checkpoint_dir, 
                    tag=f"stage{STAGE}_step{step}", 
                    client_state={'step': step}
                )
            
            step += 1
            micro_step = 0
            
            # Progress indicator
            if global_rank == 0 and step % 10 == 0:
                print(f"Progress: Weight update {step}/{TOTAL_NUM_WEIGHT_UPDATE_STEPS} completed")
    
    # Synchronize all ranks
    dist.barrier(device_ids=[LOCAL_RANK])
    
    # Calculate total training time
    total_training_time = time.perf_counter() - total_start_time if 'total_start_time' in dir() else 0
    
    if global_rank == 0:
        print(f"\n{'#'*20} FINAL STAGE {STAGE} REPORT {'#'*20}")
        print(f"Total Training Time: {total_training_time:.2f} seconds")
        print(f"Total Tokens Processed: {total_tokens}")
        print(f"Total Weight Updates: {step}")
        print(f"{'#'*50}")

    # Save model
    if global_rank == 0:
        os.makedirs(hf_output_dir, exist_ok=True)
        deep_speed_engine.module.save_pretrained(hf_output_dir)
        print(f"DEBUG: Model saved to {hf_output_dir}")

    # Cleanup
    del deep_speed_engine, model
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device=LOCAL_RANK)

    print(f"\nZeRO Stage {STAGE} completed.")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
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
    print(f"DEBUG: Concatenating test dataset input_ids...", flush=True)
    rows = test_dataset["input_ids"]
    parts = []
    for row in rows:
        if isinstance(row, torch.Tensor):
            parts.append(row.reshape(-1).long())
        else:
            parts.append(torch.tensor(row, dtype=torch.long))
    res = torch.cat(parts)
    print(f"DEBUG: Concatenation complete. Total tokens: {res.size(0)}", flush=True)
    return res

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
            if i % 5000 == 0: # Print every 5000 tokens to keep logs clean but informative
                print(f"  > PPL Progress: {i}/{sequence_length} tokens evaluated...", flush=True)
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
            input_ids = chunk.clone().detach().unsqueeze(0).to(device)

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
 

def _generate_sample(model, tokenizer, prompt, max_new_tokens=50):
    # 1. Encode with return_tensors='pt' and ensure it's on the right device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 2. Check if encoding actually produced tokens
    if inputs["input_ids"].shape[1] == 0:
        return "Error: Empty input_ids produced during tokenization."

    # 3. Generate with explicit attention_mask
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"), # Standard practice
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 4. Decode the result
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

    print(f"DEBUG: Tokenizer loaded. Special tokens - pad: {tokenizer}")

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
        # Add a stricter check inside your loop
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_tag)

        # Ensure it's a real DS checkpoint by looking for a 'latest' file or a .pt file
        if not any(fname.endswith('.pt') or fname == 'latest' for fname in os.listdir(ckpt_path)):
            print(f"Skipping {checkpoint_tag}, not a valid DeepSpeed checkpoint.")
            continue

        print(f"[Rank {rank}] Evaluating checkpoint: {checkpoint_tag}")

        if STAGE == 0:
            print(f"DEBUG: Loading checkpoint from {ckpt_path} for evaluation...")

            # --- REPLACE THE DEEPSPEED UTILITY WITH THIS ---
            model_ckpt_path = os.path.join(ckpt_path, "mp_rank_00_model_states.pt")
            
            print(f"DEBUG: Looking for model checkpoint at {model_ckpt_path}...")
            # Load the checkpoint file
            # 'map_location' ensures it loads to the current GPU correctly
            full_ckpt = torch.load(model_ckpt_path, map_location=device)
            print(f"DEBUG: Checkpoint loaded. Keys in checkpoint: {list(full_ckpt.keys())}")
            # DeepSpeed saves a dict; the actual weights are under the 'module' key
            if 'module' in full_ckpt:
                state_dict = full_ckpt['module']
            else:
                state_dict = full_ckpt
            # -----------------------------------------------

        else:

            # Load the model state dict from the DeepSpeed checkpoint. 
            # We need to gather the full model state on this rank because DeepSpeed checkpoints 
            # are sharded across ranks, and for evaluation we need the complete model parameters 
            # to compute perplexity and generate samples.
            print(f"DEBUG: Loading full model state dict from DeepSpeed checkpoint for {checkpoint_tag}...")
            state_dict = get_fp32_state_dict_from_zero_checkpoint(
                checkpoint_dir, checkpoint_tag
            )
            print(f"DEBUG: State dict loaded for {checkpoint_tag}. Keys: {list(state_dict.keys())[:5]}...")

        # Initialize the model and load the state dict. 
        # We use the same config for all checkpoints since they are from the same training run.
        model = AutoModelForCausalLM.from_config(MODEL_CONFIG)
        model.load_state_dict(state_dict)
        model.to(device).half().eval()

 
        # Metrics
        print(f"[Rank {rank}] Running PPL for {checkpoint_tag}...", flush=True)
        perplexity = _compute_perplexity(model, encodings, STRIDE, MAX_LENGTH)
        
        print(f"[Rank {rank}] Generating sample for {checkpoint_tag}...", flush=True)
        generated_response = _generate_sample(model, tokenizer, TEST_PROMPT)

        local_results.append({
            "checkpoint": checkpoint_tag,
            "perplexity": perplexity,
            "sample": {"prompt": TEST_PROMPT, "response": generated_response},
        })
 
        # Free GPU memory before loading the next checkpoint.
        del model, state_dict
        torch.cuda.empty_cache()
        gc.collect()

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
    # print(f"DESTROY DONE: {socket.gethostname()} rank {dist.get_rank()}")
    print(f"exiting...")

    
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