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
import gc
import pynvml  # For hardware-level VRAM
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
import tqdm
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
parser.add_argument("--check_point_tag", type=str, default=None)
args = parser.parse_args()

STAGE = args.stage
OUTPUTFILE = args.outputfile
CHECKPOINT_TAG = args.check_point_tag


OUTPUTS = [
    f"/scratch/chethan1/SSDS/llm_training/outputs/stage_0/checkpoints/",
    f"/scratch/chethan1/SSDS/llm_training/outputs/stage_1/checkpoints/",
    f"/scratch/chethan1/SSDS/llm_training/outputs/stage_2/checkpoints/",
    f"/scratch/chethan1/SSDS/llm_training/outputs/stage_3/checkpoints/",
]

DS_CONFIG_PATH = f"/scratch/chethan1/SSDS/llm_training/outputs/stage_{STAGE}/ds_config.json"
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

def build_deepspeed_config(
    stage: int,
    micro_batch_size: int,
    gradient_accumulation: int,
    total_num_steps: int,
    world_size: int,
    scratch_dir: str
) -> dict:
    ds_config = {
        # =========================
        # BATCHING (CRITICAL)
        # =========================
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,

        # =========================
        # OPTIMIZER (GPT-style)
        # =========================
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1.5e-5,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
        },

        # =========================
        # LR SCHEDULER (IMPORTANT)
        # =========================
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 1e-7,
                "warmup_max_lr": 1.5e-5,
                "warmup_num_steps": int(0.01 * total_num_steps),  # 1% warmup
                "total_num_steps": total_num_steps,
            },
        },

        # =========================
        # NUMERICS
        # =========================
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 12,
        },

        "gradient_clipping": 1.0,
        # =========================
        # LOGGING (VERY IMPORTANT)
        # =========================
        "steps_per_print": 1,
        "wall_clock_breakdown": True,
        "monitor": {
            "enabled": True,
            "tag": f"zero_stage_{stage}",
            "tensorboard": {
                "enabled": True,
                "output_path": f"{scratch_dir}/tensorboard",
                "job_name": f"zero_stage_{stage}"
            },
            "csv_monitor": {
                "enabled": True,
                "output_file": f"{scratch_dir}/outputs/stage_{stage}/csv_logs_stage_{stage}.csv",
                "job_name": f"zero_stage_{stage}"
            }
        },

        # =========================
        # PROFILING
        # =========================
        "flops_profiler": {
            "enabled": True,
            "profile_step": 20,
            "module_depth": -1,
            "top_modules": 3,
            "detailed": True,
            "output_file": f"{scratch_dir}/outputs/stage_{stage}/flops_stage_{stage}_{str(time.time())}.txt"
        },

        "comms_logger": {
            "enabled": True,
            "prof_all": True,
            "verbose": False,
        },

        "timers": {
            "enabled": True
        },
    }

    # =========================
    # ZeRO OPTIMIZATION
    # =========================
    if stage > 0:
        ds_config["zero_optimization"] = {
            "stage": stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": int(2e8),
            "allgather_bucket_size": int(2e8),
        }

    if stage >= 2:
        ds_config["zero_optimization"]["allgather_partitions"] = True

    if stage == 3:
        ds_config["zero_optimization"].update({
            "sub_group_size": int(1e9),
            "stage3_prefetch_bucket_size": int(5e8),
            "stage3_param_persistence_threshold": int(1e5),
            # add offload_optimizer / offload_param here only when experimenting with CPU offload
        })

    return ds_config


def step_3_training():
    """
    Train GPT-2-mini model from scratch using DeepSpeed with SLURM-based distributed initialization.
    """

    # ── Output directories ──
    now_dt = datetime.datetime.now()
    date_str = now_dt.strftime("%Y-%m-%d")
    time_str = now_dt.strftime("%H-%M-%S")
    # checkpoint_dir = os.path.join(BASE_DIR, "checkpoints", date_str, time_str)

    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    hf_output_dir  = os.path.join(BASE_DIR, "gpt2_trained")

    # date_str = "2026-04-14"
    # time_str = "02-17-44"

    # return "/scratch/chethan1/SSDS/llm_training/outputs/stage_0/checkpoints/2026-04-16/01-15-28/"
    # if STAGE == 3:
    #     return os.path.join(BASE_DIR, date_str, time_str)
    # return os.path.join(BASE_DIR, "gpt2_trained", date_str, time_str)
    print(f"DEBUG: \n{'='*80}")
    print(f"DEBUG: Training ZeRO Stage {STAGE}")
    print(f"DEBUG: {'='*80}")

    # =========================
    # ENV + DEVICE SETUP
    # =========================

    # Initialize distributed FIRST
    if not dist.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")


    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    GLOBAL_RANK = os.environ.get("RANK")
    global_rank = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    torch.cuda.set_device(LOCAL_RANK)
    print(
        f"DEBUG: RANK={GLOBAL_RANK}"
        f"DEBUG: LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
        f"DEBUG: WORLD_SIZE={os.environ.get('WORLD_SIZE')}",
        flush=True
    )
    # Now safe to query
    

    print(f"DEBUG: World Size: {WORLD_SIZE}, Local Rank: {LOCAL_RANK}")

    # =========================
    # TRAINING PARAMS
    # =========================
    TOTAL_NUM_STEPS = 1120
    GRADIENT_ACCUMULATION = 8
    MICRO_BATCH_SIZE = 16
    CHECKPOINT_FOR_STEP = 64
    PROFILE_PER_STEP = 4
    NUM_WORKERS = 2

    metrics_log = []

    # =========================
    # NVML INIT
    # =========================
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    # =========================
    # CONFIG (FIXED)
    # =========================
    print("DEBUG: Generating DeepSpeed configuration...")

    ds_config = build_deepspeed_config(
        stage=STAGE,
        micro_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        total_num_steps=TOTAL_NUM_STEPS,
        world_size=WORLD_SIZE,
        scratch_dir=SCRATCH_DIR
    )

    # =========================
    # DATASET
    # =========================
    print("DEBUG: Loading dataset...")
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
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    
    # =========================
    # MODEL
    # =========================
    print("DEBUG: Initializing model...")
    model_config = AutoConfig.from_pretrained(GPT2_CONFIG_DIR)

    if not hasattr(model_config, "loss_type"):
        model_config.loss_type = "ForCausalLMLoss" 
        
    model = AutoModelForCausalLM.from_config(model_config)

    # =========================
    # DEEPSPEED INIT
    # =========================
    print("DEBUG: Initializing DeepSpeed engine...")
    deep_speed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
    
    print("DEBUG: Checking for checkpoint...")

    out_file = os.path.join(SCRATCH_DIR, f"stage_{STAGE}_training_log_{str(time.time())}.txt")

    # =========================
    # RESUME LOGIC 
    # =========================
    start_step = 0
    start_epoch = 0

    resume_dir = checkpoint_dir

    if os.path.exists(resume_dir):
        load_path, client_state = deep_speed_engine.load_checkpoint(resume_dir,tag=CHECKPOINT_TAG)

        if load_path is not None:
            print(f"DEBUG: [Rank {dist.get_rank()}] Resumed from {load_path}")

            if client_state:
                start_step = client_state.get("step", 0)
                start_epoch = client_state.get("epoch", 0)

            print(f"DEBUG: [Rank {dist.get_rank()}] step={start_step}, epoch={start_epoch}")
        else:
            print(f"DEBUG: [Rank {dist.get_rank()}] No checkpoint found → starting fresh")
    else:
        print(f"DEBUG: [Rank {dist.get_rank()}] Resume dir not found → starting fresh")

    # =========================
    # TRAIN STATE
    # =========================
    step = start_step
    epoch = start_epoch
    micro_step = 0
    total_tokens = 0
    total_start_time = time.perf_counter()

    data_iter = iter(train_dataloader)

    # ============================
    # TIMERS
    # ============================
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end   = torch.cuda.Event(enable_timing=True)

    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end   = torch.cuda.Event(enable_timing=True)

    step_start = torch.cuda.Event(enable_timing=True)
    step_end   = torch.cuda.Event(enable_timing=True)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    # ============================
    # TRAINING LOOP
    # ============================
    while step < TOTAL_NUM_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        # Move batch to GPU (non-blocking)
        inputs = {k: v.to(deep_speed_engine.device, non_blocking=True) for k, v in batch.items()}
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        stream = torch.cuda.current_stream()

        torch.cuda.synchronize()
        iter_start.record(stream)

        # ============================
        # FORWARD
        # ============================
        fwd_start.record(stream)
        outputs = deep_speed_engine(**inputs)
        loss = outputs.loss

        torch.cuda.synchronize()
        fwd_end.record(stream)

        # ============================
        # BACKWARD
        # ============================
        bwd_start.record(stream)
        deep_speed_engine.backward(loss)

        torch.cuda.synchronize()
        bwd_end.record(stream)

        micro_step += 1

        # ============================
        # STEP (COMMUNICATION HAPPENS HERE)
        # ============================
        if micro_step % GRADIENT_ACCUMULATION == 0:
            step_start.record(stream)
            deep_speed_engine.step()
            step_end.record(stream)

            iter_end.record(stream)
            torch.cuda.synchronize()

            # ============================
            # TIMINGS (ms)
            # ============================
            fwd_t = fwd_start.elapsed_time(fwd_end)
            bwd_t = bwd_start.elapsed_time(bwd_end)
            step_t = step_start.elapsed_time(step_end)
            iter_t = iter_start.elapsed_time(iter_end)

            compute_t = fwd_t + bwd_t
            comm_t = step_t   # Best approximation

            # ============================
            # THROUGHPUT
            # ============================
            tokens_per_iter = MICRO_BATCH_SIZE * WORLD_SIZE * GRADIENT_ACCUMULATION * block_size
            tokens_per_sec = tokens_per_iter / (iter_t / 1000)

            total_tokens += tokens_per_iter

            # ============================
            # GPU STATS
            # ============================
            if (global_rank == 0 and (step % PROFILE_PER_STEP == 0 or step == 0)) or (step == TOTAL_NUM_STEPS - 1):

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                allocated_Gb = torch.cuda.memory_allocated() / (1024**3)
                reserved_Gb = torch.cuda.memory_reserved() / (1024**3)
                peak_vram_Gb = mem_info.used / (1024**3)

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                lr = deep_speed_engine.get_lr()[0] if deep_speed_engine.get_lr() else 0.0

                # ============================
                # METRICS
                # ============================
                metrics = {
                    "stage": STAGE,
                    "step": step,
                    "loss": loss.item(),
                    "lr": lr,

                    # Timing
                    "iteration_time_ms": iter_t,
                    "fwd_ms": fwd_t,
                    "bwd_ms": bwd_t,
                    "step_ms": step_t,

                    # Derived
                    "compute_ms": compute_t,
                    "comm_ms": comm_t,
                    "comm_pct": (comm_t / (compute_t + comm_t + 1e-8)) * 100,

                    # Performance
                    "throughput_tps": tokens_per_sec,

                    # GPU
                    "gpu_util": util.gpu,
                    "temp_c": temp_c,

                    # Memory
                    "alloc_gb": allocated_Gb,
                    "reserved_gb": reserved_Gb,
                    "peak_gb": peak_vram_Gb,
                }

                with open(OUTPUTFILE, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                metrics_log.append(metrics)

                print(
                    f"[Stage {STAGE} | Step {step}] "
                    f"Loss={loss.item():.4f} | "
                    f"Iter={iter_t:.2f}ms | "
                    f"FWD={fwd_t:.2f} | BWD={bwd_t:.2f} | STEP={step_t:.2f} | "
                    f"Comm%={metrics['comm_pct']:.2f} | "
                    f"TPS={tokens_per_sec:.2f} | "
                    f"VRAM={peak_vram_Gb:.2f}GB | Util={util.gpu}%"
                )

            # ============================
            # CHECKPOINT
            # ============================
            if step > 0 and step % CHECKPOINT_FOR_STEP == 0:
                deep_speed_engine.save_checkpoint(
                    checkpoint_dir,
                    tag=f"global_step{step}",
                    client_state={
                        "step": step,
                        "epoch": epoch
                    }
                )
                print(f"DEBUG: Checkpoint saved at step {step} to {checkpoint_dir}")

            step += 1
            micro_step = 0
            
            # Progress indicator
            if global_rank == 0 and step % 10 == 0:
                print(f"DEBUG: Progress: {step}/{TOTAL_NUM_STEPS} completed")
    
    # ============================
    # SYNC ALL RANKS
    # ============================
    dist.barrier()

    # ============================
    # TRAINING TIME
    # ============================
    total_training_time = time.perf_counter() - total_start_time

    # ============================
    # GATHER METRICS FROM ALL GPUS
    # ============================
    all_metrics = [None] * WORLD_SIZE
    dist.all_gather_object(all_metrics, metrics_log)

    if global_rank == 0:
        merged = []
        for m in all_metrics:
            if m:
                merged.extend(m)

        print(f"\n{'#'*20} FINAL STAGE {STAGE} REPORT {'#'*20}")
        print(f"Total Training Time: {total_training_time:.2f} seconds")
        print(f"Total Tokens Processed: {total_tokens}")
        print(f"Total Steps: {step}")
        print(f"Total weight updates: {TOTAL_NUM_STEPS // GRADIENT_ACCUMULATION}")
        print(f"{'#'*50}")

        metrics_file = os.path.join(SCRATCH_DIR, f"stage_{STAGE}_metrics_{str(time.time())}.json")
        with open(metrics_file, "w") as f:
            json.dump(merged, f, indent=2)

    # ============================
    # SAVE MODEL
    # ============================
    # Save a final checkpoint (optional, but good for resuming or evaluation)
    deep_speed_engine.save_checkpoint(
                    checkpoint_dir,
                    tag=f"global_step{step}",
                    client_state={
                        "step": step,
                        "epoch": epoch
                    }
                )
               
    if global_rank == 0:
        os.makedirs(hf_output_dir, exist_ok=True)

        if STAGE == 0:
            deep_speed_engine.module.save_pretrained(hf_output_dir)
        else:
            state_dict = get_fp32_state_dict_from_zero_checkpoint(
                checkpoint_dir,
                tag=f"global_step{step}"
            )
            config = AutoConfig.from_pretrained(GPT2_CONFIG_DIR)
            if not hasattr(config, "loss_type"):
                config.loss_type = "ForCausalLMLoss"
            model = AutoModelForCausalLM.from_config(config)
            model.load_state_dict(state_dict)
            model.save_pretrained(hf_output_dir)

        print(f"Model saved to {hf_output_dir}")

    # ============================
    # CLEANUP
    # ============================
    del deep_speed_engine
    del model

    torch.cuda.empty_cache()

    dist.barrier()

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

# =========================================================
# DATA PREPARATION
# =========================================================
def _concat_input_ids(dataset) -> torch.Tensor:
    """Concatenate 'input_ids' from dataset into a single 1-D tensor."""
    print("DEBUG: Concatenating dataset...", flush=True)
    rows = dataset["input_ids"]
    parts = [torch.tensor(row, dtype=torch.long) if not isinstance(row, torch.Tensor) 
             else row.reshape(-1).long() for row in rows]
    res = torch.cat(parts)
    return res

# =========================================================
# PERPLEXITY (OPTIMIZED SLIDING WINDOW)
# =========================================================
def _compute_perplexity(model, encodings, rank, stride=512, max_length=512):
    """Computes total loss and token count for local encodings."""
    device = next(model.parameters()).device
    seq_len = encodings.size(0)
    total_loss = 0.0
    total_tokens = 0

    # Ensure we have enough tokens to run at least one window
    if seq_len < max_length:
        return 0.0, 0

    for i in tqdm.tqdm(
        range(0, seq_len, stride), 
        desc=f"GPU {rank}", 
        position=rank,  # Stacks bars vertically based on GPU ID
        leave=False,     # Clears the bar when that checkpoint is done
        dynamic_ncols=True,
        disable=(rank != 0)  # Only show progress bar on rank 0 to reduce clutter
    ):
        begin = max(i + stride - max_length, 0)
        end = min(seq_len, i + stride)
        trg_len = end - i  # Tokens being predicted in this window

        input_ids = encodings[begin:end].unsqueeze(0).to(device)
        target_ids = input_ids.clone()
        
        # We only want to compute loss on the 'new' tokens (the stride)
        # Mask labels that are part of the 'context' but not the 'target'
        target_ids[:, :-trg_len] = -100

        # Create a mask of 1s (since we aren't using padding tokens in our concat logic)
        attn_mask = torch.ones_like(input_ids).to(device)

        outputs = model(input_ids, attention_mask=attn_mask, labels=target_ids)
        loss = outputs.loss.detach().cpu().float()
        
        total_loss += loss.item() * trg_len
        total_tokens += trg_len

    return total_loss, total_tokens

# =========================================================
# TEXT GENERATION
# =========================================================
def _generate_sample(model, tokenizer, prompt):
    """Generate 50 tokens to qualitatively assess model quality."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# =========================================================
# MAIN EVALUATION FUNCTION
# =========================================================
def step_4_evaluation(checkpoint_dir):
    if not dist.is_initialized():
        print("Distributed environment not found. Please run via deepspeed/torchrun.")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # ---------------- CONFIG ----------------
    STRIDE = 512
    MAX_LENGTH = 512
    PROMPT = "Once upon a time"
    RESULTS_PATH = os.path.join(checkpoint_dir, f"perplexity_results_{str(time.time())}.json")
    TOKENIZER_DIR = "/mnt/data/ds256_2026/as2/gpt2_tokenizer"

    # ---------------- LOAD RESOURCES ----------------
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(GPT2_CONFIG_DIR)
    if not hasattr(config, "loss_type"):
        config.loss_type = "ForCausalLMLoss" 
    
    dataset = load_from_disk(final_test_dataset)
    all_encodings = _concat_input_ids(dataset)

    # ---------------- PARTITION DATA & CHECKPOINTS ----------------
    # Assign specific checkpoints to this rank
    all_checkpoints = sorted([d for d in os.listdir(checkpoint_dir) 
                             if os.path.isdir(os.path.join(checkpoint_dir, d))])
    my_checkpoints = [ckpt for i, ckpt in enumerate(all_checkpoints) if i % world_size == rank]

    local_results = []

    # ---------------- EVALUATION LOOP ----------------
    for ckpt in my_checkpoints:
        print(f"[Rank {rank}] Evaluating {ckpt}...")
        
        # Load Model Weights
        if STAGE == 0:
            ckpt_path = os.path.join(checkpoint_dir, ckpt, "mp_rank_00_model_states.pt")
            state = torch.load(ckpt_path, map_location="cpu")
            state_dict = state["module"] if "module" in state else state
        else:
            # For ZeRO-1/2/3, we reconstruct the FP32 weights
            state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag=ckpt)

        model = AutoModelForCausalLM.from_config(config)
        model.load_state_dict(state_dict)
        model.to(device).half().eval()

        with torch.inference_mode():
            # 1. Perplexity Calculation
            # We use the full test dataset here for consistency across ranks 
            # (or partition all_encodings if the dataset is massive)
            loss_sum, token_count = _compute_perplexity(model, all_encodings, rank, STRIDE, MAX_LENGTH)
            avg_loss = loss_sum / token_count if token_count > 0 else 0
            ppl = float(np.exp(avg_loss))

            # 2. Generation Sample
            response = _generate_sample(model, tokenizer, PROMPT)

        local_results.append({
            "checkpoint": ckpt,
            "perplexity": round(ppl, 4),
            "sample": {
                "prompt": PROMPT,
                "response": response
            }
        })

        # Memory Cleanup
        del model, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------- AGGREGATION ----------------
    # Gather results from all ranks
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_results)

    if rank == 0:
        # Merge list of lists into one final list
        final_list = [item for sublist in gathered_results if sublist for item in sublist]
        final_list.sort(key=lambda x: x["checkpoint"])

        with open(RESULTS_PATH, "w") as f:
            json.dump(final_list, f, indent=4)

        print("\n" + "="*30)
        print("EVALUATION COMPLETE")
        for res in final_list:
            print(f"Checkpoint: {res['checkpoint']} | PPL: {res['perplexity']}")
        print("="*30)

    dist.barrier()

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

    # # Step 4: Evaluate model and checkpoints
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