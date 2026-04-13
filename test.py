import argparse
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
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int)
parser.add_argument("--outputfile", type=str)
args = parser.parse_args()

STAGE = args.stage
OUTPUTFILE = args.outputfile


os.makedirs(
    f"/scratch/chethan1/SSDS/llm_training/outputs/{STAGE}", exist_ok=True)
os.environ.setdefault("DEEPSPEED_USE_SOFT_ADAM", "1")
BASE_DIR = os.path.abspath(
    f"/scratch/chethan1/SSDS/llm_training/outputs/stage_{STAGE}")
DATA_DIR = os.path.abspath(f"/scratch/chethan1/SSDS/llm_training/outputs/")
SCRATCH_DIR = os.path.abspath("/scratch/chethan1/SSDS/llm_training/results")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)


# DeepSpeed config builder
def build_deepspeed_config(stage: int, micro_batch_size: int, gradient_accumulation: int) -> dict:
    """
    Build a DeepSpeed configuration dictionary based on the specified ZeRO stage and training parameters.
    """

    # Hyper parameters are from standard GPT-2 training configs
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "train_batch_size": micro_batch_size * gradient_accumulation,
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
        "wall_clock_breakdown": True,  # enables detailed timing breakdown in DeepSpeed logs

        # This will profile the FLOPs at the specified step, giving us an estimate of
        # the model's computational load and efficiency in TFLOPS.
        "flops_profiler": {
            "enabled": True,
            "output_file": f"{SCRATCH_DIR}/reflops_profile_{stage}.txt",
            # profiles FLOPs at step 10 for an estimate of model FLOPs.
            "profile_step": 2,
            "module_depth": 0,  # Set to 0 for overall model FLOPs, or higher for layer-wise breakdown
            "detailed": False,  # Set to True for a more detailed breakdown of FLOPs by module
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
            "enabled": False,
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
        },
        "runtime": {
            "timers": {
                "enabled": True
            }
        },
        # This will give us a detailed breakdown of time spent in different parts of the training loop,
        # which is crucial for analyzing the communication overhead in ZeRO stages.
        "timers": {
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
    print(f"\n{'='*80}")
    print(f"Training ZeRO Stage {STAGE} on GPUS with DeepSpeed")
    print(f"{'='*80}")
    deepspeed.init_distributed(dist_backend="nccl")

    global_rank = dist.get_rank()

    ds_config = build_deepspeed_config(STAGE, 4, 8)

    # Define model
    model = torch.nn.Linear(1024, 1024)

    # CHANGE 4: Initialize DeepSpeed without forcing a GPU
    # DeepSpeed will look at the config and the model location.
    deep_speed_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        
    )
    print(f"DeepSpeed initialized on {deep_speed_engine.device} for global rank {global_rank} local rank {deep_speed_engine.local_rank}.")
    RANK_OUTPUTFILE = f"{OUTPUTFILE.replace('.txt', '')}_rank_{global_rank}.txt"
    for step in range(10):
        # CHANGE 5: Ensure tensors are on CPU
        inputs = torch.randn(4, 1024).to(deep_speed_engine.device)  

        outputs = deep_speed_engine(inputs)
        loss = outputs.mean()
        deep_speed_engine.backward(loss)
        deep_speed_engine.step()

        if global_rank == 0:
            print(f"Step {step} completed on {deep_speed_engine.device}.")

            print(f"Logging output to {RANK_OUTPUTFILE}")
            with open(RANK_OUTPUTFILE, "a") as f:
                f.write(f"Rank {global_rank} completed step {step} on {deep_speed_engine.device}\n")

    dist.barrier()


if __name__ == "__main__":
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
    step_3_training()
