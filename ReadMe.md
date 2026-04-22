# LLM Distributed Training with DeepSpeed

## Project Overview

This project implements distributed training of a GPT-2 Mini model from scratch using DeepSpeed's ZeRO optimization stages on a SLURM-based HPC cluster. The implementation includes data preprocessing, distributed training with comprehensive performance profiling, and model evaluation.

## Repository Structure

```
├── a2p1_v0.1.py          # Part 1: Data preprocessing (tokenization & chunking)
├── a2p2_v0.1.py          # Part 2: Distributed training & evaluation
├── run_part2.sh          # SLURM launcher for multi-stage training
├── requirements.txt      # Python dependencies
├── outputs/              # Training outputs and logs
├── results/              # Metrics and evaluation results
├── logs/                 # SLURM job logs
├── .gitignore            # Git ignore rules
└── ReadMe.md             # This file
```

## Pipeline Steps

### Part 1: Data Preprocessing
- **Tokenization**: Raw text tokenization using GPT-2 tokenizer with EOS token handling
- **Chunking**: Token sequences grouped into fixed-length blocks (512 tokens)

### Part 2: Distributed Training & Evaluation
- **Step 3**: Distributed training with DeepSpeed (ZeRO Stages 0-3)
- **Step 4**: Model evaluation (perplexity calculation & text generation)

## Hardware Requirements

- **Nodes**: 4 NVIDIA GPU nodes
- **GPUs per node**: 1 (total 4 GPUs)
- **Memory**: Sufficient VRAM for model + optimizer states
- **Storage**: Local scratch space for checkpoints and logs

## Installation

### 1. Activate Conda Environment

```bash
source /mnt/data/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/miniconda3/envs/chethan1
```

### 2. Install Dependencies

```bash
# Install PyTorch (CPU-only for head node, GPUs available on compute nodes)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install DeepSpeed and monitoring tools
conda install -c conda-forge deepspeed nvidia-ml-py ninja mpi4py tensorboard

# Install transformers and datasets
conda install -c conda-forge transformers datasets

# Install modern GCC for DeepSpeed JIT compilation
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11
```

### 3. Verify Installation

```bash
python -c "import deepspeed; print(deepspeed.__version__)"
python -c "import torch; print(torch.__version__)"
```

## Configuration

### Path Configuration (MODIFY BEFORE RUNNING)

Update the following paths in both scripts:

**a2p1_v0.1.py:**
```python
BASE_DIR = os.path.abspath("/scratch/YOUR_USERNAME/SSDS/llm_training/outputs")
```

**a2p2_v0.1.py:**
```python
# Update scratch directory paths
SCRATCH_DIR = os.path.abspath("/scratch/YOUR_USERNAME/SSDS/llm_training/results")
DATA_DIR = os.path.abspath("/scratch/YOUR_USERNAME/SSDS/llm_training/outputs/")
```

**run_part2.sh:**
```bash
# Update reservation name
--reservation=YOUR_TEAM_RESERVATION_NAME

# Update output paths
OUTPUT_DIR="/scratch/YOUR_USERNAME/SSDS/llm_training/outputs"
CHECKPOINT_DIR="/scratch/YOUR_USERNAME/SSDS/llm_training/outputs/checkpoints"
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Block Size | 512 |
| Micro Batch Size | 16 |
| Gradient Accumulation | 8 |
| Total Training Steps | 1120 |
| Learning Rate | 1.5e-5 |
| Warmup Steps | 1% of total |

## Running the Pipeline

### Step 1: Data Preprocessing (Single Node)

```bash
# Check available slot
/apps/myslot.sh

# Run preprocessing on single node
srun -N 1 \
    --ntasks=1 \
    --partition=ds256 \
    --qos=ds256_qos \
    --reservation=YOUR_RESERVATION \
    -t 01:00:00 \
    /apps/run_wrapper.sh a2p1_v0.1.py
```

### Step 2: Distributed Training & Evaluation (Multi-Node)

```bash
# Make launcher executable
chmod +x run_part2.sh

# Run all ZeRO stages sequentially
./run_part2.sh
```

The launcher will execute ZeRO Stages 0, 1, 2, and 3 in sequence, automatically managing time allocation for each stage.

## Monitoring & Logging

### Real-time Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Tail training logs
tail -f logs/stage_*_output.log

# Monitor TensorBoard
tensorboard --logdir results/tensorboard --port 6006
```

### Output Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Training Metrics | `results/stage_{N}_metrics_*.json` | Per-step performance metrics |
| CSV Logs | `results/outputs/stage_{N}/csv_logs_stage_{N}.csv` | DeepSpeed CSV monitoring |
| TensorBoard Logs | `results/tensorboard/` | Visual metrics |
| Model Checkpoints | `outputs/stage_{N}/checkpoints/` | ZeRO checkpoint files |
| Final Model | `outputs/stage_{N}/gpt2_trained/` | HuggingFace format model |
| Evaluation Results | `checkpoints/perplexity_results_*.json` | Perplexity & generation samples |

### Metrics Collected

- **Timing**: Forward/backward/step/iteration time (ms)
- **Throughput**: Tokens per second
- **Memory**: Allocated, reserved, peak VRAM (GB)
- **Utilization**: GPU utilization (%)
- **Communication**: Communication time and percentage
- **Loss**: Training loss values

## Troubleshooting

### Common Issues & Solutions

| Error | Solution |
|-------|----------|
| `No space left on device` | Run `conda clean --all` and clear `~/.cache/deep_speed` |
| `GCC 9 or later required` | Install `gcc_linux-64=11` via conda |
| `Permission denied` | Run `chmod +x run_part2.sh` |
| `CUDA out of memory` | Reduce micro batch size or enable gradient checkpointing |
| `DeepSpeed JIT compilation failed` | Delete `~/.cache/torch_extensions` and re-run |
| `Slot not active` | Wait for your reserved time slot or check with `/apps/myslot.sh` |

### Storage Management

```bash
# Check quota
quota -s

# Clean conda cache
conda clean --all -y

# Clear DeepSpeed cache
rm -rf ~/.cache/deep_speed
rm -rf ~/.cache/torch_extensions

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
```

## ZeRO Stages Comparison

| Stage | Optimizer State | Gradient | Parameters | Communication |
|-------|-----------------|----------|------------|---------------|
| 0 | Full | Full | Full | Baseline |
| 1 | Partitioned | Full | Full | Moderate |
| 2 | Partitioned | Partitioned | Full | Higher |
| 3 | Partitioned | Partitioned | Partitioned | Highest |

## Expected Output

### Training Progress
```
[Stage 1 | Step 64] Loss=3.2451 | Iter=245.32ms | FWD=85.12 | BWD=92.45 | STEP=67.75 | Comm%=27.6% | TPS=26742.32 | VRAM=12.45GB | Util=94%
```

### Evaluation Results
```json
[
    {
        "checkpoint": "global_step64",
        "perplexity": 45.23,
        "sample": {
            "prompt": "Once upon a time",
            "response": "Once upon a time there was a young wizard who lived in a small village..."
        }
    }
]
```

## Academic Integrity

This project was completed as part of DS256 - Scalable Systems for Data Science (Jan 2026). The implementation follows all assignment guidelines and restrictions. No external sources, ChatGPT, or Copilot were used in the development of this code.

## Authors

Chethan 

## License

This code is for educational purposes only as part of the DS256 course at IISC.
