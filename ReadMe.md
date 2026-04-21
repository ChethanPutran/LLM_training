# LLM Training Setup on SLURM Cluster (DeepSpeed + CPU)

This project contains a testing framework for training Large Language Models (LLM) using DeepSpeed ZeRO stages on a CPU-based SLURM cluster.

## 1. Environment Initialization
The environment is hosted in a custom Miniconda path to ensure persistence across cluster nodes.

```bash
# Activate the specific conda environment
source /mnt/data/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/miniconda3/envs/chethan1
```

## 2. Installation Steps

### Core Dependencies
We used a mix of `pytorch` (CPU-optimized) and the `conda-forge` channel to ensure all C++ dependencies (like MPI and Ninja) are compatible.

```bash
# Install PyTorch for CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install DeepSpeed and monitoring tools
conda install -c conda-forge deepspeed nvidia-ml-py ninja mpi4py tensorboard
conda install -c conda-forge transformers datasets
```

### Compiler Upgrade (CRITICAL)
DeepSpeed requires JIT (Just-In-Time) compilation for its C++ extensions. The default system GCC is often too old (needs GCC 9+). We installed a modern toolchain directly into the environment:

```bash
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11
```

## 3. Storage & Quota Management
During installation, we encountered "No space left on device" (user block limit reached). Use these commands to maintain storage health:

```bash
# Check current quota status
quota -s

# Clean up conda installers (tarballs) to free ~1GB
conda clean --all

# Clear DeepSpeed and Torch JIT caches (important after failed builds)
rm -rf ~/.cache/deep_speed
rm -rf ~/.cache/torch_extensions
```

## 4. Cluster Execution (SLURM)
The training is launched across 2 nodes using `srun`.

### The Wrapper Script
Ensure the `run_wrapper.sh` is used to correctly point to the Python interpreter within the environment.

### Launching the Job
```bash
# Make the run script executable
chmod +x test_run.sh

# Submit/Run the job
./test_run.sh
```

## 5. Troubleshooting Reference

| Error | Fix |
| :--- | :--- |
| `ld: final link failed: No space left on device` | Clear `~/.cache/deep_speed` and run `conda clean --all`. |
| `error: #error "GCC 9 or later required"` | Install `gcc_linux-64=11` via conda as shown in Section 2. |
| `Permission denied` | Run `chmod +x test_run.sh`. |
| `ImportError: ... .so: No such file` | Delete `~/.cache/torch_extensions` and re-run to force a clean re-compile. |

## 6. Project Structure
- `test.py`: Main DeepSpeed training script.
- `test_run.sh`: SLURM batch/srun submission script.
- `outputs/`: Training logs and output text files.
- `results/`: TensorBoard logs and CSV metrics.