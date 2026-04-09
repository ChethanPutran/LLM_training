# DS256 - Assignment 2
## LLM Distributed Training with DeepSpeed
## 150 points (15% weightage)
## v1.0, posted on 2026-04-03
## Deadline: 2026-04-17

---

## Overview

This assignment builds a complete pipeline for training a GPT-style language model from scratch on large-scale text data. Starting from deduplicated data (Assignment 1), you will progressively transform raw text into a format suitable for efficient distributed training, train the model using DeepSpeed, and evaluate both performance and model quality.

The pipeline consists of four tightly coupled stages:

1. **Tokenization** — Convert raw text into token sequences  
2. **Preprocessing** — Construct fixed-length training samples  
3. **Distributed Training** — Train the model using DeepSpeed and analyze system performance  
4. **Evaluation** — Measure model quality and track training progression  

Each stage is designed to expose a different systems challenge: data processing, batching efficiency, distributed execution, and performance analysis.

## Evaluation and Weightage

This assignment is primarily a **systems analysis exercise**, not just an implementation task. It will graded as follows:
- **Code (Steps 1 and 2): 25%**
- **Report (Steps 3 and 4): 75%**


### Report Expectations

The report will be the primary basis of evaluation. You are expected to:

- Present **quantitative results** across all required experiments  
- Provide **clear, data-backed analysis** of system behavior  
- Demonstrate understanding of:
  - Memory–communication tradeoffs  
  - Scaling behavior across ZeRO stages  
  - Performance bottlenecks  

Superficial observations or claims without supporting data will be penalized.

### Code Expectations

Your code will be evaluated on:

- Correctness (pipeline runs end-to-end)
- Adherence to instructions (no unauthorized modifications)
- Reproducibility (results can be regenerated)
- Time taken and exact output match for the first 2 stages, and model performance for the latter 2 stages.
---

## General Rules

- Modify only explicitly marked regions in the code
- Do not alter dataset schema or pipeline structure
- All intermediate outputs must be saved to disk
- Each stage must pass validation before proceeding

Failure to follow these constraints will break downstream components.

---

# Step 1: Tokenization

## Objective

Transform raw text into token sequences using a GPT tokenizer so that the data becomes consumable by neural networks.

---

## Description

You will load the dataset in parquet format and convert each text sample into a sequence of token IDs using a GPT-2 tokenizer. Since language models rely on sequence boundaries to learn meaningful structure, each sample must be explicitly terminated using an end-of-sequence (EOS) token.

Tokenization must be implemented efficiently, as it is a CPU-bound step and operates over large datasets. The output should be stored on disk to avoid recomputation in later stages. This stage will be evaluated on `time taken` and number of `tokens` generated (exact output match expected, similar to Assignment 1).

---

## Expected Output

Each sample must contain:
- `input_ids`: tokenized representation of text  

All raw text fields must be removed, and the dataset must conform to the expected schema.

---

# Step 2: Preprocessing

## Objective

Convert variable-length token sequences into fixed-length blocks to enable efficient GPU training.

---

## Description

Transformer models require inputs of uniform size for efficient batching. Instead of padding individual sequences (which wastes computation), you will concatenate token sequences into a continuous stream and divide them into fixed-size chunks.

Each chunk represents a training example of length `block_size` (e.g., 512). Any leftover tokens that do not form a full block should be discarded. For causal language modeling, the input sequence itself serves as the training target.

This step is critical for maximizing GPU utilization and ensuring stable training behavior. 

This stage will be evaluated on `time taken` and on an exact output match, similar to Assignment 1.

---

## Expected Output

Each sample must contain:
- `input_ids`: sequence of length `block_size`  
- `labels`: identical copy of `input_ids`  

The dataset must load correctly into a DataLoader and produce tensors of consistent shape.

---

# Step 3: Distributed Training with DeepSpeed

## Objective

Train a GPT-style model (~40M parameters) from scratch using distributed GPUs and analyze the impact of system-level optimizations.

---

## Description

You will initialize a causal language model from configuration and train it using DeepSpeed in a distributed environment (SLURM-based). DeepSpeed manages gradient synchronization, memory partitioning, and communication across GPUs.

The training loop must correctly integrate DeepSpeed’s execution model, including forward pass, backward propagation, and parameter updates. In addition to training, you are expected to monitor system performance and log relevant metrics throughout execution.

Checkpointing is mandatory to ensure fault tolerance and enable later evaluation.

---

## Performance Logging

You are expected to track:

- Throughput (tokens/sec)
- GPU utilization
- Memory usage (VRAM)
- Compute time (forward + backward)
- Communication time and volume

These metrics are essential for understanding system bottlenecks.

---

## ZeRO Optimization Analysis (Mandatory)

### Objective

Evaluate how different ZeRO stages affect memory efficiency, communication overhead, and overall training performance.

---

### Description

ZeRO reduces memory usage by partitioning:

- Optimizer states (Stage 1)
- Gradients (Stage 2)
- Model parameters (Stage 3)

While this improves memory scalability, it introduces communication overhead. You must empirically study this tradeoff.

---

### Experimental Setup

You must run training under:

- ZeRO Stage 0 (baseline)
- ZeRO Stage 1
- ZeRO Stage 2
- ZeRO Stage 3

Each configuration must be evaluated independently. You may run other forms of parallelism, in addition to these.

---

### Metrics

For each run, you MAY collect:

- Total training time  
- Compute time  
- Communication time  
- Throughput  
- GPU memory usage  

You may need to instrument DeepSpeed or PyTorch internals to obtain accurate communication measurements.

---

### Report Requirements

Your report must include:

1. **Quantitative comparison** across all ZeRO stages  
2. **Analysis of tradeoffs**, including:
   - Communication scaling with ZeRO stage  
   - Memory vs runtime tradeoff  
   - Point of diminishing returns  
   - Best configuration for your setup  

Conclusions must be supported by data.

---

## Expected Insight

You are expected to observe that:

- Lower ZeRO stages are faster but memory-intensive  
- Higher ZeRO stages reduce memory but increase communication  
- Optimal performance depends on hardware and workload  

---

# Step 4: Model and Checkpoint Evaluation

## Objective

Evaluate both the quality and progression of the trained model using quantitative and qualitative methods.

---

## Description

You will evaluate multiple checkpoints saved during training to understand how the model evolves. Evaluation consists of computing perplexity on a held-out dataset and generating text samples to assess output quality.

To ensure efficiency, evaluation should be parallelized across available ranks.

---

## Evaluation Procedure

For each selected checkpoint:

- Compute **perplexity** using a sliding window approach (stride = 512)  
- Generate a short text continuation from a fixed prompt  

Using consistent prompts across checkpoints is necessary for fair comparison.

---

## Output Format

Results must be aggregated and saved as a JSON file:

- Checkpoint identifier  
- Perplexity score  
- Generated sample (prompt + response)  

---

## Expected Outcome

You should observe:

- Perplexity decreasing over training  
- Gradual improvement in text coherence  
- Potential plateau or overfitting at later stages  

---

## Final Outcome of Assignment

By completing this assignment, you will:

- Build an end-to-end LLM training pipeline  
- Understand preprocessing tradeoffs for large-scale data  
- Gain experience with distributed training systems  
- Analyze memory–communication tradeoffs in practice  
- Evaluate model quality beyond training loss  

---

