'''
## DS256 - Scalable Systems for Data Science | Jan 2026
# Assignment 2: LLM Distributed Training with DeepSpeed
# Part 1: Data Preprocessing

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

### Submission Guidelines (Part 1)
----
1. Run this script strictly on a **single node** as it handles CPU-bound tokenization. Launch configuration provided at the end of the script.
2. After implementing the steps, verify that the output passes the validation check.
3. Your final data will be saved to your `BASE_DIR`.

### Pipeline Steps (Part 1)
| Step | Name | Description |
|-------|------|-------------|
| 1 | Tokenization | Convert raw dataset text to tokenized format using GPT-2 Tokenizer|
| 2 | Preprocessing | Group tokenized sequences into blocks|
'''


# ─────────────────────────────────────────────
# START: DO NOT MODIFY THIS SECTION (IMPORT STATEMENTS, SHARED DIR, DATASET, MODEL)
# ─────────────────────────────────────────────
import os
import time
import multiprocessing
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, default_data_collator
import random

# Shared read-only directory containing the dataset
shared_dir = "/mnt/data/ds256_2026/as2" 
deduplicated_train_dataset_path = os.path.join(shared_dir, "train_dataset")
deduplicated_test_dataset_path = os.path.join(shared_dir, "test_dataset")
GPT2_TOKENIZER_DIR = os.path.join(shared_dir, "gpt2_tokenizer") 

# ──────────────────────────────────────────────────────
# END: DO NOT MODIFY THIS SECTION (IMPORT STATEMENTS, SHARED DIR, DATASET, MODEL)
# ──────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────
# START: OTHER IMPORT STATEMENTS (ALLOWED TO MODIFY)
# ──────────────────────────────────────────────────────

# Add your import statements here
import datasets
import glob
# ──────────────────────────────────────────────────────
# END: OTHER IMPORT STATEMENTS (ALLOWED TO MODIFY)
# ──────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────
# START: MODIFY THE BASE_DIR (Create these directories)
# ──────────────────────────────────────────────────────
# Local output directories (relative to working directory)
BASE_DIR = os.path.abspath("/scratch/chethan1/SSDS/llm_training/outputs")

out_train_dataset = os.path.join(BASE_DIR, "train_dataset")
out_test_dataset = os.path.join(BASE_DIR, "test_dataset")
tokenized_train_dataset = os.path.join(out_train_dataset, "tokenized")
final_train_dataset = os.path.join(out_train_dataset, "final")
tokenized_test_dataset = os.path.join(out_test_dataset, "tokenized")
final_test_dataset = os.path.join(out_test_dataset, "final")




# ──────────────────────────────────────────────────────
# END: MODIFY THE BASE_DIR (Create these directories)
# ──────────────────────────────────────────────────────

# Hyperparameters
block_size = 512

# ──────────────────────────────────────────────────────
# START: DO NOT MODIFY THIS SECTION (Schema helpers)
# ──────────────────────────────────────────────────────
STEP_1_SCHEMA = {
    'warc_id': 'string',
    'input_ids': 'array<int>'
}

def validate_schema(df, expected_columns):
    """Verifies that the dataframe contains the required columns with correct types."""
    import pandas as pd
    if isinstance(df, pd.DataFrame):
        df_cols = df.dtypes.to_dict()
        missing = [col for col in expected_columns if col not in df_cols]
        if missing:
            raise ValueError(f"Schema Validation Failed. Issues: {missing}")
        return

def validate_step_1(df):
    """Validate DataFrame schema for Step 1 (Tokenization)."""
    validate_schema(df, STEP_1_SCHEMA)
    print("STEP 1: Tokenization schema validation passed!")

def validate_step_2(train_data):
    """Validate DataLoader sanity for Step 2 (Preprocessing)."""
    print("Running DataLoader sanity check...")
    train_dataloader = DataLoader(
        train_data, shuffle=True, batch_size=1,
        collate_fn=default_data_collator
    )
    batch = next(iter(train_dataloader))
    print("Yielded input_ids shape:", batch["input_ids"].shape)
    
    assert batch["input_ids"].shape == (1, block_size), f"Expected shape (1, {block_size}), got {batch['input_ids'].shape}"
    print("STEP 2: Preprocessing (DataLoader sanity check) passed!")

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
# END: DO NOT MODIFY THIS SECTION (Schema helpers)
# ──────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────
# Step 1 - Tokenization (ALLOWED TO MODIFY)
# ──────────────────────────────────────────────────────
'''
## STEP 1: Tokenization
----

### Objective
Tokenize text using the standard GPT-2 tokenizer from Hugging Face and save the resulting dataset.

### Guidelines
1. The train dataset is present in the shared directory `/mnt/data/ds256_2026/as2/train_dataset` (read only).
   The test dataset is present in the shared directory `/mnt/data/ds256_2026/as2/test_dataset` (read only).
   The files are in the parquet format. The files are the deduplicated dataset (pre tokenization). Feel free to explore the dataset before implementing the tokenization.
2. Load the parquet train dataset from `deduplicated_train_dataset_path` and the test dataset from `deduplicated_test_dataset_path`.
3. Run the GPT-2 tokenizer over the text data.
4. Save the tokenized dataset to `tokenized_train_dataset` and the test dataset to `tokenized_test_dataset`.

### Hint
* Check the handling of the `eos_token`.
* The tokenizer is already loaded in the `tokenizer` variable. Do not change the tokenizer.
'''

#######################################
###!@1 START ANSWER STEP 1

def tokenize_dec(tokenizer):
    """
    Tokenize a list of texts using the GPT-2 tokenizer.
    """
    def tokenize(texts):
        texts_with_eos = [text + tokenizer.eos_token for text in texts["extracted_text"]]
        
        # Tokenize the list of strings
        tokenized = tokenizer(
            texts_with_eos,
            # truncation=True,  # Truncate sequences that are too long
            # max_length=block_size,   # Set max length to block_size for consistency with later steps
            padding=False  
        )
        
        #  Return only the input_ids 
        return {"input_ids": tokenized["input_ids"],
                "labels": tokenized["input_ids"]}  # For language modeling, labels are the same as input_ids
                

    return tokenize

def step_1_tokenization():
    """
    Tokenize text using GPT-2 tokenizer from Hugging Face.
    """

    ## DO NOT MODIFY ##
    tokenizer = AutoTokenizer.from_pretrained(GPT2_TOKENIZER_DIR)
    print(tokenizer)
    ## DO NOT MODIFY ##

    ## start your edits here  =================

    fields_to_remove = ['url', 'date', 'extracted_text']

    datasets.logging.set_verbosity_info()

    # Create output directories if they don't exist
    os.makedirs(tokenized_train_dataset, exist_ok=True)
    os.makedirs(tokenized_test_dataset, exist_ok=True)
    os.makedirs(out_train_dataset, exist_ok=True)
    os.makedirs(out_test_dataset, exist_ok=True)

    start_time = time.time()

    BATCH_SIZE = 100000
    NUM_PROC = min(4, multiprocessing.cpu_count())

    train_files = glob.glob(os.path.join(deduplicated_train_dataset_path, "*.parquet"))
    # test_files = glob.glob(os.path.join(deduplicated_test_dataset_path, "*.parquet"))

    # Load the parquet datasets
    train_dataset = load_dataset("parquet", data_files=train_files)["train"]
    # test_dataset = load_from_disk(deduplicated_test_dataset_path)


    # Define the tokenization function that can be applied to batches of data
    tokenizer_func = tokenize_dec(tokenizer)

    # GPT-2 doesn't have a padding token by default, so we add one
    tokenizer.pad_token = tokenizer.eos_token

    # Run tokenization across the datasets
    # remove_columns ensures we don't keep the heavy raw text in the final saved object
    train_tokenized = train_dataset.map(
        tokenizer_func, 
        batched=True, 
        remove_columns=fields_to_remove,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC, # Speed up using multiple CPU cores
        desc="Running Step 1: Tokenization"
    )

    # Save the tokenized datasets
    train_tokenized.save_to_disk(tokenized_train_dataset)

    print(f"Tokenization completed in {time.time() - start_time:.2f} seconds.")

    ## end your edits here  =================

    return

###!@1 END ANSWER STEP 1


# ──────────────────────────────────────────────────────
# Step 2 - Preprocessing (ALLOWED TO MODIFY)
# ──────────────────────────────────────────────────────
'''
## STEP 2: Preprocessing
----

### Objective
Load tokenized data, group it into sequences of a fixed `block_size`, and save them to disk.

### Guidelines
1. Load the tokenized dataset from `tokenized_train_dataset` and `tokenized_test_dataset`.
2. Concatenate all token sequences (`input_ids`) and group them into chunks of size `block_size`. 
3. Save the train dataset to `final_train_dataset` and the test dataset to `final_test_dataset`.
'''

#######################################
###!@2 START ANSWER STEP 2

def chunk_tokens(examples):
    concatenated = []
    for seq in examples["input_ids"]:
        concatenated.extend(seq)
    
    # Calculate how many complete blocks we can create
    total_length = (len(concatenated) // block_size) * block_size
    
    input_ids = [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
    labels = input_ids.copy()  # For language modeling, labels are the same as input_ids
    result = {
        "input_ids": input_ids,
        "labels": labels
    }
    
    return result

def step_2_preprocessing():
    print(">>> Starting Step 2: Preprocessing...")
    
    ## start your edits here  =================
    datasets.logging.set_verbosity_info()


    os.makedirs(final_train_dataset, exist_ok=True)
    start_time = time.time()

    BATCH_SIZE = 100
    NUM_PROC = min(4, multiprocessing.cpu_count())

    # Load the parquet datasets
    t_train_dataset = load_from_disk(tokenized_train_dataset)

    # Apply the transformation
    print("Chunking train dataset...")
    train_dataset_processed = t_train_dataset.map(
        chunk_tokens,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC, # Use multiprocessing for speed
        remove_columns=t_train_dataset.column_names, # type: ignore 
        desc="Chunking train dataset"
    )

    # Save the final datasets to disk
    train_dataset_processed.save_to_disk(final_train_dataset)

    print(f"Processing completed in {time.time() - start_time:.2f}  seconds.")

    ## end your edits here  =================

    return 

###!@2 END ANSWER STEP 2


# ──────────────────────────────────────────────────────    
# START: Main pipeline (Data Prep) (DO NOT MODIFY)
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(">>> Starting ML Assignment 2 Pipeline (Part 1)...")
    t_start = time.time()

    # Step 1: Tokenization
    if not os.path.exists(tokenized_train_dataset):
        print(f"Tokenizing raw dataset from {deduplicated_train_dataset_path}...")
        step_1_tokenization()
    
    print(f"Loading tokenized dataset from {tokenized_train_dataset}...")
    try:
        tok_dataset_val = load_from_disk(tokenized_train_dataset)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load dataset from {tokenized_train_dataset}. Ensure Step 1 completes correctly using save_to_disk().")
    
    # We do a quick load for schema validation
    validate_step_1(tok_dataset_val.select(range(min(10, len(tok_dataset_val)))).to_pandas())

    # Step 2: Preprocessing (Blocks, Splits, Sanity)
    if not os.path.exists(final_train_dataset):
        step_2_preprocessing()
    
    train_data = load_from_disk(final_train_dataset)
    validate_step_2(train_data)

    print(f"Part 1 Complete! You can now proceed to run the training script.")
    print(f"Total Execution Time: {time.time() - t_start:.2f}s")

# ──────────────────────────────────────────────────────
# END: Main pipeline (Data Prep) (DO NOT MODIFY)
# ──────────────────────────────────────────────────────

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
   This script should be run on a single node since it only performs preprocessing.
   Example Command:

    srun -N 1 \
        --ntasks=1\
        --partition=ds256 \
        --qos=ds256_qos \
        --reservation=<your_reservation_name> \
        -t <time> \
        /apps/run_wrapper.sh <your_script.py>
'''
