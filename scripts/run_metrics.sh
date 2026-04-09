#!/bin/bash

##########################################
# FUNCTION: Extract Metrics from Event Log
##########################################

extract_metrics() {
    local APP_ID=$1
    local EVENT_LOG=$(hdfs dfs -ls /user/chethan1/spark-logs 2>/dev/null | grep "$APP_ID" | awk '{print $8}' | tail -n 1)

    if [ -z "$EVENT_LOG" ]; then
        echo "0,0,0,0,0,0,0,0,0,0,0,0,0,0"
        return
    fi

    hdfs dfs -cat "$EVENT_LOG" > temp_event_log.json 2>/dev/null

    # Get CSV-ready metrics
    python3 scripts/extract_metrics.py temp_event_log.json
}


##########################################
# CORE LOGIC: EXECUTION & EXTRACTION
##########################################
run_experiment() {
    local TYPE=$1
    local num_exec=$2
    local num_cores=$3
    local mem=$4
    local data_path=$5
    local run_idx=$6
    local total_cores=$((num_exec * num_cores))
    local mem_overhead=$executor_memory_overhead

    local LOG_FILE="outputs/${TYPE}_e${num_exec}_m${mem}_run${run_idx}.log"

    echo "--------------------------------------------------------"
    echo "STARTING: $TYPE | Execs: $num_exec | Run: $run_idx/$iterations"
    echo "--------------------------------------------------------"

    # The script waits here until spark-submit exits

    spark-submit \
        --master yarn \
        --deploy-mode client \
        --num-executors "$num_exec" \
        --executor-cores "$num_cores" \
        --executor-memory "${mem}g" \
        --conf spark.pyspark.python=/opt/ds256_env/bin/python \
        --conf spark.pyspark.driver.python=/opt/ds256_env/bin/python \
        --conf spark.executor.memoryOverhead="${mem_overhead}g" \
        --conf spark.yarn.queue=root.ds256.team8 \
        --conf spark.app.name="${TYPE}_e${num_exec}_m${mem}_run${run_idx}"\
        \
        --conf spark.eventLog.enabled=true \
        --conf spark.eventLog.dir=hdfs:///user/chethan1/spark-logs \
        \
        --conf spark.executor.processTreeMetrics.enabled=true \
        --conf spark.executor.metrics.pollingInterval=5s \
        \
        --conf spark.eventLog.logStageExecutorMetrics=true \
        --conf spark.eventLog.logBlockUpdates.enabled=true \
        --files hdfs:///ds256_2026/lid.176.bin \
        "$MAIN_FILE" "$data_path" 2>&1 | tee "$LOG_FILE"

    # Extract ID and Timings
    local APP_ID=$(grep -o "application_[0-9]\{13\}_[0-9]\+" "$LOG_FILE" | tail -n 1)
    local TOTAL_TIME=$(grep -i "Total execution time" "$LOG_FILE" | awk -F': ' '{print $2}' | awk '{print $1}')
    TOTAL_TIME=${TOTAL_TIME:-0}

    # Step-wise extraction
    local S1=$(grep "Step 1 completed" "$LOG_FILE" | awk '{print $(NF-1)}'); S1=${S1:-0}
    local S2=$(grep "Step 2 completed" "$LOG_FILE" | awk '{print $(NF-1)}'); S2=${S2:-0}
    local S3=$(grep "Step 3 completed" "$LOG_FILE" | awk '{print $(NF-1)}'); S3=${S3:-0}
    local S4=$(grep "Step 4 completed" "$LOG_FILE" | awk '{print $(NF-1)}'); S4=${S4:-0}
    local S5A=$(grep "Step 5a completed" "$LOG_FILE" | awk '{print $(NF-1)}'); S5A=${S5A:-0}
    local S5B=$(grep "Step 5b completed" "$LOG_FILE" | awk '{print $(NF-1)}'); S5B=${S5B:-0}
    local S6A=$(grep "Step 6a completed" "$LOG_FILE" | awk '{print $(NF-1)}'); S6A=${S6A:-0}
    local S6B=$(grep "Step 6b completed" "$LOG_FILE" | awk '{print $(NF-1)}'); S6B=${S6B:-0}

    # Deep Metrics
    # Deep Metrics
    METRICS=$(extract_metrics "$APP_ID")

    IFS=',' read -r APP_TIME_MS EXEC_RUN_MS PARALLELISM GC GC_RATIO SHUFFLE_READ_MS SHUFFLE_WRITE_MS COMM_TIME COMM_RATIO COMPUTE_TIME COMPUTE_RATIO MEM_SPILL DISK_SPILL PEAK_HEAP <<< "$METRICS"

    echo "$TYPE,$num_exec,$num_cores,$total_cores,$mem,$data_path,$APP_ID,$S1,$S2,$S3,$S4,$S5A,$S5B,$S6A,$S6B,$TOTAL_TIME,$APP_TIME_MS,$EXEC_RUN_MS,$PARALLELISM,$GC,$GC_RATIO,$SHUFFLE_READ_MS,$SHUFFLE_WRITE_MS,$COMM_TIME,$COMM_RATIO,$COMPUTE_TIME,$COMPUTE_RATIO,$MEM_SPILL,$DISK_SPILL,$PEAK_HEAP,$run_idx" >> "$RESULTS_FILE"
    echo "COMPLETED: $APP_ID | Time: $TOTAL_TIME sec"
    echo "Waiting 10s for cluster cleanup..."
    sleep 10  # This ensures YARN containers are fully released before next request
}

# ==============================
# GLOBAL CONFIG
# ==============================

# iterations=1
iterations=3
executor_cores=4
executor_memory_overhead=2   # default unless overridden

RESULTS_FILE="outputs/scaling_results_$(date +%Y%m%d_%H%M).csv"


MAIN_FILE="/scratch/chethan1/SSDS/scalability_study/a1_v1.0.py"
echo "Type,Execs,Cores,TotalCores,MemGB,Data,AppID,S1,S2,S3,S4,S5a,S5b,S6a,S6b,TotalTime,AppTimeMs,ExecRunTimeMs,Parallelism,GCMs,GCRatio,ShuffleReadMs,ShuffleWriteMs,CommMs,CommRatio,ComputeMs,ComputeRatio,MemSpillMB,DiskSpillMB,PeakHeapMB,Run" > "$RESULTS_FILE"


# Clean temporary files
rm -f temp_event_log.json metrics_out.txt


# Size Replication size Path 
# 3.9 G 11.7 G /ds256_2026/small_1 
# 7.8 G 23.4 G /ds256_2026/small_2 
# 15.7 G 47.0 G /ds256_2026/small_4 
# 23.4 G 70.1 G /ds256_2026/small_6 
# 31.0 G 92.9 G /ds256_2026/small_8 
# 62.2 G 186.5 G /ds256_2026/small_16 

# # Parquest file sizes 
data_sizes=(
"small_1" # 3.9 G 
"small_2" # 7.8 G 
"small_4" # 15.7 G 
"small_6" # 23.4 G 
"small_8" # 31.0 G 
"small_16" # 62.2 G 
)

# # ==========================================================
# # STRONG SCALING (Fixed Data = small_2, 7.8 G )
# # ==========================================================

# ### Configuration
# strong_dataset="small_2" # 7.8 G
# # strong_dataset="small_1" 
# executors=(2 4 8 16 32) 
# executor_memory=8  # GB

# ### Execution Loop
# for n in "${executors[@]}"; do
#     for run in $(seq 1 $iterations); do
#         run_experiment \
#             "Strong" \
#             "$n" \
#             "$executor_cores" \
#             "$executor_memory" \
#             "$strong_dataset" \
#             "$run"
#     done
# done

# ===========================================================
# WEAK SCALING (Data ∝ Executors)
# ==========================================================

# ### Mapping
# # weak_executors=(16)
# weak_executors=(8 6 4 2 1 16)
# weak_datasets=(small_8 small_6 small_4 small_2 small_1 small_16)
# # weak_datasets=(small_8 small_16)
# # weak_datasets=(small_16)
# executor_memory=8   # GB (fixed to isolate scaling effects)
# executor_cores=2   # 2 cores/executor to better utilize cluster resources and observe scaling trends
# ### Execution Loop
# for idx in "${!weak_executors[@]}"; do
#     n=${weak_executors[$idx]}
#     dataset=${weak_datasets[$idx]}

#     for run in $(seq 1 $iterations); do
#         run_experiment \
#             "Weak" \
#             "$n" \
#             "$executor_cores" \
#             "$executor_memory" \
#             "$dataset" \
#             "$run"
#     done
# done

# ==========================================================
# MEMORY SENSITIVITY STUDY
# ==========================================================

### Configuration
memory_executors=16
memory_dataset="small_8"
executor_cores=2
# memory_values=(2 4 6 8 10)
memory_values=(2 8 10)


### Execution Loop
for mem in "${memory_values[@]}"; do
    if [ "$mem" -le 6 ]; then
        executor_memory_overhead=1
    else
        executor_memory_overhead=2
    fi

    for run in $(seq 1 $iterations); do
        run_experiment \
            "MemoryStudy" \
            "$memory_executors" \
            "$executor_cores" \
            "$mem" \
            "$memory_dataset" \
            "$run"
    done
done

# ==========================================================
# FINAL CLEANUP
# ==========================================================
rm -f temp_event_log.json metrics_out.txt
echo "All experiments complete."
echo "Results stored in: $RESULTS_FILE"


# # Scientific Validation

# ### Strong Scaling

# * Data fixed (31GB)
# * Cores increase
# * Expect runtime ↓ then flatten

# ### Weak Scaling

# * Data per core constant
# * Expect runtime ≈ constant

# ### Memory Study

# * Spill decreases as memory ↑
# * GC decreases
# * Diminishing returns after 8GB
