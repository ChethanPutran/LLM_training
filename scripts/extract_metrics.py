import json
import sys
from collections import defaultdict

event_log_file = sys.argv[1]

start_time = None
end_time = None

executor_run_time = 0
gc_time_ms = 0

shuffle_read_ms = 0
shuffle_write_ns = 0

memory_spill_bytes = 0
disk_spill_bytes = 0

# Track peak heap memory per executor
executor_peak_heap = defaultdict(int)

with open(event_log_file, 'r') as f:
    for line in f:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("Event", "")

        # ------------------------------
        # Application start/end
        # ------------------------------
        if event_type == "SparkListenerApplicationStart":
            start_time = event.get("Timestamp")

        elif event_type == "SparkListenerApplicationEnd":
            end_time = event.get("Timestamp")

        # ------------------------------
        # Task metrics
        # ------------------------------
        elif event_type == "SparkListenerTaskEnd":
            metrics = event.get("Task Metrics", {})

            executor_run_time += metrics.get("Executor Run Time", 0)
            gc_time_ms += metrics.get("JVM GC Time", 0)

            # Spill
            memory_spill_bytes += metrics.get("Memory Bytes Spilled", 0)
            disk_spill_bytes += metrics.get("Disk Bytes Spilled", 0)

            # Shuffle Read (ONLY Fetch Wait Time to avoid double counting)
            sr_metrics = metrics.get("Shuffle Read Metrics", {})
            shuffle_read_ms += sr_metrics.get("Fetch Wait Time", 0)

            # Shuffle Write
            sw_metrics = metrics.get("Shuffle Write Metrics", {})
            shuffle_write_ns += sw_metrics.get("Shuffle Write Time", 0)

        # ------------------------------
        # Executor memory metrics
        # ------------------------------
        elif event_type == "SparkListenerExecutorMetricsUpdate":
            exec_id = event.get("Executor ID")
            updates = event.get("Executor Metrics Updated", [])

            for update in updates:
                metrics_map = update.get("Executor Metrics", {})
                heap_mem = metrics_map.get("JVMHeapMemory", 0)

                if heap_mem > executor_peak_heap[exec_id]:
                    executor_peak_heap[exec_id] = heap_mem


# ==========================================================
# Final Calculations
# ==========================================================

if start_time is not None and end_time is not None:
    total_time = end_time - start_time
else:
    total_time = 0

shuffle_write_ms = shuffle_write_ns / 1_000_000
communication_ms = shuffle_read_ms + shuffle_write_ms

gc_ratio = gc_time_ms / (executor_run_time + 1e-9)
communication_ratio = communication_ms / (executor_run_time + 1e-9)

memory_spill_mb = memory_spill_bytes / (1024 * 1024)
disk_spill_mb = disk_spill_bytes / (1024 * 1024)

# Global peak heap across executors
global_peak_heap_mb = (
    max(executor_peak_heap.values()) / (1024 * 1024)
    if executor_peak_heap else 0
)

# Effective parallelism
parallelism_factor = executor_run_time / (total_time + 1e-9)

# Compute-only time
compute_ms = executor_run_time - communication_ms - gc_time_ms
compute_ratio = compute_ms / (executor_run_time + 1e-9)


# ==========================================================
# Output
# ==========================================================
metrics = {
    "total_time_ms": total_time,
    "executor_run_time_ms": executor_run_time,
    "parallelism_factor": parallelism_factor,
    "gc_time_ms": gc_time_ms,
    "gc_ratio": gc_ratio,
    "shuffle_read_ms": shuffle_read_ms,
    "shuffle_write_ms": shuffle_write_ms,
    "communication_ms": communication_ms,
    "communication_ratio": communication_ratio,
    "compute_ms": compute_ms,
    "compute_ratio": compute_ratio,
    "memory_spill_mb": memory_spill_mb,
    "disk_spill_mb": disk_spill_mb,
    "peak_heap_mb": global_peak_heap_mb
}
print(",".join(str(metrics[k]) for k in [
    "total_time_ms","executor_run_time_ms","parallelism_factor","gc_time_ms","gc_ratio",
    "shuffle_read_ms","shuffle_write_ms","communication_ms","communication_ratio",
    "compute_ms","compute_ratio","memory_spill_mb","disk_spill_mb","peak_heap_mb"
]))