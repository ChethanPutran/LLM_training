import json
import sys

event_log_file = sys.argv[1]

shuffle_read_time = 0
shuffle_write_time = 0
gc_time = 0
executor_run_time = 0
start_time = None
end_time = None

with open(event_log_file, 'r') as f:
    for line in f:
        event = json.loads(line)

        if event["Event"] == "SparkListenerApplicationStart":
            start_time = event["Timestamp"]

        if event["Event"] == "SparkListenerApplicationEnd":
            end_time = event["Timestamp"]

        if event["Event"] == "SparkListenerTaskEnd":
            metrics = event.get("Task Metrics", {})

            shuffle_read_time += metrics.get("Shuffle Read Metrics", {}).get("Fetch Wait Time", 0)
            shuffle_write_time += metrics.get("Shuffle Write Metrics", {}).get("Write Time", 0)
            gc_time += metrics.get("JVM GC Time", 0)
            executor_run_time += metrics.get("Executor Run Time", 0)

total_time = (end_time - start_time) / 1000 if start_time and end_time else 0

print("========== METRICS ==========")
print(f"Total Application Time (sec): {total_time}")
print(f"Executor Run Time (ms): {executor_run_time}")
print(f"Shuffle Read Time (ms): {shuffle_read_time}")
print(f"Shuffle Write Time (ms): {shuffle_write_time}")
print(f"GC Time (ms): {gc_time}")

communication_time = shuffle_read_time + shuffle_write_time
print(f"Communication Time (ms): {communication_time}")
print(f"Communication Ratio: {communication_time / (executor_run_time+1e-9):.4f}")