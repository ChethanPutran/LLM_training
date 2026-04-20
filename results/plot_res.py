import re
import json

def parse_log_file(log_file):
    # ---------------- Patterns ----------------
    stage_pattern = re.compile(
        r"\[Stage (\d+) \| Step (\d+)\] "
        r"Loss=([\d\.]+) \| "
        r"Iter=([\d\.]+)ms \| "
        r"FWD=([\d\.]+) \| "
        r"BWD=([\d\.]+) \| "
        r"STEP=([\d\.]+) \| "
        r"Comm%=([\d\.]+) \| "
        r"TPS=([\d\.]+) \| "
        r"VRAM=([\d\.]+)GB \| Util=(\d+)%"
    )

    micro_pattern = re.compile(
        r"fwd_microstep: ([\d\.]+) \| "
        r"bwd_microstep: ([\d\.]+) \| "
        r"bwd_inner_microstep: ([\d\.]+) \| "
        r"bwd_allreduce_microstep: ([\d\.]+) \| "
        r"step_microstep: ([\d\.]+)"
    )

    # ---------------- Storage ----------------
    metrics = []
    last_micro = None

    with open(log_file, "r") as f:
        for line in f:

            # -------- MICROSTEP LINE --------
            micro_match = micro_pattern.search(line)
            if micro_match:
                fwd_m, bwd_m, inner_m, allreduce_m, step_m = micro_match.groups()

                last_micro = {
                    "fwd_micro_ms": float(fwd_m),
                    "bwd_micro_ms": float(bwd_m),
                    "bwd_inner_ms": float(inner_m),
                    "bwd_allreduce_ms": float(allreduce_m),
                    "step_micro_ms": float(step_m),
                }

            # -------- STAGE LINE --------
            stage_match = stage_pattern.search(line)
            if stage_match:
                stage, step, loss, iter_t, fwd, bwd, step_t, comm, tps, vram, util = stage_match.groups()

                entry = {
                    "stage": int(stage),
                    "step": int(step),
                    "loss": float(loss),
                    "iteration_time_ms": float(iter_t),
                    "fwd_ms": float(fwd),
                    "bwd_ms": float(bwd),
                    "step_ms": float(step_t),
                    "comm_pct_logged": float(comm),
                    "throughput_tps": float(tps),
                    "vram_gb": float(vram),
                    "gpu_util": int(util),
                }

                # Attach microstep info if available
                if last_micro:
                    entry.update(last_micro)

                    # -------- REAL COMPUTE & COMM --------
                    entry["real_compute_ms"] = (
                        last_micro["fwd_micro_ms"] + last_micro["bwd_inner_ms"]
                    )

                    entry["real_comm_ms"] = last_micro["bwd_allreduce_ms"]

                    entry["real_comm_pct"] = (
                        entry["real_comm_ms"] /
                        (entry["real_comm_ms"] + entry["real_compute_ms"] + 1e-8)
                    ) * 100

                metrics.append(entry)

    return metrics


def save_json(metrics, output_file):
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    import sys

    log_file = sys.argv[1]
    output_file = sys.argv[2]

    metrics = parse_log_file(log_file)
    save_json(metrics, output_file)

    print(f"Saved {len(metrics)} entries → {output_file}")