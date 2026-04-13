### 1. Launch TensorBoard on the Remote Server
Run this command in a terminal on the machine where the training is occurring (or a login node that has access to the scratch directory):

```bash
tensorboard --logdir=/scratch/chethan1/SSDS/llm_training/results/tensorboard_logs --port=6006
```

### 2. Set Up SSH Tunneling (If working remotely)
If you are accessing the cluster via SSH from your local laptop, you won't be able to open `localhost:6006` directly. You need to "tunnel" that port to your local machine. 

Open a **new** terminal on your **local laptop** and run:
```bash
ssh -L 6006:localhost:6006 username@your_cluster_address
```
*Replace `username@your_cluster_address` with your actual login details.*

### 3. View in Browser
Once the tunnel is active, open your web browser and go to:
**`http://localhost:6006/`**

---

### What to Look For in Your Report
Since your assignment requires an analysis of **ZeRO Stages**, keep an eye on these specific tags in TensorBoard:

* **`train/loss`:** Ensure the model is actually learning and not diverging (especially in Stage 2/3 where optimization is aggressive).
* **`throughput`:** Compare the "samples per second" across different stages. You will likely notice Stage 0 is fastest but memory-heavy, while Stage 3 is slower due to communication overhead.
* **`memory/allocated`:** This will visualize how ZeRO-3 drastically reduces the memory footprint compared to Stage 0/1.

### 💡 Pro Tip for DeepSpeed
DeepSpeed doesn't always "flush" logs to disk instantly. If you don't see data appearing, you can add this line inside your training loop every 10 steps to force a refresh:

```python
if step % 10 == 0 and deep_speed_engine.monitor is not None:
    for monitor in deep_speed_engine.monitor.monitors:
        if hasattr(monitor, 'flush'):
            monitor.flush()
```
