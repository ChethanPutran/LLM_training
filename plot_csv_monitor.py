import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_metrics.csv")
# Example: Plot Throughput vs Step
plt.plot(df['step'], df['samples_per_second'])
plt.xlabel('Step')
plt.ylabel('Throughput (Samples/sec)')
plt.title('Training Efficiency')
plt.show()

    while step < total_steps:
        epoch += 1
        # Important: Re-set the seed for the sampler in distributed training 
        # to ensure each node gets different data per epoch
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        for batch in train_dataloader:
            if step >= total_steps: break
            
            # --- MEASUREMENT WINDOW ---
            # We only profile a specific window to avoid slowing down the whole run
            is_profile_step = (step == 10) 
            
            if is_profile_step: prof.start_profile()
            
            # --- FORWARD PASS ---
            # DeepSpeed handles the .to(device) and distributed data partitioning
            inputs = batch.to(model_engine.local_rank)
            outputs = model_engine(inputs)
            loss = outputs.loss

            # --- BACKWARD PASS ---
            # DeepSpeed manages the ZeRO gradient sharding/reduction here
            model_engine.backward(loss)

            # --- OPTIMIZER STEP ---
            # Handles gradient clipping, weight updates, and LR scheduling
            model_engine.step()

            if is_profile_step:
                prof.stop_profile()
                if model_engine.local_rank == 0:
                    prof.print_model_profile(profile_step=step)
                prof.end_profile()

            
