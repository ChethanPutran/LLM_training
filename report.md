In your report's Methodology section, you should explicitly state:

"To ensure data parallelism efficiency across the 4 nodes, a DistributedSampler was employed. This partitioned the final_train_dataset into 4 disjoint subsets, ensuring that each node processed unique tokens. The sampler's seed was updated at each epoch to maintain stochasticity across the distributed environment."