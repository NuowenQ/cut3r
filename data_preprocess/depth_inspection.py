import numpy as np

depth = np.load("/home/clearlab/data_plot/data_output/00008/0/00000_depth.npy")  # Replace with your actual path
print(f"Shape: {depth.shape}")
print(f"Data type: {depth.dtype}")
print(f"Min depth: {depth.min()}")
print(f"Max depth: {depth.max()}")
print(f"Mean depth: {depth.mean()}")
