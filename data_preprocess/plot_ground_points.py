import numpy as np
import matplotlib.pyplot as plt

# Load your saved CSV
output_csv = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00/000000_ground_points.csv"
data = np.loadtxt(output_csv, delimiter=',', skiprows=1)

# Ground-aligned X,Z are the last two columns
Xg = data[:, -2]
Zg = data[:, -1]

plt.figure()
plt.scatter(Xg, Zg)
plt.xlabel("Ground X (m)")
plt.ylabel("Ground Z (m)")
plt.title("Ground-aligned Clicked Points")
plt.gca().set_aspect('equal', 'box')
plt.show()
