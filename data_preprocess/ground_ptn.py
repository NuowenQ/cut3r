# import os
# import numpy as np
# from PIL import Image
# import open3d as o3d

# def write_ply(filename, points, colors):
#     """Write point cloud to a PLY file."""
#     with open(filename, "w") as f:
#         f.write("ply\nformat ascii 1.0\n")
#         f.write(f"element vertex {len(points)}\n")
#         f.write("property float x\nproperty float y\nproperty float z\n")
#         f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
#         for p, c in zip(points, colors):
#             f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

# # Set your data directory
# data_dir = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00"  # Update this to your scene/camera folder

# # Process all frames
# frame_ids = sorted([
#     f.split("_")[0] for f in os.listdir(data_dir)
#     if f.endswith("_rgb.png")
# ])

# for frame_id in frame_ids:
#     print(f"Processing {frame_id}...")
    
#     # Load RGB, depth, and camera data
#     rgb = np.array(Image.open(os.path.join(data_dir, f"{frame_id}_rgb.png"))).astype(np.uint8)
#     depth = np.load(os.path.join(data_dir, f"{frame_id}_depth.npy"))
#     cam = np.load(os.path.join(data_dir, f"{frame_id}_cam.npz"))
    
#     K = cam["intrinsics"]
#     pose = cam["pose"]  # Use "pose" instead of "cam2worlddata_dir = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00"
#     points_cam = points_cam[valid]
#     colors = rgb.reshape(-1, 3)[valid]
    
#     # Transform to world coordinates
#     R = pose[:3, :3]
#     t = pose[:3, 3]
#     points_world = (R @ points_cam.T).T + t
    
#     # Save as PLY
#     ply_path = os.path.join(data_dir, f"{frame_id}.ply")
#     write_ply(ply_path, points_world, colors)
#     print(f"[✓] Saved: {ply_path}")


import open3d as o3d
import numpy as np

# 1. Load & clean
pcd = o3d.io.read_point_cloud("/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00/000000.ply")
pcd = pcd.voxel_down_sample(0.1)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# 2. RANSAC plane
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.02,
    ransac_n=3,
    num_iterations=1000
)
ground = pcd.select_by_index(inliers)
nonground = pcd.select_by_index(inliers, invert=True)

# compute mean ground height
ground_pts = np.asarray(ground.points)
ground_z_mean = float(ground_pts[:, 2].mean())
print(f"[INFO] shifted by {-ground_z_mean:.3f} m → ground at Z=0")

# 3. SHIFT in numpy, then reassign back
#   ground
ground_pts[:, 2] -= ground_z_mean
ground.points = o3d.utility.Vector3dVector(ground_pts)



#   nonground_final (after your height refine)
pts_ng = np.asarray(nonground.points)
mask_low = pts_ng[:, 2] < (ground_z_mean + 0.05)
nonground_final = nonground.select_by_index(
    np.where(~mask_low)[0]  # keep only pts above that threshold
)
nf_pts = np.asarray(nonground_final.points)
nf_pts[:, 2] -= ground_z_mean
nonground_final.points = o3d.utility.Vector3dVector(nf_pts)

# 4. Split by z<0 and color
pts = np.asarray(nonground_final.points)
neg_idx = np.where(pts[:, 2] < 0)[0]
pos_idx = np.where(pts[:, 2] >= 0)[0]

below = nonground_final.select_by_index(neg_idx)
above = nonground_final.select_by_index(pos_idx)

ground.paint_uniform_color([0.1, 0.9, 0.1])    # green
above.paint_uniform_color([0.9, 0.1, 0.1])     # red
below.paint_uniform_color([0.1, 0.1, 0.9])     # blue

# add coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=[0, 0, 0]
)

# visualize
o3d.visualization.draw_geometries([ground, above, below, frame])