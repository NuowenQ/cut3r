#---
import os
import numpy as np
from PIL import Image

def write_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def convert_depth_units(depth):
    min_d, max_d, mean_d = depth.min(), depth.max(), depth.mean()
    print(f"[INFO] Depth stats - min: {min_d:.2f}, max: {max_d:.2f}, mean: {mean_d:.2f}")
    return depth
    # Heuristic: if average depth > 10, assume it's in mm
    if mean_d > 0:
        print("[INFO] Converting depth from millimeters to meters.")
        return depth / 1000.0
    else:
        print("[INFO] Depth is already in meters.")
        return depth

# Set your data directory

#processed_mvs
data_dir_mvs = "./processed_mvs_synth/0001"

# processed_unreal_4k
data_dir_unreal = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00"

# frame_ids = sorted([
#     f.split(".")[0] for f in os.listdir(f"{data_dir}/rgb")
#     if f.endswith(".jpg")
# ])

frame_ids = sorted([
    f.split("_")[0] for f in os.listdir(f"{data_dir_unreal}")
    if f.endswith(".png")
])

for frame_id in frame_ids:
    print(f"Processing {frame_id}...")
    
    # rgb_path = f"{data_dir}/rgb/{frame_id}.jpg"
    rgb_path = f"{data_dir_unreal}/{frame_id}_rgb.png"
    # depth_path = f"{data_dir}/depth/{frame_id}.npy"
    depth_path = f"{data_dir_unreal}/{frame_id}_depth.npy"

    rgb = np.array(Image.open(rgb_path)).astype(np.uint8)
    depth = np.load(depth_path)
    depth = convert_depth_units(depth)

    #cam_path = f"{data_dir}/cam/{frame_id}.npz"
    cam_path = f"{data_dir_unreal}/{frame_id}_cam.npz"


    cam = np.load(cam_path)
    K = cam["intrinsics"]
    pose = cam["pose"]

    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    points_cam = np.stack((x, y, z), axis=2).reshape(-1, 3)

    valid = z.reshape(-1) > 0
    points_cam = points_cam[valid]
    colors = rgb.reshape(-1, 3)[valid]



    # === CROP TO 15M CIRCULAR REGION IN YZ ===
    xy = points_cam[:, [1,2]]
    radii = np.linalg.norm(xy, axis=1)
    mask_circle = radii <= 15

    points_cropped = points_cam[mask_circle]
    colors_cropped = colors[mask_circle]

    # Transform to world coordinates
    R = pose[:3, :3]
    t = pose[:3, 3]
    points_world = (R @ points_cropped.T).T + t


    # Save as PLY
    ply_path = f"{data_dir_unreal}/{frame_id}.ply"
    write_ply(ply_path, points_world, colors_cropped)
    print(f"[✓] Saved: {ply_path}\n")
