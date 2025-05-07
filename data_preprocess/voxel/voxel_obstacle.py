import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

# === FILE PATHS ===
base_dir = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00"
frame_id = "000000"
rgb_path = f"{base_dir}/{frame_id}_rgb.png"
depth_path = f"{base_dir}/{frame_id}_depth.npy"
cam_path = f"{base_dir}/{frame_id}_cam.npz"
output_ply = f"{base_dir}/{frame_id}_translated.ply"
output_map = f"{base_dir}/{frame_id}_obstacle_map.png"

# === LOAD DATA ===
rgb = np.array(Image.open(rgb_path), dtype=np.uint8)
depth = np.load(depth_path)
cam_data = np.load(cam_path)
K = cam_data["intrinsics"]  # (3×3)
T = cam_data["pose"]         # (4×4)

print(f"[INFO] Intrinsics:\n{K}")
print(f"[INFO] RGB shape: {rgb.shape}, Depth shape: {depth.shape}")

# === DEPTH → CAMERA SPACE POINTS ===
H, W = depth.shape
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
z_cam = depth
x_cam = (i - cx) * z_cam / fx
y_cam = (j - cy) * z_cam / fy
pts_cam = np.stack((x_cam, y_cam, z_cam), axis=2).reshape(-1, 3)
mask_valid = z_cam.reshape(-1) > 0
pts_cam = pts_cam[mask_valid]
colors = rgb.reshape(-1, 3)[mask_valid] / 255.0

# === APPLY EXTRINSIC → WORLD SPACE ===
ones = np.ones((pts_cam.shape[0], 1))
pts_homog = np.hstack((pts_cam, ones))
pts_w_h = (T @ pts_homog.T).T
points = pts_w_h[:, :3]

# === CIRCULAR CROP IN WORLD FRAME ===
RADIUS = 15.0
PLANE = "yz"
axis_map = {"xy": (0,1), "xz": (0,2), "yz": (1,2)}
a, b = axis_map[PLANE]
coords = points[:, [a, b]]
mask_c = np.linalg.norm(coords, axis=1) <= RADIUS
points_c = points[mask_c]
colors_c = colors[mask_c]
print(f"[INFO] Cropped to {len(points_c)} points within {RADIUS}m in {PLANE.upper()}‐plane")

# === BUILD POINT CLOUD ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_c)
pcd.colors = o3d.utility.Vector3dVector(colors_c)

# === GROUND PLANE SEGMENTATION (RANSAC) ===
dist_th = 0.02
ransac_n = 3
n_iters = 1000
refine_th = 0.05  # ← YOUR REFINE THRESHOLD

plane_model, inliers = pcd.segment_plane(dist_th, ransac_n, n_iters)
if len(inliers) == 0:
    raise RuntimeError("RANSAC failed to find a ground plane")

ground_raw = pcd.select_by_index(inliers)
nonground_raw = pcd.select_by_index(inliers, invert=True)

# Mean Z of raw ground (world-Z)
ground_z_mean = np.mean(np.asarray(ground_raw.points)[:, 2])
print(f"[INFO] Estimated ground height (world Z): {ground_z_mean:.4f} m")

# Refine ground by Z-threshold
pts_ng = np.asarray(nonground_raw.points)
mask_low = (pts_ng[:, 2] < ground_z_mean + refine_th)
idx_low = np.where(mask_low)[0]
ground_refined = nonground_raw.select_by_index(idx_low)
nonground_final = nonground_raw.select_by_index(idx_low, invert=True)

# Merge ground clouds
ground_pts = np.vstack((np.asarray(ground_raw.points), np.asarray(ground_refined.points)))
ground_cols = np.vstack((np.asarray(ground_raw.colors), np.asarray(ground_refined.colors)))
ground_final = o3d.geometry.PointCloud()
ground_final.points = o3d.utility.Vector3dVector(ground_pts)
ground_final.colors = o3d.utility.Vector3dVector(ground_cols)

# === VISUALIZE GROUND PLANES BEFORE TRANSLATION ===
# Given plane in world frame: ax + by + cz + d = 0
a_plane = -0.000711
b_plane = -0.000078
c_plane = -0.999999
d_plane = 0.003138
adjusted_d = d_plane + c_plane * ground_z_mean

# Generate grid bounds
pts = np.asarray(pcd.points)
x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
x_range = np.linspace(x_min, x_max, 50)
y_range = np.linspace(y_min, y_max, 50)
xx, yy = np.meshgrid(x_range, y_range)

# Compute Z from plane equation
zz = (-adjusted_d - a_plane * xx - b_plane * yy) / c_plane
zz0 = np.zeros_like(xx)  # Z = 0 plane

# Flatten for LineSet
grid_points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
grid0_pts = np.stack((xx, yy, zz0), axis=-1).reshape(-1, 3)

# Create LineSet
lines = []
step = len(y_range)
for i in range(len(x_range)):
    for j in range(len(y_range) - 1):
        lines.append([i * step + j, i * step + j + 1])
for j in range(len(y_range)):
    for i in range(len(x_range) - 1):
        lines.append([i * step + j, (i + 1) * step + j])

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(grid_points),
    lines=o3d.utility.Vector2iVector(lines),
).paint_uniform_color([1, 0, 0])  # Red: Ground plane

line_set0 = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(grid0_pts),
    lines=o3d.utility.Vector2iVector(lines),
).paint_uniform_color([0, 1, 0])  # Green: Z = 0

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=(0, 0, 0))

# Visualize point cloud with ground and non-ground
print("[INFO] Visualizing original world-frame point cloud with ground plane...")
o3d.visualization.draw_geometries(
    [ground_final, nonground_final, frame, line_set, line_set0],
    window_name="Original World-Frame Segmentation",
    width=1280,
    height=720,
    zoom=0.6
)

# === TRANSLATE POINT CLOUD SO GROUND LIES AT Z = 0 ===
translation = np.array([0.0, 0.0, -ground_z_mean])
translated_points = points_c + translation
translated_colors = colors_c

# Rebuild translated point cloud
pcd_translated = o3d.geometry.PointCloud()
pcd_translated.points = o3d.utility.Vector3dVector(translated_points)
pcd_translated.colors = o3d.utility.Vector3dVector(translated_colors)

# Save translated point cloud
o3d.io.write_point_cloud(output_ply, pcd_translated)
print(f"[INFO] Saved translated point cloud to: {output_ply}")

# Visualize translated point cloud
print("[INFO] Visualizing translated point cloud with Z = 0 ground plane...")
o3d.visualization.draw_geometries(
    [pcd_translated, frame, line_set],
    window_name="Translated Point Cloud (Ground at Z = 0)",
    width=1280,
    height=720,
    zoom=0.6
)

# === VOXELIZATION ===
voxel_size = 0.1
pcd_voxeled = pcd_translated.voxel_down_sample(voxel_size)
print("[INFO] Visualizing voxelized point cloud...")
o3d.visualization.draw_geometries(
    [pcd_voxeled, frame, line_set],
    window_name="Voxelized Point Cloud",
    width=1280,
    height=720,
    zoom=0.6
)

# === GENERATE 2D OBSTACLE MAP ===
def build_obstacle_map(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    height_threshold: float
) -> tuple[np.ndarray, float, float, float, float]:
    pts = np.asarray(pcd.points)
    xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    nx = int(np.ceil((x_max - x_min) / voxel_size)) + 1
    ny = int(np.ceil((y_max - y_min) / voxel_size)) + 1

    mask = zs > height_threshold
    ix = np.floor((xs[mask] - x_min) / voxel_size).astype(int)
    iy = np.floor((ys[mask] - y_min) / voxel_size).astype(int)

    occ_map = np.zeros((nx, ny), dtype=np.uint8)
    occ_map[ix, iy] = 1  # mark occupied

    return occ_map, x_min, x_max, y_min, y_max

def save_and_plot_map(
    occ_map: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    height_threshold: float,
    output_path: str
):
    plt.figure(figsize=(6,6))
    plt.imshow(
        occ_map.T,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        interpolation='nearest'
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Obstacle Map @ z > {height_threshold} m")
    plt.colorbar(label="Occupied (1) vs Free (0)")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved 2D obstacle map to: {output_path}")
    plt.show()

# Build and save obstacle map
occ_map, x_min, x_max, y_min, y_max = build_obstacle_map(
    pcd_voxeled, voxel_size, height_threshold=0.05
)

save_and_plot_map(
    occ_map, x_min, x_max, y_min, y_max, 0.05, output_map
)