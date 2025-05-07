import numpy as np
from PIL import Image
import open3d as o3d
import csv

# === FILE PATHS ===
base_dir = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00"
frame_id = "000000"

csv_path = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00/000000_ground_points.csv"
sphere_list = []
csv_points_list = []  # To store CSV points for plane fitting

# === LOAD CSV POINTS ===
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x = float(row['X_wld'])
        y = float(row['Y_wld'])
        z = float(row['Z_wld'])

        # Store coordinates for plane fitting
        csv_points_list.append([x, y, z])

        # Create a small red sphere for visualization
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate([x, y, z])
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        sphere.compute_vertex_normals()
        sphere_list.append(sphere)

# === LOAD DATA ===
rgb_path = f"{base_dir}/{frame_id}_rgb.png"
depth_path = f"{base_dir}/{frame_id}_depth.npy"
cam_path = f"{base_dir}/{frame_id}_cam.npz"

rgb = np.array(Image.open(rgb_path), dtype=np.uint8)
depth = np.load(depth_path)
cam_data = np.load(cam_path)
K = cam_data["intrinsics"]  # (3×3)
T = cam_data["pose"]         # (4×4) camera → world extrinsic

print(f"[INFO] Intrinsics:\n{K}")
print(f"[INFO] RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
print(f"[INFO] Loaded extrinsic T_cam2world:\n{T}")

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

# === APPLY EXTRINSIC → WORLD SPACE ===
ones = np.ones((pts_cam.shape[0], 1), dtype=np.float64)
pts_homog = np.hstack((pts_cam, ones))  # (N,4)
pts_w_h = (T @ pts_homog.T).T          # (N,4)
points = pts_w_h[:, :3]                # drop w
colors = rgb.reshape(-1, 3)[mask_valid] / 255.0

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
refine_th = 0.05

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
ground_pts = np.vstack((np.asarray(ground_raw.points),
                        np.asarray(ground_refined.points)))
ground_cols = np.vstack((np.asarray(ground_raw.colors),
                         np.asarray(ground_refined.colors)))

ground_final = o3d.geometry.PointCloud()
ground_final.points = o3d.utility.Vector3dVector(ground_pts)
ground_final.colors = o3d.utility.Vector3dVector(ground_cols)

# === FIT PLANE TO CSV POINTS ===
csv_points = np.array(csv_points_list)
csv_pcd = o3d.geometry.PointCloud()
csv_pcd.points = o3d.utility.Vector3dVector(csv_points)

# Fit plane using RANSAC
dist_th_csv = 0.02
ransac_n_csv = 3
n_iters_csv = 1000

plane_model_csv, inliers_csv = csv_pcd.segment_plane(
    distance_threshold=dist_th_csv,
    ransac_n=ransac_n_csv,
    num_iterations=n_iters_csv
)

a_csv, b_csv, c_csv, d_csv = plane_model_csv
print(f"[INFO] Fitted CSV plane equation: {a_csv:.6f}x + {b_csv:.6f}y + {c_csv:.6f}z + {d_csv:.6f} = 0")
z0 = -d_csv / c_csv

print("height required to adjust", z0)
# Generate grid for plane visualization
x_coords = csv_points[:, 0]
y_coords = csv_points[:, 1]
x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()

x_range = np.linspace(x_min, x_max, 50)
y_range = np.linspace(y_min, y_max, 50)
xx, yy = np.meshgrid(x_range, y_range)
zz = (-d_csv - a_csv * xx - b_csv * yy) / c_csv  # Plane equation

xx0, yy0 = np.meshgrid(x_range, y_range)
zz0      = np.zeros_like(xx0)                 # all Z=0

print("zz", zz)

# Flatten for LineSet
grid_points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
grid0_pts = np.stack((xx0, yy0, zz0), axis=-1).reshape(-1, 3)

# Create LineSet for the fitted plane
step = len(y_range)
lines = []
for i in range(len(x_range)):
    for j in range(len(y_range) - 1):
        lines.append([i * step + j, i * step + j + 1])
for j in range(len(y_range)):
    for i in range(len(x_range) - 1):
        lines.append([i * step + j, (i + 1) * step + j])

line_set_csv_plane = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(grid_points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set_csv_plane.paint_uniform_color([0.0, 0.0, 1.0])  # Blue

line_set_z0 = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(grid0_pts),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set_z0.paint_uniform_color([0.0, 1.0, 0.0])  # green

# === VISUALIZATION ===
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=(0, 0, 0))

# add the z=0 grid
visualization_list = [
    ground_final,
    nonground_final,
    frame,
    line_set_csv_plane,
    line_set_z0,
] + sphere_list

print(f"[INFO] Fitted plane to CSV points. Visualizing in blue, z=0 in green.")
o3d.visualization.draw_geometries(
    visualization_list,
    window_name="World-Frame with Fitted Plane and Z=0 Grid",
    width=1280,
    height=720,
    zoom=0.6
)