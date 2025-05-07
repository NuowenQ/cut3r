import numpy as np
import os
from PIL import Image
import open3d as o3d
import csv
import matplotlib.pyplot as plt

# === FILE PATHS ===
base_dir = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00"
csv_path = os.path.join(base_dir, "000000_ground_points.csv")

# === PARAMETERS ===
downsample_factor = 5  # No downsampling
RADIUS = 7             # Circular crop radius (m)
PLANE = "yz"           # Plane to crop
axis_map = {"xy": (0,1), "xz": (0,2), "yz": (1,2)}
a, b = axis_map[PLANE]

# === OUTPUT DATA ===
global_points = []
global_colors = []
sphere_list = []
csv_points_list = []

# === GET ALL FRAME IDS ===
rgb_files = [f for f in os.listdir(base_dir) if f.endswith("_rgb.png")]
frame_ids = sorted([f.split("_")[0] for f in rgb_files])
print(f"[INFO] Found {len(frame_ids)} frames.")

# === LOAD CSV POINTS (for plane fitting in first frame) ===
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x = float(row['X_wld'])
        y = float(row['Y_wld'])
        z = float(row['Z_wld'])
        csv_points_list.append([x, y, z])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate([x, y, z])
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.compute_vertex_normals()
        sphere_list.append(sphere)

# === PROCESS ALL FRAMES ===
for i, frame_id in enumerate(frame_ids):
    print(f"[INFO] Processing frame {i+1}/{len(frame_ids)}: {frame_id}")
    rgb_path   = os.path.join(base_dir, f"{frame_id}_rgb.png")
    depth_path = os.path.join(base_dir, f"{frame_id}_depth.npy")
    cam_path   = os.path.join(base_dir, f"{frame_id}_cam.npz")
    rgb      = np.array(Image.open(rgb_path), dtype=np.uint8)
    depth    = np.load(depth_path)
    cam_data = np.load(cam_path)
    K        = cam_data["intrinsics"]
    T        = cam_data["pose"]

    # === DEPTH → CAMERA SPACE POINTS ===
    H, W = depth.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    i_grid, j_grid = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    z_cam = depth
    x_cam = (i_grid - cx) * z_cam / fx
    y_cam = (j_grid - cy) * z_cam / fy
    pts_cam = np.stack((x_cam, y_cam, z_cam), axis=2).reshape(-1, 3)
    mask_valid = z_cam.reshape(-1) > 0
    pts_cam = pts_cam[mask_valid]

    # === APPLY EXTRINSIC → WORLD SPACE ===
    ones = np.ones((pts_cam.shape[0], 1))
    pts_homog = np.hstack((pts_cam, ones))
    pts_world = (T @ pts_homog.T).T[:, :3]

    # === CROP IN WORLD FRAME ===
    coords = pts_world[:, [a, b]]
    mask_c = np.linalg.norm(coords, axis=1) <= RADIUS
    pts_cropped = pts_world[mask_c]
    colors_cropped = rgb.reshape(-1, 3)[mask_valid][mask_c] / 255.0

    # === DOWNSAMPLE TO AVOID MEMORY ISSUES ===
    n_points = pts_cropped.shape[0]
    if n_points == 0:
        continue
    indices = np.random.choice(n_points, size=n_points // downsample_factor, replace=False)
    global_points.append(pts_cropped[indices])
    global_colors.append(colors_cropped[indices])

# === COMBINE ALL POINTS ===
combined_points = np.vstack(global_points)
combined_colors = np.vstack(global_colors)

# Create combined point cloud
combined_pcd = o3d.geometry.PointCloud()
combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.05)
print(f"[INFO] Final combined point cloud has {len(combined_pcd.points)} points.")

# === FIT PLANE TO CSV POINTS (FIRST FRAME ONLY) ===
csv_points = np.array(csv_points_list)
csv_pcd = o3d.geometry.PointCloud()
csv_pcd.points = o3d.utility.Vector3dVector(csv_points)
plane_model, inliers = csv_pcd.segment_plane(
    distance_threshold=0.02, ransac_n=3, num_iterations=1000
)
a_csv, b_csv, c_csv, d_csv = plane_model
print(f"[INFO] Fitted CSV plane: {a_csv:.6f}x + {b_csv:.6f}y + {c_csv:.6f}z + {d_csv:.6f} = 0")

# === SHIFT POINT CLOUD SO GROUND PLANE IS AT Z = 0 ===
z0 = -d_csv / c_csv
print(f"[INFO] Shifting all points down by {z0:.4f} meters to align ground plane at z=0.")
combined_points = np.asarray(combined_pcd.points)
combined_points[:, 2] -= z0
combined_pcd.points = o3d.utility.Vector3dVector(combined_points)

# === GENERATE HEIGHT-WISE OCCUPIED AREA RATIO PLOT ===
def compute_ratios(points, edges, grid=0.05):
    ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        layer = points[(points[:, 2] >= lo) & (points[:, 2] < hi)]
        if layer.size == 0:
            ratios.append(0.0)
            continue
        xy = layer[:, [0, 1]]
        xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
        ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
        side = max(xmax - xmin, ymax - ymin)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        X0, X1 = cx - side / 2, cx + side / 2
        Y0, Y1 = cy - side / 2, cy + side / 2
        xi = np.floor((xy[:, 0] - X0) / grid).astype(int)
        yi = np.floor((xy[:, 1] - Y0) / grid).astype(int)
        occ = len(set(zip(xi, yi)))
        tot = int(np.ceil(side / grid)) ** 2
        ratios.append(occ / tot if tot > 0 else 0.0)
    return np.array(ratios)

def plot_ratios(centers, ratios, ground_peak_h, save_dir, filename="height_distribution.png"):
    width = centers[1] - centers[0]
    plt.figure(figsize=(10, 5))
    plt.bar(centers, ratios, width=width, edgecolor='black')
    plt.axvline(ground_peak_h, color='red', linestyle='--',
                label=f'Ground plane @ {ground_peak_h:.2f}')
    plt.xlabel('Height (Z)')
    plt.ylabel('Occupied Area Ratio')
    plt.title('Height-wise Occupied Area Ratio (Z-up)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    print(f"[✓] Saved plot: {save_path}")
    plt.close()

# Extract shifted point cloud for plotting
all_pts = np.asarray(combined_pcd.points)
z_values = all_pts[:, 2]

# Generate histogram bins
edges = np.histogram_bin_edges(z_values, bins=100)
centers = 0.5 * (edges[:-1] + edges[1:])

# Compute occupied area ratios
ratios = compute_ratios(all_pts, edges, grid=0.05)

# Plot and save
plot_ratios(centers, ratios, ground_peak_h=z0, save_dir=base_dir)

# === USER INPUT FOR VISIBILITY MAP ===
print("\n[INFO] Generating visibility map based on user-defined bounds.")
lower_bound_vis = float(input("Enter lower bound (Z in meters, e.g., 0.0): "))
upper_bound_vis = float(input("Enter upper bound (Z in meters, e.g., 2.0): "))

# === BUILD VISIBILITY MAP ===
def build_visibility_map_from_voxel_range(vg, lower_bound, upper_bound):
    voxel_size = vg.voxel_size
    origin = vg.origin
    voxels = vg.get_voxels()
    if not voxels:
        raise RuntimeError("VoxelGrid has no voxels!")

    grid_indices = np.array([v.grid_index for v in voxels], dtype=int)
    ixs, iys, izs = grid_indices.T
    nx, ny = ixs.max() + 1, iys.max() + 1
    vis_map = np.zeros((nx, ny), dtype=bool)
    valid_cells = []

    for voxel in voxels:
        i, j, k = voxel.grid_index
        center_z = origin[2] + (k + 0.5) * voxel_size  # center of voxel in world Z
        if lower_bound <= center_z <= upper_bound:
            valid_cells.append((i, j))

    if valid_cells:
        unique_xy = np.unique(valid_cells, axis=0)
        vis_map[unique_xy[:, 0], unique_xy[:, 1]] = True

    x_min = origin[0]
    y_min = origin[1]
    x_max = x_min + nx * voxel_size
    y_max = y_min + ny * voxel_size
    return vis_map, x_min, x_max, y_min, y_max

def save_and_plot_visibility_map(map_arr, x_min, x_max, y_min, y_max, title, output_path):
    plt.figure(figsize=(6,6))
    plt.imshow(
        map_arr.T,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        interpolation='nearest',
        cmap='gray'
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.colorbar(label="1 = visible / 0 = not visible")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved visibility map to: {output_path}")
    plt.show()

voxel_size_vis = 0.1
vg_vis = o3d.geometry.VoxelGrid.create_from_point_cloud(combined_pcd, voxel_size_vis)
vis_map, x_min_vis, x_max_vis, y_min_vis, y_max_vis = build_visibility_map_from_voxel_range(
    vg_vis, lower_bound_vis, upper_bound_vis
)

output_vis_map = os.path.join(base_dir, f"visibility_map_{lower_bound_vis}_{upper_bound_vis}.png")
save_and_plot_visibility_map(
    vis_map.astype(np.uint8),
    x_min_vis, x_max_vis,
    y_min_vis, y_max_vis,
    title=f"Visibility Map (z ∈ [{lower_bound_vis}, {upper_bound_vis}] m)",
    output_path=output_vis_map
)

# === USER INPUT FOR OBSTACLE MAP ===
print("\n[INFO] Generating obstacle map based on user-defined bounds.")
lower_bound = float(input("Enter lower bound (Z in meters, e.g., 0.0): "))
upper_bound = float(input("Enter upper bound (Z in meters, e.g., 2.0): "))

# === BUILD OBSTACLE MAP ===
def build_obstacle_map(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    lower_bound: float,
    upper_bound: float
) -> tuple[np.ndarray, float, float, float, float]:
    pts = np.asarray(pcd.points)
    xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    nx = int(np.ceil((x_max - x_min) / voxel_size)) + 1
    ny = int(np.ceil((y_max - y_min) / voxel_size)) + 1
    mask = (zs >= lower_bound) & (zs <= upper_bound)
    ix = np.floor((xs[mask] - x_min) / voxel_size).astype(int)
    iy = np.floor((ys[mask] - y_min) / voxel_size).astype(int)
    occ_map = np.zeros((nx, ny), dtype=np.uint8)
    occ_map[ix, iy] = 1  # mark occupied
    return occ_map, x_min, x_max, y_min, y_max

def save_and_plot_map(
    occ_map: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    lower_bound: float,
    upper_bound: float,
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
    plt.title(f"Obstacle Map (z ∈ [{lower_bound}, {upper_bound}] m)")
    plt.colorbar(label="Occupied (1) vs Free (0)")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved 2D obstacle map to: {output_path}")
    plt.show()

# === GENERATE AND SAVE OBSTACLE MAP ===
voxel_size = 0.1
pcd_voxeled = combined_pcd.voxel_down_sample(voxel_size)
occ_map, x_min, x_max, y_min, y_max = build_obstacle_map(
    pcd_voxeled, voxel_size, lower_bound, upper_bound
)
output_map = os.path.join(base_dir, f"obstacle_map_{lower_bound}_{upper_bound}.png")
save_and_plot_map(occ_map, x_min, x_max, y_min, y_max, lower_bound, upper_bound, output_map)

# === VISUALIZATION OF H PLOT DATA COLLECTION ===
selected_indices = [0, 5, 20]
grid_size = 0.05
vis_geometries = []
for idx in selected_indices:
    lo = edges[idx]
    hi = edges[idx + 1]
    z_center = centers[idx]
    layer_mask = (all_pts[:, 2] >= lo) & (all_pts[:, 2] < hi)
    layer_points = all_pts[layer_mask]
    if len(layer_points) == 0:
        continue
    xy = layer_points[:, [0, 1]]
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    side = max(xmax - xmin, ymax - ymin)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    X0, X1 = cx - side / 2, cx + side / 2
    Y0, Y1 = cy - side / 2, cy + side / 2
    x_steps = np.arange(X0, X1 + grid_size, grid_size)
    y_steps = np.arange(Y0, Y1 + grid_size, grid_size)
    lines = []
    for x in x_steps:
        lines.append([[x, Y0, z_center], [x, Y1, z_center]])
    for y in y_steps:
        lines.append([[X0, y, z_center], [X1, y, z_center]])
    line_points = []
    line_indices = []
    for i, line in enumerate(lines):
        line_points.extend(line)
        line_indices.append([2*i, 2*i + 1])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(line_points)),
        lines=o3d.utility.Vector2iVector(line_indices)
    ).paint_uniform_color([0.7, 0.7, 0.7])
    vis_geometries.append(line_set)
    xi = np.floor((xy[:, 0] - X0) / grid_size).astype(int)
    yi = np.floor((xy[:, 1] - Y0) / grid_size).astype(int)
    occupied_cells = set(zip(xi, yi))
    occupied_points = []
    for (xi_cell, yi_cell) in occupied_cells:
        x_center = X0 + xi_cell * grid_size + grid_size / 2
        y_center = Y0 + yi_cell * grid_size + grid_size / 2
        occupied_points.append([x_center, y_center, z_center])
    if occupied_points:
        occupied_pcd = o3d.geometry.PointCloud()
        occupied_pcd.points = o3d.utility.Vector3dVector(occupied_points)
        occupied_pcd.paint_uniform_color([0.0, 1.0, 0.0])
        vis_geometries.append(occupied_pcd)

# === FINAL VISUALIZATION ===
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
visualization_list = [combined_pcd, frame] + sphere_list + vis_geometries
o3d.visualization.draw_geometries(
    visualization_list,
    window_name="H Plot Visualization: Grids & Occupied Cells",
    width=1280,
    height=720,
    zoom=0.6
)