import numpy as np
import cv2
import open3d as o3d
import csv

# === CONFIGURE THESE PATHS ===
rgb_path     = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00/000000_rgb.png"
depth_path   = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00/000000_depth.npy"
cam_path     = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00/000000_cam.npz"
output_csv   = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00/000000_ground_points.csv"

# --- Load data --------------------------------------------------------------
bgr = cv2.imread(rgb_path)
if bgr is None:
    raise FileNotFoundError(f"Could not read {rgb_path}")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

depth = np.load(depth_path)
if depth.ndim != 2:
    raise ValueError("Depth map must be 2D")

cam_data   = np.load(cam_path)
K          = cam_data["intrinsics"]
fx, fy     = K[0,0], K[1,1]
cx, cy     = K[0,2], K[1,2]

T = cam_data["pose"]  # 4x4 camera-to-world extrinsic matrix

# --- Storage for results ----------------------------------------------------
results = []  # px, py, depth, Xc, Yc, Zc, Xw, Yw, Zw
click_pts = []

# --- Mouse callback ---------------------------------------------------------
def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    d = depth[y, x]
    if d <= 0:
        print(f"[WARN] depth at ({x},{y}) invalid ({d})")
        return

    # 1. Back-project to camera space
    Xc = (x - cx) * d / fx
    Yc = (y - cy) * d / fy
    Zc = d
    cam_pt = np.array([Xc, Yc, Zc])

    # 2. World-space
    homo = np.hstack([cam_pt, 1.0])
    world_pt = T @ homo
    Xw, Yw, Zw = world_pt[:3]

    print(
        f"Pixel=({x},{y}) d={d:.3f}m  "
        f"Cam=({Xc:.3f},{Yc:.3f},{Zc:.3f})  "
        f"Wld=({Xw:.3f},{Yw:.3f},{Zw:.3f})"
    )

    results.append([x, y, d, Xc, Yc, Zc, Xw, Yw, Zw])
    click_pts.append((x, y))

# --- Open window & set callback ---------------------------------------------
win = "RGB (click to project)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win, on_mouse)

while True:
    display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for px, py in click_pts:
        cv2.circle(display, (px, py), 5, (0, 0, 255), -1)
    cv2.imshow(win, display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# --- Save to CSV -----------------------------------------------------------
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["px", "py", "depth_m", "X_cam", "Y_cam", "Z_cam", "X_wld", "Y_wld", "Z_wld"])
    writer.writerows(results)

print(f"[INFO] Saved {len(results)} clicked points to {output_csv}")

# --- Load camera points for plane estimation -------------------------------
# Use camera coordinates (Xc, Yc, Zc) for RANSAC plane estimation
points_camera = np.array([[row[3], row[4], row[5]] for row in results])
if len(points_camera) < 3:
    raise ValueError("Not enough points to estimate a plane (minimum 3 required).")

# --- Estimate ground plane in camera space ---------------------------------
def estimate_ground_plane(points, distance_threshold=0.04, ransac_n=3, num_iterations=1000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return plane_model, inliers

plane_model, inliers = estimate_ground_plane(points_camera)
a, b, c, d = plane_model
print(f"[RESULT] Camera Plane: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

# --- Transform plane to world space using camera extrinsic ------------------
# Extract rotation and translation from T (4x4 camera-to-world matrix)
R = T[:3, :3]  # Rotation
t = T[:3, 3]   # Translation

# 1. Transform normal vector
normal_camera = np.array([a, b, c])
normal_camera /= np.linalg.norm(normal_camera)
normal_world = R @ normal_camera  # Rotate normal

# 2. Find a point on the camera plane: P_c = -d * normal_camera
P_c = -d * normal_camera

# 3. Transform point to world space
P_w = R @ P_c + t

# 4. Compute world plane distance
d_world = -np.dot(normal_world, P_w)

# Final world plane equation: normal_world · X + d_world = 0
print(f"[RESULT] World Plane: {normal_world[0]:.6f}x + {normal_world[1]:.6f}y + {normal_world[2]:.6f}z + {d_world:.6f} = 0")

# --- Generate large plane grid in world space -------------------------------
def generate_plane_grid(normal, d, center, size=10.0, resolution=50):
    # Choose two orthogonal vectors in the plane
    u = np.array([1.0, 0.0, -normal[0]/normal[2]] if abs(normal[2]) > 1e-6 else [1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, -normal[1]/normal[2]] if abs(normal[2]) > 1e-6 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    u *= size
    v *= size

    u_steps = np.linspace(-0.5, 0.5, resolution)
    v_steps = np.linspace(-0.5, 0.5, resolution)

    grid_points = []
    for u_val in u_steps:
        for v_val in v_steps:
            pt = center + u * u_val + v * v_val
            grid_points.append(pt)

    # Connect grid lines
    lines = []
    for i in range(resolution):
        for j in range(resolution - 1):
            lines.append([i * resolution + j, i * resolution + j + 1])
    for j in range(resolution):
        for i in range(resolution - 1):
            lines.append([i * resolution + j, (i + 1) * resolution + j])

    return np.array(grid_points), lines

# Generate a large plane grid around the center point
grid_points, lines = generate_plane_grid(normal_world, d_world, P_w, size=10.0, resolution=50)

# --- Visualize in Open3D ---------------------------------------------------
# World points
world_points = np.array([[row[6], row[7], row[8]] for row in results])
pcd_world = o3d.geometry.PointCloud()
pcd_world.points = o3d.utility.Vector3dVector(world_points)

# Plane grid
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(grid_points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.paint_uniform_color([1, 0, 0])  # Red

# Coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

# Draw everything
o3d.visualization.draw_geometries(
    [pcd_world, line_set, frame],
    window_name="World Space Ground Plane",
    width=1280,
    height=720,
    zoom=0.6
)