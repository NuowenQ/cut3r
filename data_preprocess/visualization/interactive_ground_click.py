import cv2
import numpy as np
import csv

# === CONFIGURE THESE PATHS ===
rgb_path     = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00/000000_rgb.png"
depth_path   = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00/000000_depth.npy"
cam_path     = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00/000000_cam.npz"   # must contain "intrinsics" and "cam2world"
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

cam2world = cam_data.get("pose", None)
if cam2world is None or cam2world.shape != (4,4):
    raise ValueError("Need a 4×4 'cam2world' matrix in your .npz")
R_cw = cam2world[:3, :3]

# --- Build the align‐to‐ground rotation ------------------------------------
def make_align_rotation(ground_normal: np.ndarray) -> np.ndarray:
    """
    Returns a 3×3 R so that R @ ground_normal = [0,1,0].
    """
    n = ground_normal / np.linalg.norm(ground_normal)
    up = np.array([0.0, 1.0, 0.0])
    v = np.cross(n, up)
    s = np.linalg.norm(v)
    c = np.dot(n, up)
    if s < 1e-8:
        # Already aligned (c≈1) or opposite (c≈–1)
        return np.eye(3) if c > 0 else np.diag([1, -1, 1])
    vx = np.array([
        [   0, -v[2],  v[1]],
        [ v[2],    0, -v[0]],
        [-v[1], v[0],     0]
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

# ground_normal in camera frame is world‐up [0,1,0] pulled back:
ground_normal_cam = R_cw.T @ np.array([0.0, 1.0, 0.0])
R_align = make_align_rotation(ground_normal_cam)

# --- Storage for results and click coords ----------------------------------
# Now adding Xg and Zg at the end of each row
results   = []  # px,py,depth, Xc,Yc,Zc, Xw,Yw,Zw, Xg,Zg
click_pts = []

# --- Mouse callback ---------------------------------------------------------
def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    d = depth[y, x]
    if d <= 0:
        print(f"[WARN] depth at ({x},{y}) invalid ({d})")
        return

    # 1) Back‐project to camera space
    Xc = (x - cx) * d / fx
    Yc = (y - cy) * d / fy
    Zc = d
    cam_pt = np.array([Xc, Yc, Zc])

    # 2) World‐space (if available)
    homo    = np.hstack([cam_pt, 1.0])
    world_pt = cam2world @ homo
    Xw, Yw, Zw = world_pt[:3]

    # 3) Align to ideal ground‐parallel frame
    gp_pt = R_align @ cam_pt
    Xg, Yg, Zg = gp_pt

    print(
        f"Pixel=({x},{y}) d={d:.3f}m  "
        f"Cam=({Xc:.3f},{Yc:.3f},{Zc:.3f})  "
        f"Wld=({Xw:.3f},{Yw:.3f},{Zw:.3f})  "
        f"Ground=(X={Xg:.3f}, Z={Zg:.3f}, h={Yg:.3f})"
    )

    # Store everything
    results.append([x, y, d, Xc, Yc, Zc, Xw, Yw, Zw, Xg, Zg])
    click_pts.append((x, y))

# --- Open window & set callback --------------------------------------------
win = "RGB (click to project)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win, on_mouse)

# --- Main loop --------------------------------------------------------------
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
    writer.writerow([
        "px","py","depth_m",
        "X_cam","Y_cam","Z_cam",
        "X_wld","Y_wld","Z_wld",
        "X_ground","Z_ground"
    ])
    writer.writerows(results)

print(f"Saved {len(results)} points (incl. ground‐aligned) to {output_csv}")
