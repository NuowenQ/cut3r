import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d

def load_frames_world(data_dir: str, num_frames: int = 100):
    """
    Read num_frames PLYs from data_dir, look for matching *_cam.npz extrinsics,
    and return a list of point clouds in the WORLD frame.
    """
    frames = []
    for i in range(num_frames):
        frame_id = f"{i:06d}"
        ply_fp   = os.path.join(data_dir, f"{frame_id}.ply")
        cam_fp   = os.path.join(data_dir, f"{frame_id}_cam.npz")

        if not os.path.exists(ply_fp):
            print(f"[WARN] missing file: {ply_fp}")
            continue
        if not os.path.exists(cam_fp):
            print(f"[WARN] missing file: {cam_fp}")
            continue

        # 1) load raw PLY points (camera/local frame)
        pts_local = []
        with open(ply_fp, "r") as f:
            in_header = True
            for line in f:
                if in_header:
                    if line.strip() == "end_header":
                        in_header = False
                    continue
                x, y, z = map(float, line.split()[:3])
                pts_local.append([x, y, z])
        pts_local = np.asarray(pts_local, dtype=np.float32)
        if pts_local.size == 0:
            continue

        # 2) load camera->world extrinsic
        cam_data = np.load(cam_fp)
        T        = cam_data["pose"]        # shape (4,4)
        R, t     = T[:3, :3], T[:3, 3]     # R: (3,3), t: (3,)

        # 3) transform into world frame
        pts_world = (R @ pts_local.T).T + t  # (N,3)

        frames.append(pts_world)

    if not frames:
        raise RuntimeError("No *.ply files found (in world frame) – check data_dir!")
    return frames

# ───────────────────────────────────────────────────────────────────────────────


# ======  every function below now uses Z (col 2) instead of Y (col 1) =========
def detect_negative_z(frames, tol: float = 0.0) -> bool:
    """Return True if ANY frame contains at least one point with z < tol."""
    has_neg = any((f[:, 2] < tol).any() for f in frames)
    if has_neg:
        zmin = min(f[:, 2].min() for f in frames)
        print(f"[INFO] Negative Z detected (lowest z = {zmin:.3f}).")
    else:
        print("[INFO] No negative Z values detected.")
    return has_neg


def shift_frames_to_ground(frames):
    """Make every frame’s minimum z exactly zero (keeps relative heights)."""
    for pts in frames:
        pts[:, 2] -= pts[:, 2].min()


def filter_by_height(frames, z_thresh: float):
    """Keep only points strictly below z_thresh (after ground-shifting)."""
    out = []
    for pts in frames:
        mask = pts[:, 2] < z_thresh
        if not np.any(mask):
            print("[WARN] a frame has no points below the height threshold "
                  f"({z_thresh}); it will be skipped.")
            continue
        out.append(pts[mask])
    if not out:
        raise RuntimeError("After height filtering no points remain! "
                           "Lower the threshold or check your data.")
    return out


def find_ground_peak(frame_points, max_h=0.5, bins=100):
    """Histogram along Z instead of Y."""
    all_pts = np.vstack(frame_points)
    counts, edges = np.histogram(all_pts[:, 2], bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = centers <= max_h
    print(f"[DEBUG] Z range: {all_pts[:,2].min()} to {all_pts[:,2].max()}")
    if not np.any(mask):
        raise ValueError(f"No histogram bins fall below max_h = {max_h}. Try increasing it.")

    peak_idx = np.argmax(counts[mask])
    ground_peak_h = centers[mask][peak_idx]
    print(f"[INFO] ground_peak_h = {ground_peak_h:.3f}")
    return ground_peak_h, edges


def adjust_frames(frame_points, ground_peak_h):
    """Raise frames where min-z is below the histogram peak."""
    for pts in frame_points:
        frame_min = pts[:, 2].min()
        if frame_min < ground_peak_h:
            shift = ground_peak_h - frame_min
            pts[:, 2] += shift


def compute_ratios(points, edges, grid=0.05):
    """
    For every horizontal slice ( z layer ) compute the occupied-cell ratio
    over the X–Y plane (not X–Z as before).
    """
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


def plot_ratios(centers, ratios, ground_peak_h,
                save_dir, filename="height_distribution.png"):
    width = centers[1] - centers[0]
    plt.figure(figsize=(10, 5))
    plt.bar(centers, ratios, width=width, edgecolor='black')
    plt.axvline(ground_peak_h, color='red', linestyle='--',
                label=f'Ground peak @ {ground_peak_h:.2f}')
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
# ============================================================================


# --------------------------------------------------------------------------- #
#  Per-frame normalisation: every frame’s min-z → 0
# --------------------------------------------------------------------------- #
def normalise_ground_per_frame(frames, eps: float = 1e-6):
    """
    Shift EACH frame so its own minimum z becomes 0.

    For a frame with min-z = m:
        pts[:, 2] += (-m)
    (No shift when |m| ≤ eps.)
    """
    for idx, pts in enumerate(frames):
        m = pts[:, 2].min()
        if abs(m) <= eps:                     # already at ground
            continue
        pts[:, 2] -= m                       # shift up or down
        direction = "down" if m > 0 else "up"
        print(f"[INFO] Frame {idx:03d} shifted {direction} by {abs(m):.3f} "
              "→ min-z = 0")
        

        # NEW: drop any remaining points with z < 0
        before = frames[idx].shape[0]
        mask_pos = frames[idx][:, 2] >= 0.0
        frames[idx] = frames[idx][mask_pos]
        dropped = before - frames[idx].shape[0]
        if dropped > 0:
            print(f"[INFO] Frame {idx:03d}: dropped {dropped} points with z<0 after shift")

def normalize_ground_ransac(
    frames,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000
) -> float:
    """
    Merge all frames into one point cloud, RANSAC-fit the dominant ground plane,
    compute its mean height, then shift each frame so that plane sits at Z=0.
    Returns the mean ground height that was removed.
    """
    # merge
    all_pts = np.vstack(frames)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)

    # RANSAC plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    if len(inliers) == 0:
        raise RuntimeError("RANSAC failed to find any ground plane.")

    ground_pc = pcd.select_by_index(inliers)
    ground_z_mean = 0

    # shift every frame
    for pts in frames:
        pts[:, 2] -= ground_z_mean

    print(f"[INFO] Shifted all frames by {-ground_z_mean:.3f} m so ground plane at z=0")
    return ground_z_mean


def normalize_ground_ransac_per_frame(
    frames,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000
):
    """
    For each frame:
      1) RANSAC-fit its ground plane
      2) Compute its mean ground height
      3) Shift the frame so its plane sits at Z = 0
      4) Drop any points still with z < 0
    """
    ground_heights = []
    for idx, pts in enumerate(frames):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        if not inliers:
            raise RuntimeError(f"Frame {idx:03d}: RANSAC failed to find ground plane.")

        inlier_pts = np.asarray(pcd.select_by_index(inliers).points)
        ground_z_mean = 1.55
        ground_heights.append(ground_z_mean)

        # 3) shift
        frames[idx][:, 2] -= ground_z_mean
        print(f"[INFO] Frame {idx:03d}: shifted by {-ground_z_mean:.3f} m → ground at Z=0")

        # 4) filter out any remaining negatives
        before = frames[idx].shape[0]
        mask_pos = frames[idx][:, 2] >= 0.0
        frames[idx] = frames[idx][mask_pos]
        dropped = before - frames[idx].shape[0]
        if dropped > 0:
            print(f"[INFO] Frame {idx:03d}: dropped {dropped} points with z<0 after shift")

    return ground_heights


if __name__ == "__main__":
    data_dir         = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00"
    NUM_FRAMES       = 99
    HEIGHT_THRESHOLD = 2.0    # units of Z

    # 1) Load & diagnose
    frames_raw = load_frames_world(data_dir, NUM_FRAMES)
    _ = detect_negative_z(frames_raw)

    # 2) **Per-frame RANSAC normalization**
    ground_z_means = normalize_ground_ransac_per_frame(
        frames_raw,
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000
    )

    # 3) Height-based filtering
    frames_clip = filter_by_height(frames_raw, HEIGHT_THRESHOLD)

    # 4) (Optional) histogram-based ground-peak & adjust
    ground_peak_h, edges = find_ground_peak(frames_clip, max_h=0.5, bins=100)
    adjust_frames(frames_clip, ground_peak_h)

    # 5) Compute & plot height-wise ratios
    all_pts = np.vstack(frames_clip)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ratios  = compute_ratios(all_pts, edges, grid=0.05)
    plot_ratios(centers, ratios, ground_peak_h, save_dir=data_dir)