import numpy as np
import os
import open3d as o3d

def load_frames(data_dir: str, num_frames: int = 100):
    """Read *.ply files, flipping Z."""
    frames = []
    for i in range(num_frames):
        fp = os.path.join(data_dir, f"{i:06d}.ply")
        if not os.path.exists(fp):
            print(f"[WARN] missing file: {fp}")
            continue

        pts = []
        with open(fp, "r") as f:
            in_header = True
            for line in f:
                if in_header:
                    if line.strip() == "end_header":
                        in_header = False
                    continue
                x, y, z = map(float, line.split()[:3])
                pts.append([x, y, -z])  # Flip Z
        if pts:
            frames.append(np.asarray(pts, dtype=np.float32))
    if not frames:
        raise RuntimeError("No .ply files found – check data_dir!")
    return frames

def save_segmented_frame(
    pts: np.ndarray,
    idx: int,
    output_dir: str,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    frame_size: float = 0.5,
    max_height: float = 1.5 # Added: upper bound height above ground
):
    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # RANSAC plane fitting
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    if not inliers:
        print(f"[WARN] Frame {idx:03d}: no ground plane found")
        return

    ground = pcd.select_by_index(inliers)
    nonground = pcd.select_by_index(inliers, invert=True)

    # Compute mean ground height
    ground_pts = np.asarray(ground.points)
    ground_z_mean = float(ground_pts[:, 2].mean())
    print(f"[INFO] Frame {idx:03d}: shifted by {-ground_z_mean:.3f} m → ground at Z=0")

    # Shift ground and non-ground points to Z=0
    ground_pts[:, 2] -= ground_z_mean
    nonground_pts = np.asarray(nonground.points)
    nonground_pts[:, 2] -= ground_z_mean

    # Apply upper bound: keep points within max_height above ground
    keep_ground = ground_pts[:, 2] <= max_height
    ground_pts = ground_pts[keep_ground]
    keep_nonground = nonground_pts[:, 2] <= max_height
    nonground_pts = nonground_pts[keep_nonground]

    # Update point clouds
    ground.points = o3d.utility.Vector3dVector(ground_pts)
    nonground.points = o3d.utility.Vector3dVector(nonground_pts)

    # Save ground and non-ground points
    os.makedirs(output_dir, exist_ok=True)
    ground_path = os.path.join(output_dir, f"ground_{idx:06d}.ply")
    nonground_path = os.path.join(output_dir, f"nonground_{idx:06d}.ply")
    o3d.io.write_point_cloud(ground_path, ground)
    o3d.io.write_point_cloud(nonground_path, nonground)
    print(f"[✓] Saved: {ground_path}, {nonground_path}")

if __name__ == "__main__":
    data_dir   = "/home/nuowen/research/cut3r/hypersim_processed/ai_001_001/cam_00"
    NUM_FRAMES = 98
    out_dir    = os.path.join(data_dir, "segmented_frames")

    frames = load_frames(data_dir, NUM_FRAMES)
    for idx, pts in enumerate(frames):
        save_segmented_frame(
            pts, idx, out_dir,
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000,
            frame_size=0.5,
            max_height=1.5  # use 3m upper bound
        )


