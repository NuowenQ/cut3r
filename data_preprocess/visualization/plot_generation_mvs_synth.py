import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d

# ─────────── unchanged ─────────────────────────────────────────────────────────
def load_frames(data_dir: str, num_frames: int = 100):
    """Read *.ply files without touching the coordinates."""
    frames = []
    for i in range(num_frames):
        fp = os.path.join(data_dir, f"{i:04d}.ply")
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
    
                pts.append([x, y, z * -1])
        if pts:
            frames.append(np.asarray(pts, dtype=np.float32))

    if not frames:
        raise RuntimeError("No *.ply files found – check data_dir!")

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

# After your existing processing
# -------------------------------------------------------

# Visualize occupied cells at height ~0.5
def visualize_occupied_grid(points, center_height, delta=0.02, grid_size=0.05):
    """
    Visualize the occupied grid cells at a certain height slice.
    
    - center_height: target height (e.g., 0.5m)
    - delta: thickness above and below (e.g., +-0.02m)
    - grid_size: cell size (same as compute_ratios)
    """
    # Step 1: Filter points within [center_height - delta, center_height + delta]
    mask = (points[:, 2] >= center_height - delta) & (points[:, 2] < center_height + delta)
    slice_pts = points[mask]
    
    if slice_pts.shape[0] == 0:
        print(f"[WARN] No points found around height {center_height}m")
        return
    
    # Step 2: Only (x, y)
    xy = slice_pts[:, [0, 1]]

    # Step 3: Define grid bounds
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    side = max(xmax - xmin, ymax - ymin)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    X0, X1 = cx - side / 2, cx + side / 2
    Y0, Y1 = cy - side / 2, cy + side / 2

    # Step 4: Map points into grid
    xi = np.floor((xy[:, 0] - X0) / grid_size).astype(int)
    yi = np.floor((xy[:, 1] - Y0) / grid_size).astype(int)
    occupied = set(zip(xi, yi))
    total_cells = int(np.ceil(side / grid_size)) ** 2
    occupied_ratio = len(occupied) / total_cells if total_cells > 0 else 0

    print(f"[INFO] Occupied cells: {len(occupied)} / {total_cells}")
    print(f"[INFO] Occupied area ratio = {occupied_ratio:.4f}")

    # Step 5: Plot
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(X0, X1, grid_size))
    ax.set_yticks(np.arange(Y0, Y1, grid_size))
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

    # Draw occupied cells
    for (xg, yg) in occupied:
        rect = patches.Rectangle(
            (X0 + xg * grid_size, Y0 + yg * grid_size),
            grid_size, grid_size,
            linewidth=1, edgecolor='red', facecolor='red', alpha=0.3
        )
        ax.add_patch(rect)

    # Draw points
    ax.scatter(xy[:, 0], xy[:, 1], color='blue', s=5, zorder=5)

    plt.title(f"Occupied Area at Height ~{center_height:.2f}m\nRatio = {occupied_ratio:.2f}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()



# def visualize_point_cloud_open3d(frames):
#     """
#     Visualize the combined point cloud using Open3D.
#     """
#     all_pts = np.vstack(frames)  # Merge all frames into one big array

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(all_pts[:, :3])  # Set (x, y, z)

#     # Optionally, colorize based on Z (height)
#     z_vals = all_pts[:, 2]
#     z_min, z_max = z_vals.min(), z_vals.max()
#     normalized_z = (z_vals - z_min) / (z_max - z_min + 1e-8)  # normalize to [0,1]
#     colors = plt.get_cmap('jet')(normalized_z)[:, :3]  # use matplotlib colormap
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # Visualize
#     o3d.visualization.draw_geometries([pcd])

def visualize_loaded_frames_open3d(frames,
                                    frame_size: float = 0.5,
                                    frame_origin: str | list[float] = "centroid"):
    """
    Visualize the loaded point cloud frames with a coordinate frame.

    - Keep original color
    - Flip Z-axis
    - Add coordinate frame at origin / centroid / custom [x,y,z]
    """

    # Merge all frames
    all_pts = np.vstack(frames)

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts[:, :3])  # XYZ only

    # (Optional) check if color information exists (your load_frames didn't save color though)
    # If your frames already have RGB later, you can add:
    # pcd.colors = o3d.utility.Vector3dVector(all_pts[:, 3:6])

    # Flip Z-axis (like your visualize_ply function)
    pts = np.asarray(pcd.points)
    pts[:, 2] *= -1
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Decide where to put the frame
    if frame_origin == "centroid":
        origin = pcd.get_center()
    elif frame_origin == "origin":
        origin = (0.0, 0.0, 0.0)
    elif isinstance(frame_origin, (list, tuple)) and len(frame_origin) == 3:
        origin = frame_origin
    else:
        raise ValueError("frame_origin must be 'origin', 'centroid', or [x,y,z]")

    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=origin)

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name="Loaded Point Cloud",
        width=800,
        height=600,
        zoom=1
    )

if __name__ == "__main__":
    #data_dir        = "./data_output/00008/1"
    data_dir = "./processed_mvs_synth/0001"
    NUM_FRAMES      = 100
    HEIGHT_THRESHOLD = 2.0    # units of Z

    frames_raw   = load_frames(data_dir, NUM_FRAMES)

    # Visualize raw point cloud

    _            = detect_negative_z(frames_raw)      # (1) check for Z < 0
    normalise_ground_per_frame(frames_raw)                  # (2) min-Z → 0
    visualize_loaded_frames_open3d(frames_raw)
    # all_pts = np.vstack(frames_raw)
    # print(f"All points Z min: {all_pts[:,2].min():.3f}, max: {all_pts[:,2].max():.3f}")

    
    # frames_clip  = filter_by_height(frames_raw, HEIGHT_THRESHOLD)     # (3)

    # ground_peak_h, edges = find_ground_peak(frames_clip, max_h=0.5, bins=100)
    # adjust_frames(frames_clip, ground_peak_h)

    # all_pts = np.vstack(frames_clip)
    # centers = 0.5 * (edges[:-1] + edges[1:])
    # ratios  = compute_ratios(all_pts, edges, grid=0.05)

    # plot_ratios(centers, ratios, ground_peak_h, save_dir=data_dir)


