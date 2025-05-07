import open3d as o3d
import numpy as np


def visualize_ply_with_original_color(
    ply_path: str,
    voxel_size: float = 0.1,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    refine_thresh: float = 0.05,
    frame_size: float = 0.5
):
    """
    Load a PLY, clean, flip z-axis, segment ground vs non-ground by RANSAC + height-refine,
    then visualize each segment using the original RGB colors.
    """
    # 1) Load & clean
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"No points loaded from {ply_path}")
    pcd = pcd.voxel_down_sample(voxel_size)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    pcd = pcd.select_by_index(ind)

    # 2) Flip Z
    pts = np.asarray(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 3) RANSAC plane segmentation
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    if not inliers:
        raise RuntimeError("RANSAC failed to find a ground plane.")

    ground_raw = pcd.select_by_index(inliers)
    nonground_raw = pcd.select_by_index(inliers, invert=True)

    # 4) Height-based refinement
    ground_z_mean = np.mean(np.asarray(ground_raw.points)[:, 2])
    pts_ng = np.asarray(nonground_raw.points)
    low_mask = pts_ng[:, 2] < (ground_z_mean + refine_thresh)

    ground_refined = nonground_raw.select_by_index(np.where(low_mask)[0])
    nonground_final = nonground_raw.select_by_index(np.where(low_mask)[0], invert=True)

    # 5) Translate so ground plane at z=0
    translation = (0, 0, -ground_z_mean)
    pcd.translate(translation)
    ground_raw.translate(translation)
    ground_refined.translate(translation)
    nonground_final.translate(translation)

    # 6) Merge ground segments and preserve original colors
    ground_points = np.vstack([
        np.asarray(ground_raw.points),
        np.asarray(ground_refined.points)
    ])
    ground_colors = np.vstack([
        np.asarray(ground_raw.colors),
        np.asarray(ground_refined.colors)
    ])
    ground_final = o3d.geometry.PointCloud()
    ground_final.points = o3d.utility.Vector3dVector(ground_points)
    ground_final.colors = o3d.utility.Vector3dVector(ground_colors)

    # 7) Prepare coordinate frame at origin
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=4,
        origin=(0.0, 0.0, 0.0)
    )

    # 8) Visualize
    o3d.visualization.draw_geometries(
        [ground_final, nonground_final, frame],
        window_name="Segmented Cloud (Original RGB)",
        width=800, height=600, zoom=1
    )


if __name__ == "__main__":
    visualize_ply_with_original_color(
        ply_path="./processed_unreal/00008/1/00199.ply",
        voxel_size=0.1,
        nb_neighbors=20,
        std_ratio=2.0,
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000,
        refine_thresh=0.05,
        frame_size=0.5
    )
