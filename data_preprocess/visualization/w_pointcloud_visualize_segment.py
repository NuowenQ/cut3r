import open3d as o3d
import numpy as np

def visualize_ply_with_segmentation(
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
    # 1) Load & clean
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"No points loaded from {ply_path}")
    pcd = pcd.voxel_down_sample(voxel_size)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    pcd = pcd.select_by_index(ind)

    # 2) Flip Z so that “down” → negative becomes positive
    pts = np.asarray(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 3) RANSAC plane fit
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    if not inliers:
        raise RuntimeError("RANSAC failed to find a ground plane.")

    ground_raw   = pcd.select_by_index(inliers)
    nongroundRaw = pcd.select_by_index(inliers, invert=True)

    # 4) Height-refine
    ground_z_mean = np.mean(np.asarray(ground_raw.points)[:, 2])
    pts_ng = np.asarray(nongroundRaw.points)
    low_mask = pts_ng[:, 2] < (ground_z_mean + refine_thresh)

    ground_refined  = nongroundRaw.select_by_index(np.where(low_mask)[0])
    nonground_final = nongroundRaw.select_by_index(np.where(low_mask)[0], invert=True)

    # 5) Translate so ground sits at Z=0
    #    (since we shift by the plane mean, ground_raw's plane → Z≈0,
    #     but ground_refined may be slightly above; this is fine)
    pcd.translate((0, 0, -ground_z_mean))
    ground_raw.translate((0, 0, -ground_z_mean))
    ground_refined.translate((0, 0, -ground_z_mean))
    nonground_final.translate((0, 0, -ground_z_mean))

    # 6) ***CORRECT MERGING*** of the two ground sets
    ground_pts = np.vstack([
        np.asarray(ground_raw.points),
        np.asarray(ground_refined.points)
    ])
    ground_final = o3d.geometry.PointCloud()
    ground_final.points = o3d.utility.Vector3dVector(ground_pts)

    # 7) Colorize
    ground_final.paint_uniform_color([0.1, 0.9, 0.1])
    nonground_final.paint_uniform_color([0.9, 0.1, 0.1])

    # 8) Build frame at the true origin
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=4,
        origin=(0.0, 0.0, 0.0)
    )

    # 9) Visualize
    o3d.visualization.draw_geometries(
        [ground_final, nonground_final, frame],
        window_name="Ground vs. Obstacles",
        width=800, height=600, zoom=1
    )

if __name__ == "__main__":
    visualize_ply_with_segmentation(
        "./processed_unreal/00008/1/00198.ply",
        voxel_size=0.1,
        nb_neighbors=20,
        std_ratio=2.0,
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000,
        refine_thresh=0.05,
        frame_size=0.5
    )
