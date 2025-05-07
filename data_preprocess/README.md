## Overview
Details about following folders:
- `preprocess` folder contains files to preprocess data, taking from CUT3R repository
- `voxel` folder contains files to generate voxel maps. They will generate visibility and obstacle voxel maps
- `visualization` folder contains files to generate histogram plot, pointcloud 3D visualization

Details about following files:
- `process_point_cloud` file generate point cloud based on the depth and rgb provided from the dataset
- `ground_segment_all_frames` help detect points that are recognized as the ground and segment them using a different color. This will segment the pointcloud of all frames and merge them together in a single visualization.
- `ground_ptn` file help help detect points that are recognized as the ground in a single frame.
