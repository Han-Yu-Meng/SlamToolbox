# Dynamic Removal Branch Notes

This document records the changes prepared for the dynamic-obstacle-removal
branch.

## Summary

The branch changes the dynamic removal workflow to avoid using accumulated
`frame/` point clouds as algorithm input. ERASOR2, Removert, Local Hash Voxel,
and Raycast Voxel now use KITTI-style per-scan data generated from the source
bag whenever possible.

The main goal is to keep each scan in the correct local sensor/base frame while
preserving the real sensor trajectory in `poses_odom_base.txt`.

## Data Conversion

- `frame/` is no longer used as the default source for KITTI conversion.
- KITTI data is generated directly from the source bag point cloud topic,
  usually `/cloud_registered`.
- If the source point cloud is already in an odom/global frame, conversion now
  transforms points through:

```text
source_cloud_frame -> fixed_frame/odom -> base_link
```

- The written `velodyne/*.bin` files are local `base_link` scans.
- `poses_odom_base.txt` stores the true `odom -> base_link` trajectory.
- `poses_suma_optim.txt` remains the ERASOR2-compatible compensated trajectory.
- `conversion_notes.txt` now records `point_transform` so newer converted
  datasets can be distinguished from old or frame-based datasets.
- Old KITTI datasets without the new bag-local conversion marker are discarded
  and regenerated from bag.

This fixes the previous Raycast Voxel failure mode where `/cloud_registered`
was treated as local scan data even though it was already in an odom/global
frame. Raycasting then applied `poses_odom_base.txt` again, which caused
double-transforming and scattered output.

## ERASOR2

- ERASOR2 conversion now uses bag-derived per-scan KITTI data instead of
  accumulated `frame/` PCD files.
- This avoids generating very large accumulated KITTI frames that can exceed
  ERASOR2 mapgen's fixed loader buffer.
- The earlier point/label mismatch issue is documented separately in
  `docs/issues/erasor2-large-frame-load-limit.md`.

## Removert

- Removert no longer requires `frame/` to exist before running.
- It uses the same bag-derived KITTI dataset path as the other dynamic removal
  methods.
- Existing old/frame-based KITTI data is regenerated from bag before use.
- Runtime output is now shown as stage-level progress bars:

```text
读取点云
构建地图
动态清除
写出结果
```

- The full Docker output is still written to `removert_docker.log`.

## Local Hash Voxel

- Added Local Hash Voxel dynamic removal as a selectable method.
- The method reads local KITTI scans and `poses_odom_base.txt`.
- It rejects datasets that only have identity/no-motion poses, because the
  method needs real sensor origins.
- This means YunJingFull-like datasets without real sensor trajectory are
  intentionally rejected with an explanatory error.

## Raycast Voxel

- Added Raycast Voxel Cleanup as a selectable dynamic removal method.
- The method reads local KITTI scans and `poses_odom_base.txt`.
- It uses raycasting from real sensor poses to classify occupied/free voxels.
- Outputs are copied to the map directory as:

```text
map_raycast_voxel_static.pcd
map_raycast_voxel_removed.pcd
```

- For the `BLGX_3.25` map, the KITTI dataset was regenerated after fixing the
  bag-to-local conversion. The old raycast output in `map/` must be regenerated
  by running Raycast Voxel again.

## Timestamped Outputs

Dynamic removal and map-building outputs are now stored in timestamped run
directories:

```text
<map>/runs/erasor2/YYYYmmdd_HHMMSS/
<map>/runs/removert/YYYYmmdd_HHMMSS/
<map>/runs/local_hash_voxel/YYYYmmdd_HHMMSS/
<map>/runs/raycast_voxel/YYYYmmdd_HHMMSS/
<map>/runs/map_builder/YYYYmmdd_HHMMSS/
```

The latest user-facing results are still copied into:

```text
<map>/map/
```

## Interactive SLAM Docker GUI

- The Docker GUI launch was updated for X11 access.
- The launcher grants local root X11 access with `xhost`.
- `DISPLAY` and `XAUTHORITY` are passed into the container.
- The current map is mounted both as `/Map` and `/root/Map` so saved map files
  are visible from the host user's map directory.

## CLI Flow

- The map action menu now includes `更换地图`.
- Selecting it returns to the previous map-selection step instead of exiting the
  whole tool.

## Verification Performed

- `python3 -m py_compile slam_toolbox/dynamic_removal.py`
- Simulated Removert progress-log parsing.
- Regenerated `/home/timory/Map/BLGX_3.25/erasor2_dataset` from bag:
  4180 scans, 4180 poses, 4180 timestamps.
- Spot-checked regenerated `BLGX_3.25` scans and confirmed that sample frames
  are now in local coordinates rather than large odom/global coordinates.

## Push Notes

Before pushing, rerun the affected method on any map whose previous output was
generated from the old conversion logic. In particular, rerun Raycast Voxel for
`BLGX_3.25` because the existing `map_raycast_voxel_static.pcd` and
`map_raycast_voxel_removed.pcd` were produced before the conversion fix.
