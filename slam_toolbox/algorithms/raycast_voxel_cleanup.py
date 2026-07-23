#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


class OctreeNode:
    __slots__ = (
        "origin",
        "size",
        "children",
        "log_odds",
        "hit_frames",
        "miss_frames",
        "hit_points",
    )

    def __init__(self, origin: tuple[int, int, int], size: int) -> None:
        self.origin = origin
        self.size = size
        self.children: list[OctreeNode | None] | None = None
        self.log_odds = 0.0
        self.hit_frames = 0
        self.miss_frames = 0
        self.hit_points = 0

    @property
    def is_leaf_cell(self) -> bool:
        return self.size == 1


class OctreeMap:
    def __init__(self, min_log_odds: float, max_log_odds: float) -> None:
        self.root: OctreeNode | None = None
        self.min_log_odds = min_log_odds
        self.max_log_odds = max_log_odds
        self.leaf_count = 0

    @staticmethod
    def _contains(node: OctreeNode, coord: tuple[int, int, int]) -> bool:
        ox, oy, oz = node.origin
        size = node.size
        return (
            ox <= coord[0] < ox + size
            and oy <= coord[1] < oy + size
            and oz <= coord[2] < oz + size
        )

    def _ensure_contains(self, coord: tuple[int, int, int]) -> None:
        if self.root is None:
            self.root = OctreeNode(coord, 1)
            return

        while not self._contains(self.root, coord):
            old = self.root
            old_origin = old.origin
            old_size = old.size
            new_size = old_size * 2
            new_origin = (
                old_origin[0] - old_size if coord[0] < old_origin[0] else old_origin[0],
                old_origin[1] - old_size if coord[1] < old_origin[1] else old_origin[1],
                old_origin[2] - old_size if coord[2] < old_origin[2] else old_origin[2],
            )
            new_root = OctreeNode(new_origin, new_size)
            new_root.children = [None] * 8
            ix = 1 if old_origin[0] >= new_origin[0] + old_size else 0
            iy = 1 if old_origin[1] >= new_origin[1] + old_size else 0
            iz = 1 if old_origin[2] >= new_origin[2] + old_size else 0
            new_root.children[ix | (iy << 1) | (iz << 2)] = old
            self.root = new_root

    def get_or_create_leaf(self, coord: tuple[int, int, int]) -> OctreeNode:
        self._ensure_contains(coord)
        assert self.root is not None
        node = self.root
        while node.size > 1:
            half = node.size // 2
            ox, oy, oz = node.origin
            ix = 1 if coord[0] >= ox + half else 0
            iy = 1 if coord[1] >= oy + half else 0
            iz = 1 if coord[2] >= oz + half else 0
            child_idx = ix | (iy << 1) | (iz << 2)
            child_origin = (ox + ix * half, oy + iy * half, oz + iz * half)
            if node.children is None:
                node.children = [None] * 8
            child = node.children[child_idx]
            if child is None:
                child = OctreeNode(child_origin, half)
                node.children[child_idx] = child
                if half == 1:
                    self.leaf_count += 1
            node = child
        if self.leaf_count == 0 and self.root is node:
            self.leaf_count = 1
        return node

    def get_leaf(self, coord: tuple[int, int, int]) -> OctreeNode | None:
        node = self.root
        if node is None or not self._contains(node, coord):
            return None
        while node.size > 1:
            if node.children is None:
                return None
            half = node.size // 2
            ox, oy, oz = node.origin
            ix = 1 if coord[0] >= ox + half else 0
            iy = 1 if coord[1] >= oy + half else 0
            iz = 1 if coord[2] >= oz + half else 0
            node = node.children[ix | (iy << 1) | (iz << 2)]
            if node is None:
                return None
        return node

    def update_hit(self, coord: tuple[int, int, int], count: int, hit_log_odds: float) -> None:
        leaf = self.get_or_create_leaf(coord)
        leaf.hit_frames += 1
        leaf.hit_points += count
        leaf.log_odds = min(self.max_log_odds, leaf.log_odds + hit_log_odds)

    def update_miss(self, coord: tuple[int, int, int], miss_log_odds: float) -> None:
        leaf = self.get_leaf(coord)
        if leaf is None:
            return
        leaf.miss_frames += 1
        leaf.log_odds = max(self.min_log_odds, leaf.log_odds - miss_log_odds)

    def is_occupied(self, coord: tuple[int, int, int], min_hit_frames: int, occupied_threshold: float) -> bool:
        leaf = self.get_leaf(coord)
        return bool(
            leaf is not None
            and leaf.hit_frames >= min_hit_frames
            and leaf.log_odds >= occupied_threshold
        )

    def count_removed_or_free(self, min_miss_frames: int, free_threshold: float) -> int:
        count = 0
        for leaf in self.iter_leaves():
            if leaf.miss_frames >= min_miss_frames or leaf.log_odds <= free_threshold:
                count += 1
        return count

    def count_occupied(self, min_hit_frames: int, occupied_threshold: float) -> int:
        count = 0
        for leaf in self.iter_leaves():
            if leaf.hit_frames >= min_hit_frames and leaf.log_odds >= occupied_threshold:
                count += 1
        return count

    def iter_leaves(self):
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.size == 1:
                yield node
            elif node.children is not None:
                stack.extend(child for child in node.children if child is not None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Raycast voxel cleanup for converted KITTI-style lidar sequences. "
            "Use local scans plus true poses_odom_base.txt; do not use identity global scans."
        )
    )
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--seq", default="00")
    p.add_argument("--pose", type=Path, help="Default: dataset/sequences/<seq>/poses_odom_base.txt")
    p.add_argument("--out", type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    p.add_argument("--voxel-size", type=float, default=0.30)
    p.add_argument("--max-range", type=float, default=35.0, help="Horizontal range in local frame; 0 disables.")
    p.add_argument("--body-radius", type=float, default=0.8, help="Drop points near the vehicle in local xy; 0 disables.")
    p.add_argument(
        "--ground-protect-local-z-max",
        type=float,
        help=(
            "Force-keep points whose original local/base_link z is <= this value. "
            "Useful when raycasting removes floor points; try 0.0 or -0.1."
        ),
    )
    p.add_argument("--ray-point-stride", type=int, default=8, help="Use every Nth point for free-space raycasting.")
    p.add_argument("--ray-step-factor", type=float, default=0.75, help="Ray sample step = voxel_size * factor.")
    p.add_argument("--endpoint-margin", type=float, default=0.60, help="Meters before endpoint left untouched by free rays.")
    p.add_argument("--hit-log-odds", type=float, default=0.85)
    p.add_argument("--miss-log-odds", type=float, default=0.45)
    p.add_argument("--occupied-threshold", type=float, default=0.5)
    p.add_argument("--free-threshold", type=float, default=-1.0)
    p.add_argument("--min-hit-frames", type=int, default=2)
    p.add_argument("--min-miss-frames", type=int, default=2)
    p.add_argument("--max-log-odds", type=float, default=5.0)
    p.add_argument("--min-log-odds", type=float, default=-5.0)
    p.add_argument(
        "--deduplicate",
        choices=("none", "exact", "quantized"),
        default="none",
        help=(
            "Deduplicate raycast_after.pcd only. exact removes identical float32 xyzi points; "
            "quantized keeps one point per coordinate bin of --dedup-resolution."
        ),
    )
    p.add_argument(
        "--dedup-resolution",
        type=float,
        default=0.02,
        help="Coordinate resolution in meters for --deduplicate quantized.",
    )
    p.add_argument(
        "--spatial-sort-after",
        action="store_true",
        help=(
            "Spatially sort raycast_after.pcd before writing. This can make CloudCompare navigation "
            "smoother for accumulated multi-frame clouds, at the cost of extra time and memory."
        ),
    )
    p.add_argument(
        "--spatial-sort-resolution",
        type=float,
        default=0.0,
        help="Coordinate resolution in meters for --spatial-sort-after; 0 uses --voxel-size.",
    )
    p.add_argument(
        "--interleave-after",
        action="store_true",
        help=(
            "Interleave raycast_after.pcd by spatial bins before writing. This is aimed at "
            "CloudCompare interaction LOD: any sampled subset covers the whole map better than "
            "scan-order or pure spatial-order output."
        ),
    )
    p.add_argument(
        "--interleave-resolution",
        type=float,
        default=0.0,
        help="Coordinate resolution in meters for --interleave-after; 0 uses --voxel-size.",
    )
    p.add_argument(
        "--shuffle-after",
        action="store_true",
        help=(
            "Deterministically shuffle raycast_after.pcd before writing. This often matches original "
            "large maps better in CloudCompare interaction LOD because sampled subsets cover the whole map."
        ),
    )
    p.add_argument("--shuffle-seed", type=int, default=0, help="Random seed for --shuffle-after.")
    p.add_argument("--write-before", action="store_true", help="Also write raycast_before.pcd.")
    p.add_argument("--progress-interval", type=int, default=20)
    return p.parse_args()


def load_poses(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) == 12:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
        elif len(vals) == 16:
            mat = np.asarray(vals, dtype=np.float64).reshape(4, 4)
        else:
            raise ValueError(f"unsupported pose line {line_no}: {len(vals)} values in {path}")
        poses.append(mat)
    return poses


def transform_scan(scan_path: Path, pose: np.ndarray, body_radius: float, max_range: float) -> tuple[np.ndarray, np.ndarray]:
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    scan = scan[np.isfinite(scan).all(axis=1)]
    if scan.size == 0:
        return scan, scan
    dist2 = scan[:, 0].astype(np.float64) ** 2 + scan[:, 1].astype(np.float64) ** 2
    keep = np.ones(scan.shape[0], dtype=bool)
    if body_radius > 0:
        keep &= dist2 >= body_radius * body_radius
    if max_range > 0:
        keep &= dist2 <= max_range * max_range
    scan = scan[keep]
    if scan.size == 0:
        return scan, scan
    xyz = scan[:, :3].astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
    out = np.empty_like(scan, dtype=np.float32)
    out[:, :3] = xyz.astype(np.float32)
    out[:, 3] = scan[:, 3]
    return out, scan


def voxel_coords_for_points(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(xyz.astype(np.float64) / voxel_size).astype(np.int64)


def ray_free_coords(
    origin: np.ndarray,
    endpoints: np.ndarray,
    voxel_size: float,
    step_factor: float,
    endpoint_margin: float,
) -> set[tuple[int, int, int]]:
    free: set[tuple[int, int, int]] = set()
    step = max(voxel_size * step_factor, voxel_size * 0.25)
    for endpoint in endpoints:
        vec = endpoint.astype(np.float64) - origin
        dist = float(np.linalg.norm(vec))
        usable = dist - endpoint_margin
        if usable <= step:
            continue
        samples = max(1, int(math.floor(usable / step)))
        ts = (np.arange(1, samples + 1, dtype=np.float64) * step) / dist
        pts = origin[None, :] + ts[:, None] * vec[None, :]
        coords = np.unique(voxel_coords_for_points(pts, voxel_size), axis=0)
        free.update((int(c[0]), int(c[1]), int(c[2])) for c in coords)
    return free


def dedup_mask(points: np.ndarray, mode: str, resolution: float, seen: set) -> np.ndarray:
    if mode == "none" or points.size == 0:
        return np.ones(points.shape[0], dtype=bool)
    if mode == "quantized" and resolution <= 0:
        raise ValueError("--dedup-resolution must be > 0 when --deduplicate quantized")

    keep = np.zeros(points.shape[0], dtype=bool)
    if mode == "exact":
        contiguous = np.ascontiguousarray(points.astype(np.float32, copy=False))
        keys = contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).reshape(-1)
        for i, key in enumerate(keys):
            key_bytes = bytes(key)
            if key_bytes in seen:
                continue
            seen.add(key_bytes)
            keep[i] = True
    elif mode == "quantized":
        q = np.floor(points[:, :3].astype(np.float64) / resolution).astype(np.int64)
        for i, coord in enumerate(q):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            if key in seen:
                continue
            seen.add(key)
            keep[i] = True
    else:
        raise ValueError(f"unsupported deduplication mode: {mode}")
    return keep


def write_pcd_header(out_path: Path, point_count: int) -> None:
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {point_count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {point_count}\n"
        "DATA binary\n"
    ).encode("ascii")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(header)


def write_pcd_from_payload(payload_path: Path, point_count: int, out_path: Path) -> None:
    write_pcd_header(out_path, point_count)
    with out_path.open("ab") as dst, payload_path.open("rb") as src:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)


def spatially_sort_payload(
    payload_path: Path,
    point_count: int,
    resolution: float,
    sorted_payload_path: Path,
) -> None:
    if point_count == 0:
        sorted_payload_path.write_bytes(b"")
        return
    if resolution <= 0:
        raise ValueError("spatial sort resolution must be > 0")

    points = np.memmap(payload_path, dtype=np.float32, mode="r", shape=(point_count, 4))
    coords = np.floor(points[:, :3].astype(np.float64) / resolution).astype(np.int64)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))

    chunk_size = 1_000_000
    with sorted_payload_path.open("wb") as out:
        for start in range(0, point_count, chunk_size):
            idx = order[start : start + chunk_size]
            points[idx].astype(np.float32, copy=False).tofile(out)
    del points


def interleave_payload_by_spatial_bins(
    payload_path: Path,
    point_count: int,
    resolution: float,
    interleaved_payload_path: Path,
) -> None:
    if point_count == 0:
        interleaved_payload_path.write_bytes(b"")
        return
    if resolution <= 0:
        raise ValueError("interleave resolution must be > 0")

    points = np.memmap(payload_path, dtype=np.float32, mode="r", shape=(point_count, 4))
    coords = np.floor(points[:, :3].astype(np.float64) / resolution).astype(np.int64)
    group_order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    sorted_coords = coords[group_order]
    new_group = np.empty(point_count, dtype=bool)
    new_group[0] = True
    new_group[1:] = np.any(sorted_coords[1:] != sorted_coords[:-1], axis=1)
    group_starts = np.flatnonzero(new_group)
    group_lengths = np.diff(np.append(group_starts, point_count))
    rank_sorted = np.arange(point_count, dtype=np.int64) - np.repeat(group_starts, group_lengths)
    rank = np.empty(point_count, dtype=np.int64)
    rank[group_order] = rank_sorted
    del group_order, sorted_coords, new_group, group_starts, group_lengths, rank_sorted
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], rank))

    chunk_size = 1_000_000
    with interleaved_payload_path.open("wb") as out:
        for start in range(0, point_count, chunk_size):
            idx = order[start : start + chunk_size]
            points[idx].astype(np.float32, copy=False).tofile(out)
    del points


def shuffle_payload(
    payload_path: Path,
    point_count: int,
    seed: int,
    shuffled_payload_path: Path,
) -> None:
    if point_count == 0:
        shuffled_payload_path.write_bytes(b"")
        return
    points = np.memmap(payload_path, dtype=np.float32, mode="r", shape=(point_count, 4))
    rng = np.random.default_rng(seed)
    order = rng.permutation(point_count)
    chunk_size = 1_000_000
    with shuffled_payload_path.open("wb") as out:
        for start in range(0, point_count, chunk_size):
            idx = order[start : start + chunk_size]
            points[idx].astype(np.float32, copy=False).tofile(out)
    del points


def choose_out_dir(dataset: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    workspace = Path(__file__).resolve().parent.parent
    run_root = workspace / "run_results" / dataset.resolve().name / "raycast_voxel_runs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_raycast_voxel"
    out = run_root / stamp
    suffix = 1
    while out.exists():
        out = run_root / f"{stamp}_{suffix}"
        suffix += 1
    return out


def print_progress(phase: str, done: int, total: int, started: float, detail: str = "") -> None:
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, done / total))
    percent = 100.0 * fraction
    elapsed = time.time() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    bar_width = 30
    filled = int(bar_width * fraction)
    bar = "=" * filled + "-" * (bar_width - filled)
    suffix = f" | {detail}" if detail else ""
    print(
        f"[{phase:<9}] [{bar}] {percent:6.2f}% "
        f"({done}/{total}) elapsed={elapsed:.1f}s eta={eta:.1f}s{suffix}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be > 0")
    if args.stride <= 0 or args.ray_point_stride <= 0:
        raise ValueError("--stride and --ray-point-stride must be > 0")
    if args.ray_step_factor <= 0:
        raise ValueError("--ray-step-factor must be > 0")
    if args.deduplicate == "quantized" and args.dedup_resolution <= 0:
        raise ValueError("--dedup-resolution must be > 0 when --deduplicate quantized")
    if args.spatial_sort_after and args.spatial_sort_resolution < 0:
        raise ValueError("--spatial-sort-resolution must be >= 0")
    if args.interleave_after and args.interleave_resolution < 0:
        raise ValueError("--interleave-resolution must be >= 0")
    reorder_count = int(args.spatial_sort_after) + int(args.interleave_after) + int(args.shuffle_after)
    if reorder_count > 1:
        raise ValueError("--spatial-sort-after, --interleave-after, and --shuffle-after are mutually exclusive")

    seq_dir = args.dataset / "dataset" / "sequences" / args.seq
    velodyne_dir = seq_dir / "velodyne"
    scan_paths = sorted(velodyne_dir.glob("*.bin"))
    if not scan_paths:
        raise FileNotFoundError(f"no .bin scans found in {velodyne_dir}")
    pose_path = args.pose or (seq_dir / "poses_odom_base.txt")
    if not pose_path.exists():
        raise FileNotFoundError(f"pose file not found: {pose_path}")
    poses = load_poses(pose_path)

    end = args.end if args.end is not None else len(scan_paths) - 1
    frame_ids = list(range(args.start, end + 1, args.stride))
    if args.start < 0 or end >= len(scan_paths) or not frame_ids:
        raise ValueError(f"invalid frame range {args.start}..{end}; scan count={len(scan_paths)}")
    if len(poses) <= end:
        raise ValueError(f"pose count {len(poses)} is smaller than end frame {end}")
    if pose_path.name == "poses_identity.txt":
        raise ValueError(
            "poses_identity.txt does not provide real sensor origins. "
            "Use datasets with poses_odom_base.txt for raycasting."
        )

    out_dir = choose_out_dir(args.dataset, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="raycast_voxel_", dir=str(out_dir)))

    octree = OctreeMap(args.min_log_odds, args.max_log_odds)

    started = time.time()
    total_hit_points = 0
    total_ray_points = 0
    print(
        f"[raycast] dataset={args.dataset} seq={args.seq} frames={args.start}..{end} "
        f"step={args.stride} pose={pose_path} out={out_dir}",
        flush=True,
    )

    for done, frame_id in enumerate(frame_ids, 1):
        pose = poses[frame_id]
        origin = pose[:3, 3].astype(np.float64)
        cloud, _local_cloud = transform_scan(scan_paths[frame_id], pose, args.body_radius, args.max_range)
        if cloud.size == 0:
            continue

        xyz = cloud[:, :3]
        hit_coords_all = voxel_coords_for_points(xyz, args.voxel_size)
        unique_hit_coords, counts = np.unique(hit_coords_all, axis=0, return_counts=True)
        total_hit_points += int(cloud.shape[0])

        ray_xyz = xyz[:: args.ray_point_stride]
        total_ray_points += int(ray_xyz.shape[0])
        free_coords = ray_free_coords(
            origin,
            ray_xyz,
            args.voxel_size,
            args.ray_step_factor,
            args.endpoint_margin,
        )
        hit_coord_set = {(int(c[0]), int(c[1]), int(c[2])) for c in unique_hit_coords}
        free_coords.difference_update(hit_coord_set)

        for coord in free_coords:
            octree.update_miss(coord, args.miss_log_odds)

        for coord_np, count_np in zip(unique_hit_coords, counts):
            coord = (int(coord_np[0]), int(coord_np[1]), int(coord_np[2]))
            octree.update_hit(coord, int(count_np), args.hit_log_odds)

        if args.progress_interval and (
            done == 1 or done % args.progress_interval == 0 or done == len(frame_ids)
        ):
            print_progress(
                "integrate",
                done,
                len(frame_ids),
                started,
                f"octree_leaves={octree.leaf_count} hit_points={total_hit_points} ray_points={total_ray_points}",
            )

    kept_leaf_count = octree.count_occupied(args.min_hit_frames, args.occupied_threshold)
    removed_or_free_leaf_count = octree.count_removed_or_free(args.min_miss_frames, args.free_threshold)

    after_payload = tmp_dir / "after.payload"
    removed_payload = tmp_dir / "removed.payload"
    before_payload = tmp_dir / "before.payload"
    after_count = 0
    after_pre_dedup_count = 0
    deduplicated_after_points = 0
    removed_count = 0
    before_count = 0
    dedup_seen: set = set()

    second_started = time.time()
    with after_payload.open("wb") as after_f, removed_payload.open("wb") as removed_f:
        before_f = before_payload.open("wb") if args.write_before else None
        try:
            for done, frame_id in enumerate(frame_ids, 1):
                cloud, local_cloud = transform_scan(scan_paths[frame_id], poses[frame_id], args.body_radius, args.max_range)
                if cloud.size == 0:
                    continue
                coords = voxel_coords_for_points(cloud[:, :3], args.voxel_size)
                keep_mask = np.fromiter(
                    (
                        octree.is_occupied(
                            (int(c[0]), int(c[1]), int(c[2])),
                            args.min_hit_frames,
                            args.occupied_threshold,
                        )
                        for c in coords
                    ),
                    dtype=bool,
                    count=coords.shape[0],
                )
                if args.ground_protect_local_z_max is not None:
                    keep_mask |= local_cloud[:, 2] <= args.ground_protect_local_z_max
                kept = cloud[keep_mask]
                removed = cloud[~keep_mask]
                if kept.size:
                    kept = kept.astype(np.float32, copy=False)
                    after_pre_dedup_count += int(kept.shape[0])
                    dedup_keep = dedup_mask(kept, args.deduplicate, args.dedup_resolution, dedup_seen)
                    kept_dedup = kept[dedup_keep]
                    if kept_dedup.size:
                        kept_dedup.tofile(after_f)
                    after_count += int(kept_dedup.shape[0])
                    deduplicated_after_points += int(kept.shape[0] - kept_dedup.shape[0])
                if removed.size:
                    removed.astype(np.float32, copy=False).tofile(removed_f)
                    removed_count += int(removed.shape[0])
                if before_f is not None:
                    cloud.astype(np.float32, copy=False).tofile(before_f)
                    before_count += int(cloud.shape[0])
                if args.progress_interval and (
                    done == 1 or done % args.progress_interval == 0 or done == len(frame_ids)
                ):
                    print_progress(
                        "write",
                        done,
                        len(frame_ids),
                        second_started,
                        f"after={after_count} removed={removed_count} deduped={deduplicated_after_points}",
                    )
        finally:
            if before_f is not None:
                before_f.close()

    after_payload_to_write = after_payload
    spatial_sort_resolution = args.spatial_sort_resolution or args.voxel_size
    interleave_resolution = args.interleave_resolution or args.voxel_size
    if args.spatial_sort_after:
        sort_started = time.time()
        sorted_after_payload = tmp_dir / "after_spatial_sorted.payload"
        print(
            f"[sort     ] spatially sorting after cloud at resolution={spatial_sort_resolution:g} "
            f"points={after_count}",
            flush=True,
        )
        spatially_sort_payload(after_payload, after_count, spatial_sort_resolution, sorted_after_payload)
        print(f"[sort     ] done elapsed={time.time() - sort_started:.1f}s", flush=True)
        after_payload_to_write = sorted_after_payload
    if args.interleave_after:
        interleave_started = time.time()
        interleaved_after_payload = tmp_dir / "after_interleaved.payload"
        print(
            f"[interleave] interleaving after cloud at resolution={interleave_resolution:g} "
            f"points={after_count}",
            flush=True,
        )
        interleave_payload_by_spatial_bins(after_payload, after_count, interleave_resolution, interleaved_after_payload)
        print(f"[interleave] done elapsed={time.time() - interleave_started:.1f}s", flush=True)
        after_payload_to_write = interleaved_after_payload
    if args.shuffle_after:
        shuffle_started = time.time()
        shuffled_after_payload = tmp_dir / "after_shuffled.payload"
        print(
            f"[shuffle ] shuffling after cloud seed={args.shuffle_seed} points={after_count}",
            flush=True,
        )
        shuffle_payload(after_payload, after_count, args.shuffle_seed, shuffled_after_payload)
        print(f"[shuffle ] done elapsed={time.time() - shuffle_started:.1f}s", flush=True)
        after_payload_to_write = shuffled_after_payload

    write_pcd_from_payload(after_payload_to_write, after_count, out_dir / "raycast_after.pcd")
    write_pcd_from_payload(removed_payload, removed_count, out_dir / "raycast_removed.pcd")
    if args.write_before:
        write_pcd_from_payload(before_payload, before_count, out_dir / "raycast_before.pcd")

    summary = [
        f"dataset: {args.dataset}",
        f"seq: {args.seq}",
        f"pose: {pose_path}",
        f"frames: {args.start}..{end}",
        f"stride: {args.stride}",
        f"voxel_size: {args.voxel_size}",
        f"max_range: {args.max_range}",
        f"body_radius: {args.body_radius}",
        f"ground_protect_local_z_max: {args.ground_protect_local_z_max}",
        f"ray_point_stride: {args.ray_point_stride}",
        f"ray_step_factor: {args.ray_step_factor}",
        f"endpoint_margin: {args.endpoint_margin}",
        f"hit_log_odds: {args.hit_log_odds}",
        f"miss_log_odds: {args.miss_log_odds}",
        f"occupied_threshold: {args.occupied_threshold}",
        f"free_threshold: {args.free_threshold}",
        f"min_hit_frames: {args.min_hit_frames}",
        f"min_miss_frames: {args.min_miss_frames}",
        f"deduplicate: {args.deduplicate}",
        f"dedup_resolution: {args.dedup_resolution}",
        f"spatial_sort_after: {args.spatial_sort_after}",
        f"spatial_sort_resolution: {spatial_sort_resolution}",
        f"interleave_after: {args.interleave_after}",
        f"interleave_resolution: {interleave_resolution}",
        f"shuffle_after: {args.shuffle_after}",
        f"shuffle_seed: {args.shuffle_seed}",
        "map_structure: octree",
        f"octree_root_origin: {octree.root.origin if octree.root is not None else None}",
        f"octree_root_size_voxels: {octree.root.size if octree.root is not None else 0}",
        f"octree_leaf_voxels: {octree.leaf_count}",
        f"kept_voxels: {kept_leaf_count}",
        f"removed_or_free_voxels: {removed_or_free_leaf_count}",
        f"input_points: {total_hit_points}",
        f"raycast_sample_points: {total_ray_points}",
        f"after_points_before_dedup: {after_pre_dedup_count}",
        f"deduplicated_after_points: {deduplicated_after_points}",
        f"after_points: {after_count}",
        f"removed_points: {removed_count}",
        f"elapsed_sec: {time.time() - started:.3f}",
    ]
    (out_dir / "raycast_summary.txt").write_text("\n".join(summary) + "\n")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[done] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
