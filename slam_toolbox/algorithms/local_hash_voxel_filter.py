#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np


KEY_BITS = 21
KEY_BIAS = 1 << (KEY_BITS - 1)
KEY_MASK = (1 << KEY_BITS) - 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local visibility hash-voxel filter for removing short-lived dynamic objects."
    )
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--seq", default="00")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--pose", type=Path)
    p.add_argument("--voxel-size", type=float, default=0.4)
    p.add_argument("--max-range", type=float, default=30.0)
    p.add_argument("--local-z-min", type=float, default=-2.5)
    p.add_argument("--local-z-max", type=float, default=3.0)
    p.add_argument("--body-radius", type=float, default=0.0)
    p.add_argument(
        "--ground-protect-local-z-max",
        type=float,
        help=(
            "Force-keep points whose original local z is <= this value. "
            "Useful when floor points are classified as dynamic; try 0.0 or -0.1."
        ),
    )
    p.add_argument("--lidar-hz", type=float, default=10.0)
    p.add_argument("--ray-stride", type=int, default=4, help="Trace one ray per N endpoint voxels; 1 traces all.")
    p.add_argument("--max-ray-endpoints", type=int, default=25000, help="Hard cap traced endpoint voxels per frame; 0 disables.")
    p.add_argument("--min-visible-frames", type=int, default=10)
    p.add_argument("--min-visible-time", type=float, default=2.0)
    p.add_argument("--static-min-hit-ratio", type=float, default=0.5)
    p.add_argument("--dynamic-max-hit-ratio", type=float, default=0.15)
    p.add_argument("--dynamic-max-hit-time", type=float, default=2.0)
    p.add_argument("--unknown-policy", choices=("keep", "drop"), default="keep")
    p.add_argument("--progress-interval", type=int, default=25, help="Print progress every N frames; 0 disables frame progress.")
    p.add_argument("--no-before", action="store_true")
    p.add_argument(
        "--deduplicate",
        choices=("none", "exact", "quantized"),
        default="none",
        help=(
            "Deduplicate local_hash_voxel_after.pcd only. exact removes identical float32 xyzi points; "
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
            "Spatially sort local_hash_voxel_after.pcd before writing. This can make CloudCompare navigation "
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
            "Interleave local_hash_voxel_after.pcd by spatial bins before writing. This is aimed at "
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
            "Deterministically shuffle local_hash_voxel_after.pcd before writing. This often matches original "
            "large maps better in CloudCompare interaction LOD because sampled subsets cover the whole map."
        ),
    )
    p.add_argument("--shuffle-seed", type=int, default=0, help="Random seed for --shuffle-after.")
    return p.parse_args()


def choose_pose_path(seq_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for name in ("poses_odom_base.txt", "poses_suma_optim.txt", "poses_identity.txt"):
        path = seq_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no pose file found in {seq_dir}")


def load_poses(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) == 12:
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :4] = np.asarray(vals, dtype=np.float32).reshape(3, 4)
        elif len(vals) == 16:
            mat = np.asarray(vals, dtype=np.float32).reshape(4, 4)
        else:
            raise ValueError(f"unsupported pose line {line_no} with {len(vals)} values in {path}")
        poses.append(mat)
    return poses


def load_times(seq_dir: Path, frame_count: int, lidar_hz: float) -> np.ndarray:
    path = seq_dir / "times.txt"
    if path.exists():
        times = np.asarray([float(x) for x in path.read_text().split()], dtype=np.float64)
        if times.shape[0] >= frame_count:
            return times
    return np.arange(frame_count, dtype=np.float64) / float(lidar_hz)


def pack_keys(ixyz: np.ndarray) -> np.ndarray:
    shifted = ixyz.astype(np.int64) + KEY_BIAS
    if shifted.size and ((shifted < 0).any() or (shifted > KEY_MASK).any()):
        raise ValueError(f"voxel coordinate exceeds +/-{KEY_BIAS}; increase KEY_BITS or voxel size")
    return (
        (shifted[:, 0].astype(np.uint64) << np.uint64(42))
        | (shifted[:, 1].astype(np.uint64) << np.uint64(21))
        | shifted[:, 2].astype(np.uint64)
    )


def pack_one(ix: int, iy: int, iz: int) -> int:
    sx = ix + KEY_BIAS
    sy = iy + KEY_BIAS
    sz = iz + KEY_BIAS
    if sx < 0 or sy < 0 or sz < 0 or sx > KEY_MASK or sy > KEY_MASK or sz > KEY_MASK:
        raise ValueError(f"voxel coordinate exceeds +/-{KEY_BIAS}; increase KEY_BITS or voxel size")
    return int((sx << 42) | (sy << 21) | sz)


def write_pcd_from_payload(payload_path: Path, point_count: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    total_bytes = payload_path.stat().st_size
    copied_bytes = 0
    progress_started = time.time()
    progress_step = max(1, math.ceil(total_bytes / 20))
    next_progress = progress_step
    with out_path.open("wb") as dst, payload_path.open("rb") as src:
        dst.write(header)
        while chunk := src.read(8 * 1024 * 1024):
            dst.write(chunk)
            copied_bytes += len(chunk)
            if copied_bytes >= next_progress or copied_bytes == total_bytes:
                print_progress("pcd", copied_bytes, total_bytes, progress_started, f"file={out_path.name}")
                next_progress = copied_bytes + progress_step
    if total_bytes == 0:
        print_progress("pcd", 0, 0, progress_started, f"file={out_path.name}")


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


def choose_output_dir(dataset: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    workspace = Path(__file__).resolve().parent.parent
    run_root = workspace / "run_results" / dataset.resolve().name / "local_hash_voxel_runs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_local_hash_voxel"
    out = run_root / run_id
    suffix = 1
    while out.exists():
        out = run_root / f"{run_id}_{suffix}"
        suffix += 1
    return out


def scan_to_global(
    scan_path: Path,
    pose: np.ndarray,
    body_radius: float,
    max_range: float,
    local_z_min: float,
    local_z_max: float,
    apply_local_z: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    raw_count = int(scan.shape[0])
    keep = np.isfinite(scan).all(axis=1)
    if body_radius > 0 or max_range > 0:
        dist2 = scan[:, 0].astype(np.float64) ** 2 + scan[:, 1].astype(np.float64) ** 2
        if body_radius > 0:
            keep &= dist2 >= body_radius * body_radius
        if max_range > 0:
            keep &= dist2 <= max_range * max_range
    if apply_local_z:
        keep &= scan[:, 2] >= local_z_min
        keep &= scan[:, 2] <= local_z_max
    scan = scan[keep]
    xyz = scan[:, :3] @ pose[:3, :3].T + pose[:3, 3]
    return xyz, scan[:, 3], raw_count


def ray_voxels(origin: np.ndarray, endpoint: np.ndarray, voxel_size: float) -> list[int]:
    start = np.floor(origin / voxel_size).astype(np.int64)
    end = np.floor(endpoint / voxel_size).astype(np.int64)
    current = start.copy()
    direction = endpoint - origin
    step = np.sign(direction).astype(np.int64)

    t_max = np.empty(3, dtype=np.float64)
    t_delta = np.empty(3, dtype=np.float64)
    for axis in range(3):
        if direction[axis] > 0:
            next_boundary = (current[axis] + 1) * voxel_size
            t_max[axis] = (next_boundary - origin[axis]) / direction[axis]
            t_delta[axis] = voxel_size / direction[axis]
        elif direction[axis] < 0:
            next_boundary = current[axis] * voxel_size
            t_max[axis] = (next_boundary - origin[axis]) / direction[axis]
            t_delta[axis] = -voxel_size / direction[axis]
        else:
            t_max[axis] = math.inf
            t_delta[axis] = math.inf

    out: list[int] = []
    max_steps = int(np.abs(end - start).sum()) + 1
    for _ in range(max_steps + 1):
        out.append(pack_one(int(current[0]), int(current[1]), int(current[2])))
        if np.array_equal(current, end):
            break
        axis = int(np.argmin(t_max))
        current[axis] += step[axis]
        t_max[axis] += t_delta[axis]
    return out


def update_stat(
    stats: dict[int, list[int]],
    key: int,
    visible_inc: int,
    hit_frame_inc: int,
    hit_count_inc: int,
    frame_id: int,
) -> None:
    stat = stats.get(key)
    if stat is None:
        first_hit = frame_id if hit_frame_inc else -1
        last_hit = frame_id if hit_frame_inc else -1
        stats[key] = [visible_inc, hit_frame_inc, hit_count_inc, first_hit, last_hit]
        return
    stat[0] += visible_inc
    stat[1] += hit_frame_inc
    stat[2] += hit_count_inc
    if hit_frame_inc:
        if stat[3] < 0:
            stat[3] = frame_id
        stat[4] = frame_id


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def print_progress(
    phase: str,
    done: int,
    total: int,
    phase_started: float,
    detail: str = "",
) -> None:
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, done / total))
    percent = 100.0 * fraction
    elapsed = time.time() - phase_started
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    bar_width = 30
    filled = int(bar_width * fraction)
    bar = "=" * filled + "-" * (bar_width - filled)
    suffix = f" | {detail}" if detail else ""
    line = (
        f"[{phase:<8}] [{bar}] {percent:6.2f}% "
        f"({done}/{total}) elapsed={format_duration(elapsed)} eta={format_duration(remaining)}{suffix}"
    )
    complete = total <= 0 or done >= total
    if sys.stdout.isatty():
        sys.stdout.write(f"\r\033[2K{line}")
        if complete:
            sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        print(line, flush=True)


def main() -> int:
    args = parse_args()
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be > 0")
    if args.max_range <= 0:
        raise ValueError("--max-range must be > 0")
    if args.ray_stride <= 0:
        raise ValueError("--ray-stride must be > 0")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be >= 0")
    if args.local_z_min > args.local_z_max:
        raise ValueError("--local-z-min must be <= --local-z-max")
    if args.deduplicate == "quantized" and args.dedup_resolution <= 0:
        raise ValueError("--dedup-resolution must be > 0 when --deduplicate quantized")
    if args.spatial_sort_after and args.spatial_sort_resolution < 0:
        raise ValueError("--spatial-sort-resolution must be >= 0")
    if args.interleave_after and args.interleave_resolution < 0:
        raise ValueError("--interleave-resolution must be >= 0")
    reorder_count = int(args.spatial_sort_after) + int(args.interleave_after) + int(args.shuffle_after)
    if reorder_count > 1:
        raise ValueError("--spatial-sort-after, --interleave-after, and --shuffle-after are mutually exclusive")

    started = time.time()
    args.out = choose_output_dir(args.dataset, args.out)
    seq_dir = args.dataset / "dataset" / "sequences" / args.seq
    velodyne_dir = seq_dir / "velodyne"
    scan_paths = sorted(velodyne_dir.glob("*.bin"))
    if not scan_paths:
        raise FileNotFoundError(f"no .bin scans found in {velodyne_dir}")

    pose_path = choose_pose_path(seq_dir, args.pose)
    poses = load_poses(pose_path)
    times = load_times(seq_dir, len(scan_paths), args.lidar_hz)
    end = args.end if args.end is not None else len(scan_paths) - 1
    if args.start < 0 or end >= len(scan_paths) or args.start > end:
        raise ValueError(f"invalid frame range {args.start}..{end}; frame count is {len(scan_paths)}")
    if len(poses) <= end:
        raise ValueError(f"pose count {len(poses)} is smaller than end frame {end}")
    args.out.mkdir(parents=True, exist_ok=True)
    total_frames = end - args.start + 1
    print(
        f"[local-hash] dataset={args.dataset}, seq={args.seq}, frames={args.start}..{end} "
        f"({total_frames} frames), pose={pose_path}, out={args.out}",
        flush=True,
    )

    stats: dict[int, list[int]] = {}
    total_raw_points = 0
    total_roi_points = 0
    total_endpoint_voxels = 0
    total_traced_rays = 0
    total_visible_updates = 0

    stats_started = time.time()
    for frame_id in range(args.start, end + 1):
        done_frames = frame_id - args.start + 1
        xyz, _intensity, raw_count = scan_to_global(
            scan_paths[frame_id],
            poses[frame_id],
            args.body_radius,
            args.max_range,
            args.local_z_min,
            args.local_z_max,
            True,
        )
        total_raw_points += raw_count
        total_roi_points += int(xyz.shape[0])
        if xyz.size == 0:
            if args.progress_interval and (
                done_frames == 1 or done_frames % args.progress_interval == 0 or frame_id == end
            ):
                print_progress(
                    "stats",
                    done_frames,
                    total_frames,
                    stats_started,
                    f"frame={frame_id}, stats_voxels={len(stats)}, "
                    f"roi_points={total_roi_points}, traced_rays={total_traced_rays}",
                )
            continue

        coords = np.floor(xyz / args.voxel_size).astype(np.int64)
        point_keys = pack_keys(coords)
        unique_keys, first_idx, counts = np.unique(point_keys, return_index=True, return_counts=True)
        total_endpoint_voxels += int(unique_keys.shape[0])

        origin = poses[frame_id][:3, 3].astype(np.float64)
        ray_indices = np.arange(0, unique_keys.shape[0], args.ray_stride, dtype=np.int64)
        if args.max_ray_endpoints > 0 and ray_indices.shape[0] > args.max_ray_endpoints:
            ray_indices = ray_indices[: args.max_ray_endpoints]

        visible_keys: set[int] = set()
        for idx in ray_indices:
            endpoint = xyz[first_idx[int(idx)]].astype(np.float64)
            visible_keys.update(ray_voxels(origin, endpoint, args.voxel_size))
        total_traced_rays += int(ray_indices.shape[0])
        total_visible_updates += len(visible_keys)

        hit_keys = {int(k): int(c) for k, c in zip(unique_keys, counts)}
        for key in visible_keys:
            update_stat(stats, key, 1, 0, 0, frame_id)
        for key, count in hit_keys.items():
            update_stat(stats, key, 0, 1, count, frame_id)

        if args.progress_interval and (
            done_frames == 1 or done_frames % args.progress_interval == 0 or frame_id == end
        ):
            print_progress(
                "stats",
                done_frames,
                total_frames,
                stats_started,
                f"frame={frame_id}, stats_voxels={len(stats)}, "
                f"roi_points={total_roi_points}, traced_rays={total_traced_rays}",
            )

    print("[classify] classifying voxels", flush=True)
    static_keys: set[int] = set()
    dynamic_keys: set[int] = set()
    unknown_keys: set[int] = set()
    hit_voxels = 0
    classify_started = time.time()
    total_stats_voxels = len(stats)
    classify_progress_step = max(1, math.ceil(total_stats_voxels / 20))
    next_classify_progress = classify_progress_step
    for done_voxels, (key, stat) in enumerate(stats.items(), 1):
        visible_frames, hit_frames, hit_count, first_hit, last_hit = stat
        if hit_frames > 0:
            hit_voxels += 1
            visible_time = max(0.0, visible_frames / float(args.lidar_hz))
            hit_time = max(0.0, float(times[last_hit] - times[first_hit])) if first_hit >= 0 else 0.0
            hit_ratio = hit_frames / max(1, visible_frames)
            enough_visible = visible_frames >= args.min_visible_frames and visible_time >= args.min_visible_time
            if enough_visible and hit_ratio >= args.static_min_hit_ratio:
                static_keys.add(key)
            elif enough_visible and hit_ratio <= args.dynamic_max_hit_ratio and hit_time <= args.dynamic_max_hit_time:
                dynamic_keys.add(key)
            else:
                unknown_keys.add(key)

        if done_voxels >= next_classify_progress or done_voxels == total_stats_voxels:
            print_progress(
                "classify",
                done_voxels,
                total_stats_voxels,
                classify_started,
                f"static={len(static_keys)}, dynamic={len(dynamic_keys)}, unknown={len(unknown_keys)}",
            )
            next_classify_progress = done_voxels + classify_progress_step
    if total_stats_voxels == 0:
        print_progress("classify", 0, 0, classify_started)

    kept_keys = set(static_keys)
    if args.unknown_policy == "keep":
        kept_keys.update(unknown_keys)

    tmp_dir = Path(tempfile.mkdtemp(prefix="local_hash_voxel_", dir=str(args.out)))
    before_payload = tmp_dir / "before_points.bin"
    static_payload = tmp_dir / "static_points.bin"
    dynamic_payload = tmp_dir / "dynamic_points.bin"
    before_count = 0
    static_count = 0
    static_count_before_dedup = 0
    deduplicated_after_points = 0
    dynamic_count = 0
    spatial_sort_resolution = args.spatial_sort_resolution or args.voxel_size
    interleave_resolution = args.interleave_resolution or args.voxel_size
    try:
        before_file = before_payload.open("wb") if not args.no_before else None
        dedup_seen: set = set()
        write_started = time.time()
        with static_payload.open("wb") as static_file, dynamic_payload.open("wb") as dynamic_file:
            try:
                for frame_id in range(args.start, end + 1):
                    done_frames = frame_id - args.start + 1
                    scan = np.fromfile(scan_paths[frame_id], dtype=np.float32).reshape(-1, 4)
                    keep = np.isfinite(scan).all(axis=1)
                    if args.body_radius > 0 or args.max_range > 0:
                        dist2 = scan[:, 0].astype(np.float64) ** 2 + scan[:, 1].astype(np.float64) ** 2
                        if args.body_radius > 0:
                            keep &= dist2 >= args.body_radius * args.body_radius
                        if args.max_range > 0:
                            keep &= dist2 <= args.max_range * args.max_range
                    scan = scan[keep]
                    if scan.size == 0:
                        if args.progress_interval and (
                            done_frames == 1 or done_frames % args.progress_interval == 0 or frame_id == end
                        ):
                            print_progress(
                                "write",
                                done_frames,
                                total_frames,
                                write_started,
                                f"frame={frame_id}, static_points={static_count}, dynamic_points={dynamic_count}, "
                                f"deduped={deduplicated_after_points}",
                            )
                        continue
                    local_z = scan[:, 2]
                    xyz = scan[:, :3] @ poses[frame_id][:3, :3].T + poses[frame_id][:3, 3]
                    intensity = scan[:, 3]
                    point_keys = pack_keys(np.floor(xyz / args.voxel_size).astype(np.int64))
                    in_dynamic_roi = (local_z >= args.local_z_min) & (local_z <= args.local_z_max)
                    classified_static = np.fromiter((int(k) in kept_keys for k in point_keys), dtype=bool, count=point_keys.shape[0])
                    ground_protected = (
                        local_z <= args.ground_protect_local_z_max
                        if args.ground_protect_local_z_max is not None
                        else np.zeros(local_z.shape[0], dtype=bool)
                    )
                    is_static = (~in_dynamic_roi) | classified_static | ground_protected
                    points = np.column_stack((xyz, intensity)).astype(np.float32, copy=False)

                    if before_file is not None:
                        points.tofile(before_file)
                        before_count += int(points.shape[0])

                    static_points = points[is_static]
                    static_count_before_dedup += int(static_points.shape[0])
                    dedup_keep = dedup_mask(static_points, args.deduplicate, args.dedup_resolution, dedup_seen)
                    static_points = static_points[dedup_keep]
                    if static_points.size:
                        static_points.tofile(static_file)
                    points[~is_static].tofile(dynamic_file)
                    static_count += int(static_points.shape[0])
                    deduplicated_after_points = static_count_before_dedup - static_count
                    dynamic_count += int((~is_static).sum())

                    if args.progress_interval and (
                        done_frames == 1 or done_frames % args.progress_interval == 0 or frame_id == end
                    ):
                        print_progress(
                            "write",
                            done_frames,
                            total_frames,
                            write_started,
                            f"frame={frame_id}, static_points={static_count}, dynamic_points={dynamic_count}, "
                            f"deduped={deduplicated_after_points}",
                        )
            finally:
                if before_file is not None:
                    before_file.close()

        print("[pcd] writing output PCD files", flush=True)
        if not args.no_before:
            write_pcd_from_payload(before_payload, before_count, args.out / "local_hash_voxel_before.pcd")
        static_payload_to_write = static_payload
        if args.spatial_sort_after:
            sort_started = time.time()
            sorted_static_payload = tmp_dir / "static_points_spatial_sorted.bin"
            print(
                f"[sort    ] spatially sorting after cloud at resolution={spatial_sort_resolution:g} "
                f"points={static_count}",
                flush=True,
            )
            spatially_sort_payload(
                static_payload,
                static_count,
                spatial_sort_resolution,
                sorted_static_payload,
            )
            print(f"[sort    ] done elapsed={format_duration(time.time() - sort_started)}", flush=True)
            static_payload_to_write = sorted_static_payload
        if args.interleave_after:
            interleave_started = time.time()
            interleaved_static_payload = tmp_dir / "static_points_interleaved.bin"
            print(
                f"[interleave] interleaving after cloud at resolution={interleave_resolution:g} "
                f"points={static_count}",
                flush=True,
            )
            interleave_payload_by_spatial_bins(
                static_payload,
                static_count,
                interleave_resolution,
                interleaved_static_payload,
            )
            print(f"[interleave] done elapsed={format_duration(time.time() - interleave_started)}", flush=True)
            static_payload_to_write = interleaved_static_payload
        if args.shuffle_after:
            shuffle_started = time.time()
            shuffled_static_payload = tmp_dir / "static_points_shuffled.bin"
            print(
                f"[shuffle ] shuffling after cloud seed={args.shuffle_seed} points={static_count}",
                flush=True,
            )
            shuffle_payload(static_payload, static_count, args.shuffle_seed, shuffled_static_payload)
            print(f"[shuffle ] done elapsed={format_duration(time.time() - shuffle_started)}", flush=True)
            static_payload_to_write = shuffled_static_payload
        write_pcd_from_payload(static_payload_to_write, static_count, args.out / "local_hash_voxel_after.pcd")
        write_pcd_from_payload(dynamic_payload, dynamic_count, args.out / "local_hash_voxel_dynamic.pcd")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - started
    summary = args.out / "point_count_summary.txt"
    summary.write_text(
        "\n".join(
            [
                "tool: local_hash_voxel_filter",
                f"dataset: {args.dataset}",
                f"sequence: {args.seq}",
                f"pose: {pose_path}",
                f"frames: {args.start}..{end}",
                f"voxel_size: {args.voxel_size}",
                f"max_range: {args.max_range}",
                f"local_z_min: {args.local_z_min}",
                f"local_z_max: {args.local_z_max}",
                f"ground_protect_local_z_max: {args.ground_protect_local_z_max}",
                f"ray_stride: {args.ray_stride}",
                f"max_ray_endpoints: {args.max_ray_endpoints}",
                f"min_visible_frames: {args.min_visible_frames}",
                f"min_visible_time: {args.min_visible_time}",
                f"static_min_hit_ratio: {args.static_min_hit_ratio}",
                f"dynamic_max_hit_ratio: {args.dynamic_max_hit_ratio}",
                f"dynamic_max_hit_time: {args.dynamic_max_hit_time}",
                f"unknown_policy: {args.unknown_policy}",
                f"deduplicate: {args.deduplicate}",
                f"dedup_resolution: {args.dedup_resolution}",
                f"spatial_sort_after: {args.spatial_sort_after}",
                f"spatial_sort_resolution: {spatial_sort_resolution}",
                f"interleave_after: {args.interleave_after}",
                f"interleave_resolution: {interleave_resolution}",
                f"shuffle_after: {args.shuffle_after}",
                f"shuffle_seed: {args.shuffle_seed}",
                f"raw_points: {total_raw_points}",
                f"roi_points: {total_roi_points}",
                f"endpoint_voxels: {total_endpoint_voxels}",
                f"traced_rays: {total_traced_rays}",
                f"visible_frame_updates: {total_visible_updates}",
                f"stats_voxels: {len(stats)}",
                f"hit_voxels: {hit_voxels}",
                f"static_voxels: {len(static_keys)}",
                f"dynamic_voxels: {len(dynamic_keys)}",
                f"unknown_voxels: {len(unknown_keys)}",
                f"before_points: {before_count}",
                f"static_points_before_dedup: {static_count_before_dedup}",
                f"static_points: {static_count}",
                f"deduplicated_after_points: {deduplicated_after_points}",
                f"dynamic_points: {dynamic_count}",
                f"elapsed_sec: {elapsed:.3f}",
            ]
        )
        + "\n"
    )
    print(f"Output: {args.out}")
    print(summary.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
