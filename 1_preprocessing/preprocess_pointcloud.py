"""Clean and voxel-downsample a laser scan (E57) for scan-to-BIM.

Steps: (a) remove invalid / scanner-origin / duplicate records, (b) remove points far outside the building,
(c) statistical outlier removal (SOR), (d) voxel downsampling. Writes original.ply, preprocessed.ply,
scan_positions.json (scanner poses from the E57 header) and a report.

Set the voxel size below (VOXEL_SIZE_M), or give it when starting the script with --voxel-size.
Usage:  python preprocess_pointcloud.py                      (uses VOXEL_SIZE_M below)
        python preprocess_pointcloud.py --voxel-size 0.03    (overrides it for this run)
Put the .e57 scan file in the same folder as this script, or name it with --input.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def ensure_packages() -> None:
    """Install any missing third-party library on the first run (from requirements.txt when present)."""
    import importlib.util
    import subprocess
    import sys

    needed = {"open3d": "open3d", "pye57": "pye57", "numpy": "numpy", "scipy": "scipy"}
    missing = [pip_name for module, pip_name in needed.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    req = Path(__file__).resolve().parent / "requirements.txt"
    cmd = [sys.executable, "-m", "pip", "install"] + (["-r", str(req)] if req.exists() else missing)
    print(f"Missing Python libraries: {', '.join(missing)}. Installing them now (one time only)...")
    try:
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, OSError):
        raise SystemExit(
            "Automatic installation failed. Install the libraries manually with:\n"
            f"    {sys.executable} -m pip install -r requirements.txt\n"
            "and run the script again."
        )
    importlib.invalidate_caches()


ensure_packages()

import numpy as np
import open3d as o3d
import pye57
from scipy.stats import iqr


ROOT = Path(__file__).resolve().parent
ORIGINAL_PLY = ROOT / "original.ply"
PREPROCESSED_PLY = ROOT / "preprocessed.ply"
REPORT_PATH = ROOT / "preprocessing_report.txt"
SCAN_POSITIONS_JSON = ROOT / "scan_positions.json"

# ============================================================================
# VOXEL SIZE (metres) - SET THIS NUMBER BEFORE RUNNING
#   0.02 = 2 cm cubes (recommended for labelling and modelling)
#   0.03 = 3 cm cubes (smaller files)
# A value given with --voxel-size when starting the script overrides this one.
VOXEL_SIZE_M = 0.02
# ============================================================================

SAMPLE_PER_SCAN = 5_000
RANGE_IQR_MULTIPLIER = 3.0

# SOR setting for indoor scans (Ebrahimi & Czarnuch, 2021): 4 neighbours, one standard deviation.
SOR_NB_NEIGHBORS = 4
SOR_STD_RATIO = 1.0



@dataclass
class StepStat:
    name: str
    before: int
    after: int
    params: str
    justification: str

    @property
    def removed(self) -> int:
        return self.before - self.after

    @property
    def removed_pct(self) -> float:
        return (self.removed / self.before * 100.0) if self.before else 0.0


class BinaryPlyWriter:
    def __init__(self, path: Path, point_count: int) -> None:
        self.path = path
        self.point_count = point_count
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {point_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        self.handle = path.open("wb")
        self.handle.write(header.encode("ascii"))
        self.written = 0

    def write(self, points: np.ndarray, colors: np.ndarray) -> None:
        if len(points) == 0:
            return
        rec = np.empty(
            len(points),
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        rec["x"] = points[:, 0].astype(np.float32, copy=False)
        rec["y"] = points[:, 1].astype(np.float32, copy=False)
        rec["z"] = points[:, 2].astype(np.float32, copy=False)
        rec["red"] = colors[:, 0]
        rec["green"] = colors[:, 1]
        rec["blue"] = colors[:, 2]
        rec.tofile(self.handle)
        self.written += len(points)

    def close(self) -> None:
        self.handle.close()
        if self.written != self.point_count:
            raise RuntimeError(
                f"PLY vertex count mismatch for {self.path}: "
                f"header={self.point_count:,}, written={self.written:,}"
            )


def fmt_count(n: int) -> str:
    return f"{n:,}"


def print_step(label: str, name: str, before: int, after: int) -> None:
    removed = before - after
    pct = (removed / before * 100.0) if before else 0.0
    print(
        f"[{label}] {name}  |  Before: {before:,} pts  ->  "
        f"After: {after:,} pts  |  Removed: {removed:,} ({pct:.1f}%)",
        flush=True,
    )


def find_input_e57(arg: Path | None) -> Path:
    if arg is not None:
        path = rooted_path(arg)
        if not path.exists():
            raise FileNotFoundError(f"E57 input not found: {path}")
        return path
    here = sorted(ROOT.glob("*.e57"))
    if len(here) == 1:
        return here[0]
    if len(here) > 1:
        names = ", ".join(p.name for p in here)
        raise SystemExit(f"Several .e57 files found next to the script ({names}). Choose one with --input <file.e57>.")
    raise SystemExit("No .e57 file found. Put the scan file in the same folder as this script, or give it with --input <file.e57>.")


def unique_xyz_float32(
    points: np.ndarray, colors: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray | None, int]:
    if len(points) == 0:
        return points, colors, 0
    packed = np.ascontiguousarray(points).view(
        np.dtype((np.void, points.dtype.itemsize * points.shape[1]))
    ).reshape(-1)
    _, first_idx = np.unique(packed, return_index=True)
    removed = len(points) - len(first_idx)
    if removed == 0:
        return points, colors, 0
    first_idx.sort()
    points = points[first_idx]
    if colors is not None:
        colors = colors[first_idx]
    return points, colors, removed


def write_scan_positions(e57: pye57.E57, scan_count: int, e57_path: Path, out_path: Path) -> None:
    """Save every scanner position and orientation from the E57 header."""
    import json

    stations = []
    for i in range(scan_count):
        header = e57.get_header(i)
        stations.append({
            "scan": i,
            "name": f"scan{i}",
            "xyz": [float(v) for v in header.translation],
            "quat_wxyz": [float(v) for v in header.rotation],
            "points": int(header.point_count),
        })
    out_path.write_text(json.dumps({
        "source": f"{e57_path.name} (header poses only, read-only)",
        "frame": "same as original.ply and preprocessed.ply",
        "stations": stations,
    }, indent=1), encoding="utf-8")


def read_scan_arrays(
    e57: pye57.E57, scan_index: int, with_colors: bool
) -> Tuple[np.ndarray, np.ndarray]:
    data = e57.read_scan(
        scan_index,
        colors=with_colors,
        row_column=False,
        transform=True,
        ignore_missing_fields=True,
    )
    n = len(data["cartesianX"])
    points = np.empty((n, 3), dtype=np.float32)
    points[:, 0] = data["cartesianX"].astype(np.float32, copy=False)
    points[:, 1] = data["cartesianY"].astype(np.float32, copy=False)
    points[:, 2] = data["cartesianZ"].astype(np.float32, copy=False)

    if with_colors and all(k in data for k in ("colorRed", "colorGreen", "colorBlue")):
        colors = np.empty((n, 3), dtype=np.uint8)
        colors[:, 0] = data["colorRed"].astype(np.uint8, copy=False)
        colors[:, 1] = data["colorGreen"].astype(np.uint8, copy=False)
        colors[:, 2] = data["colorBlue"].astype(np.uint8, copy=False)
    else:
        colors = np.full((n, 3), 255, dtype=np.uint8)
    return points, colors


def clean_artifacts(
    points: np.ndarray, colors: np.ndarray | None
) -> Tuple[np.ndarray, np.ndarray | None, int, int]:
    before = len(points)
    valid = np.all(np.isfinite(points), axis=1) & ~np.all(points == 0.0, axis=1)
    points = points[valid]
    if colors is not None:
        colors = colors[valid]
    invalid_removed = before - len(points)
    points, colors, duplicate_removed = unique_xyz_float32(points, colors)
    return points, colors, invalid_removed, duplicate_removed


def estimate_range(sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lows = np.empty(3, dtype=np.float64)
    highs = np.empty(3, dtype=np.float64)
    for axis in range(3):
        col = sample[:, axis].astype(np.float64)
        med = float(np.median(col))
        fence = RANGE_IQR_MULTIPLIER * float(iqr(col))
        lows[axis] = med - fence
        highs[axis] = med + fence
    return lows, highs


def apply_range_filter(
    points: np.ndarray, colors: np.ndarray, lows: np.ndarray, highs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(points), dtype=bool)
    for axis in range(3):
        keep &= (points[:, axis] >= lows[axis]) & (points[:, axis] <= highs[axis])
    return points[keep], colors[keep]


def apply_sor(
    points: np.ndarray, colors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    _, indices = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO
    )
    indices = np.asarray(indices, dtype=np.int64)
    return points[indices], colors[indices]


def add_voxel_accumulates(
    voxels: Dict[int, List[float]],
    points: np.ndarray,
    colors: np.ndarray,
    lows: np.ndarray,
    voxel_size: float,
    stride_yz: int,
    stride_z: int,
) -> None:
    if len(points) == 0:
        return
    grid = np.floor((points.astype(np.float64) - lows) / voxel_size).astype(np.int64)
    keys = grid[:, 0] * stride_yz + grid[:, 1] * stride_z + grid[:, 2]
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    sx = np.bincount(inverse, weights=points[:, 0], minlength=len(unique_keys))
    sy = np.bincount(inverse, weights=points[:, 1], minlength=len(unique_keys))
    sz = np.bincount(inverse, weights=points[:, 2], minlength=len(unique_keys))
    sr = np.bincount(inverse, weights=colors[:, 0], minlength=len(unique_keys))
    sg = np.bincount(inverse, weights=colors[:, 1], minlength=len(unique_keys))
    sb = np.bincount(inverse, weights=colors[:, 2], minlength=len(unique_keys))

    for i, key in enumerate(unique_keys):
        key_int = int(key)
        existing = voxels.get(key_int)
        if existing is None:
            voxels[key_int] = [
                float(sx[i]),
                float(sy[i]),
                float(sz[i]),
                float(sr[i]),
                float(sg[i]),
                float(sb[i]),
                int(counts[i]),
            ]
        else:
            existing[0] += float(sx[i])
            existing[1] += float(sy[i])
            existing[2] += float(sz[i])
            existing[3] += float(sr[i])
            existing[4] += float(sg[i])
            existing[5] += float(sb[i])
            existing[6] += int(counts[i])


def voxels_to_point_cloud(voxels: Dict[int, List[float]]) -> o3d.geometry.PointCloud:
    n = len(voxels)
    points = np.empty((n, 3), dtype=np.float64)
    colors = np.empty((n, 3), dtype=np.float64)
    for i, acc in enumerate(voxels.values()):
        count = acc[6]
        points[i, 0] = acc[0] / count
        points[i, 1] = acc[1] / count
        points[i, 2] = acc[2] / count
        colors[i, 0] = acc[3] / count / 255.0
        colors[i, 1] = acc[4] / count / 255.0
        colors[i, 2] = acc[5] / count / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def pcd_bounds(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pcd.points)
    return pts.min(axis=0), pts.max(axis=0)


def write_report(
    report_path: Path,
    input_path: Path,
    output_paths: List[Path],
    scan_count: int,
    raw_records: int,
    original_count: int,
    preprocessed_count: int,
    original_bounds: Tuple[np.ndarray, np.ndarray],
    preprocessed_bounds: Tuple[np.ndarray, np.ndarray],
    step_stats: List[StepStat],
    range_lows: np.ndarray,
    range_highs: np.ndarray,
    voxel_note: str,
    elapsed_s: float,
) -> None:
    now = datetime.now().astimezone()
    lines: List[str] = [
        "POINT CLOUD PREPROCESSING REPORT",
        "=" * 80,
        "",
        "1. RUN",
        f"Date/time: {now.isoformat(timespec='seconds')}",
        f"Elapsed time: {elapsed_s / 60.0:.1f} minutes",
        f"Input E57: {input_path}",
        f"Scans: {scan_count}",
        f"Raw E57 records: {raw_records:,}",
        f"Valid points: {original_count:,}",
        f"Preprocessed points: {preprocessed_count:,}",
        "Output files:",
        *[f"  - {p}" for p in output_paths],
        "",
        "2. PARAMETERS",
        f"  - Range filter: median +/- {RANGE_IQR_MULTIPLIER:.1f} x IQR per axis. "
        f"Fences: X [{range_lows[0]:.3f}, {range_highs[0]:.3f}], "
        f"Y [{range_lows[1]:.3f}, {range_highs[1]:.3f}], "
        f"Z [{range_lows[2]:.3f}, {range_highs[2]:.3f}].",
        f"  - SOR: nb_neighbors={SOR_NB_NEIGHBORS}, std_ratio={SOR_STD_RATIO} (Ebrahimi & Czarnuch, 2021).",
        f"  - Voxel size: {voxel_note}",
        "",
        "3. STEPS",
    ]
    for stat in step_stats:
        lines += [
            stat.name,
            f"  Before: {stat.before:,}",
            f"  After: {stat.after:,}",
            f"  Removed: {stat.removed:,} ({stat.removed_pct:.1f}%)",
            f"  Parameters: {stat.params}",
            f"  Purpose: {stat.justification}",
        ]
    lines += [
        "",
        "4. BOUNDS (m)",
        "  Original min X/Y/Z: " + ", ".join(f"{v:.3f}" for v in original_bounds[0]),
        "  Original max X/Y/Z: " + ", ".join(f"{v:.3f}" for v in original_bounds[1]),
        "  Preprocessed min X/Y/Z: " + ", ".join(f"{v:.3f}" for v in preprocessed_bounds[0]),
        "  Preprocessed max X/Y/Z: " + ", ".join(f"{v:.3f}" for v in preprocessed_bounds[1]),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def rooted_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and voxel-downsample an E57 laser scan."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="E57 scan file. If omitted, the single .e57 file in the script folder is used.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel size in metres for this run (overrides VOXEL_SIZE_M in the script), e.g. 0.02 or 0.03.",
    )
    parser.add_argument(
        "--original-output",
        type=Path,
        default=ORIGINAL_PLY,
        help="Original cleaned binary PLY output path.",
    )
    parser.add_argument(
        "--preprocessed-output",
        type=Path,
        default=PREPROCESSED_PLY,
        help="Preprocessed binary PLY output path.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=REPORT_PATH,
        help="Text report output path.",
    )
    parser.add_argument(
        "--skip-original",
        action="store_true",
        help="Do not rewrite original.ply; useful when it already exists from a previous run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_ply = rooted_path(args.original_output)
    preprocessed_ply = rooted_path(args.preprocessed_output)
    report_path = rooted_path(args.report_output)

    start = time.time()
    print("=" * 80)
    print("POINT CLOUD PREPROCESSING FOR SCAN-TO-BIM")
    print("=" * 80)

    if args.voxel_size is not None and args.voxel_size <= 0 or args.voxel_size is None and VOXEL_SIZE_M <= 0:
        raise SystemExit("The voxel size must be a positive number of metres, e.g. 0.02.")

    print("\nLOAD E57")
    e57_path = find_input_e57(args.input)
    e57 = pye57.E57(str(e57_path))
    scan_count = e57.scan_count
    raw_records = sum(int(e57.get_header(i).point_count) for i in range(scan_count))
    print(f"Input: {e57_path}")
    print(f"Scans: {scan_count}")
    print(f"Raw E57 records: {raw_records:,}")
    write_scan_positions(e57, scan_count, e57_path, SCAN_POSITIONS_JSON)
    print(f"Scanner positions written: {SCAN_POSITIONS_JSON} ({scan_count} positions)")

    print("\nPASS 1 - COUNT CLEAN RAW POINTS AND ESTIMATE PARAMETERS")
    rng = np.random.default_rng(20260609)
    samples: List[np.ndarray] = []
    artifact_clean_count = 0
    invalid_numeric_removed = 0
    duplicate_removed = 0
    original_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    original_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for scan_idx in range(scan_count):
        points, _ = read_scan_arrays(e57, scan_idx, with_colors=False)
        points, _, invalid_removed, dup_removed = clean_artifacts(points, None)
        invalid_numeric_removed += invalid_removed
        duplicate_removed += dup_removed
        artifact_clean_count += len(points)
        if len(points):
            original_min = np.minimum(original_min, points.min(axis=0))
            original_max = np.maximum(original_max, points.max(axis=0))
            n_sample = min(SAMPLE_PER_SCAN, len(points))
            samples.append(points[rng.choice(len(points), n_sample, replace=False)])
        if (scan_idx + 1) % 10 == 0 or scan_idx + 1 == scan_count:
            print(
                f"  counted scan {scan_idx + 1}/{scan_count}: "
                f"{artifact_clean_count:,} clean raw points",
                flush=True,
            )

    sample = np.vstack(samples)
    range_lows, range_highs = estimate_range(sample)
    if args.voxel_size is not None:
        voxel_size = float(args.voxel_size)
        voxel_note = f"{voxel_size:.4f} m, given with --voxel-size."
    else:
        voxel_size = float(VOXEL_SIZE_M)
        voxel_note = f"{voxel_size:.4f} m, set as VOXEL_SIZE_M in the script."
    if voxel_size <= 0:
        raise SystemExit("The voxel size must be a positive number of metres, e.g. 0.02.")
    print_step("Step a", "Invalid / artefact point removal", raw_records, artifact_clean_count)
    print(f"  pye57 invalid-state records removed: {raw_records - artifact_clean_count - invalid_numeric_removed - duplicate_removed:,}")
    print(f"  numeric NaN/Inf/zero-origin removed: {invalid_numeric_removed:,}")
    print(f"  exact per-scan duplicate XYZ float32 removed: {duplicate_removed:,}")
    print(
        "  range fences:",
        f"X[{range_lows[0]:.3f}, {range_highs[0]:.3f}]",
        f"Y[{range_lows[1]:.3f}, {range_highs[1]:.3f}]",
        f"Z[{range_lows[2]:.3f}, {range_highs[2]:.3f}]",
    )
    print(f"  Voxel size: {voxel_size:.4f} m ({voxel_note})")

    grid_dims = np.ceil((range_highs - range_lows) / voxel_size).astype(np.int64) + 3
    stride_z = int(grid_dims[2])
    stride_yz = int(grid_dims[1] * grid_dims[2])

    print("\nPASS 2 - WRITE ORIGINAL AND PREPROCESS")
    original_writer = None
    if args.skip_original:
        print(f"  skipping original PLY rewrite; using existing/reference path {original_ply}")
    else:
        original_writer = BinaryPlyWriter(original_ply, artifact_clean_count)
    voxels: Dict[int, List[float]] = {}
    range_count = 0
    sor_count = 0

    for scan_idx in range(scan_count):
        points, colors = read_scan_arrays(e57, scan_idx, with_colors=True)
        points, colors, _, _ = clean_artifacts(points, colors)
        if original_writer is not None:
            original_writer.write(points, colors)

        points, colors = apply_range_filter(points, colors, range_lows, range_highs)
        range_count += len(points)

        points, colors = apply_sor(points, colors)
        sor_count += len(points)

        add_voxel_accumulates(
            voxels,
            points,
            colors,
            range_lows,
            voxel_size,
            stride_yz,
            stride_z,
        )
        if (scan_idx + 1) % 5 == 0 or scan_idx + 1 == scan_count:
            print(
                f"  processed scan {scan_idx + 1}/{scan_count}: "
                f"range={range_count:,}, sor={sor_count:,}, voxels={len(voxels):,}",
                flush=True,
            )

    if original_writer is not None:
        original_writer.close()
    voxel_count = len(voxels)

    print_step("Step b", "Range / exterior point removal", artifact_clean_count, range_count)
    print_step("Step c", "Statistical outlier removal (SOR)", range_count, sor_count)
    print_step("Step d", "Voxel downsampling", sor_count, voxel_count)

    pcd = voxels_to_point_cloud(voxels)
    del voxels

    print("\nEXPORT")
    o3d.io.write_point_cloud(str(preprocessed_ply), pcd, write_ascii=False)
    pre_min, pre_max = pcd_bounds(pcd)

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Original':>14} {'Preprocessed':>14}")
    print("-" * 60)
    print(f"{'Point count':<30} {artifact_clean_count:>14,} {len(pcd.points):>14,}")
    reduction = (1 - len(pcd.points) / artifact_clean_count) * 100.0
    print(f"{'Reduction':<30} {'-':>14} {reduction:>13.1f}%")
    for i, axis in enumerate(["X", "Y", "Z"]):
        original_range = f"{original_min[i]:.2f} - {original_max[i]:.2f}"
        pre_range = f"{pre_min[i]:.2f} - {pre_max[i]:.2f}"
        print(f"{'Range ' + axis + ' (m)':<30} {original_range:>14} {pre_range:>14}")
    print(f"{'Has colours':<30} {'yes':>14} {'yes' if pcd.has_colors() else 'no':>14}")
    print(f"{'Voxel size (m)':<30} {'-':>14} {voxel_size:>14.4f}")
    print("=" * 60)

    step_stats = [
        StepStat(
            name="a. Invalid / artefact point removal",
            before=raw_records,
            after=artifact_clean_count,
            params="Remove E57 invalid-state records, NaN/Inf, exact (0,0,0), and exact per-scan duplicate XYZ float32 points.",
            justification="Removes numeric corruption and scanner-origin artefacts before any geometric filtering.",
        ),
        StepStat(
            name="b. Range / exterior point removal",
            before=artifact_clean_count,
            after=range_count,
            params=f"Per-axis median +/- {RANGE_IQR_MULTIPLIER:.1f} x IQR from {len(sample):,} sampled clean points.",
            justification="Rejects distant exterior/reflection points without manual ROI selection.",
        ),
        StepStat(
            name="c. Statistical outlier removal (SOR)",
            before=range_count,
            after=sor_count,
            params=f"nb_neighbors={SOR_NB_NEIGHBORS}, std_ratio={SOR_STD_RATIO}, applied scan-by-scan after range filtering.",
            justification="Removes sparse noise points that would create false feature vectors during segmentation.",
        ),
        StepStat(
            name="d. Voxel downsampling",
            before=sor_count,
            after=voxel_count,
            params=f"voxel_size={voxel_size:.4f} m; centroids and RGB averages retained per occupied voxel.",
            justification="Produces uniform density and a manageable point count while retaining architectural geometry.",
        ),
    ]

    print("\nREPORT")
    write_report(
        report_path=report_path,
        input_path=e57_path,
        output_paths=[original_ply, preprocessed_ply, SCAN_POSITIONS_JSON, report_path],
        scan_count=scan_count,
        raw_records=raw_records,
        original_count=artifact_clean_count,
        preprocessed_count=len(pcd.points),
        original_bounds=(original_min, original_max),
        preprocessed_bounds=(pre_min, pre_max),
        step_stats=step_stats,
        range_lows=range_lows,
        range_highs=range_highs,
        voxel_note=voxel_note,
        elapsed_s=time.time() - start,
    )
    print(f"Report written: {report_path}")
    if original_writer is None:
        print(f"Original PLY not rewritten; reference path: {original_ply}")
    else:
        print(f"Original PLY written: {original_ply}")
    print(f"Preprocessed PLY written: {preprocessed_ply}")
    print(f"Elapsed: {(time.time() - start) / 60.0:.1f} minutes")


if __name__ == "__main__":
    main()
