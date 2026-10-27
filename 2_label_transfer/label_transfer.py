"""Starting labels and tiling for manual point labelling.

Gives every point of a cleaned point cloud a starting label, using a model that was
already trained on a manually labelled office floor (no training is needed here). Then writes the
labelled cloud either as one file or cut into square tiles, ready for manual correction in the Point Label Corrector.

Steps: (a) read the cleaned point cloud, (b) group the points into 5 cm cubes and describe the shape
and colour around each cube, (c) predict a label and a confidence (0-1) with the trained model,
(d) give every point the label of its cube, (e) write one file or tiles, a tile map and a report.

Set the tile size below (TILE_SIZE_M), or give it when starting the script with --tile-size.
Usage:  python label_transfer.py                    (uses TILE_SIZE_M below)
        python label_transfer.py --tile-size 0      (no tiling: one file for the whole floor)
Input: preprocessed.ply next to this script or in ../1_preprocessing, or a file named with --input.
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path


def ensure_packages() -> None:
    """Install any missing third-party library on the first run (from requirements.txt when present)."""
    import importlib.util
    import subprocess

    needed = {"numpy": "numpy", "scipy": "scipy", "lightgbm": "lightgbm"}
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


if __name__ == "__main__":
    ensure_packages()

# LightGBM optionally imports dask, which breaks with NumPy 2 in some Anaconda installations; not needed here.
sys.modules.setdefault("dask", None)

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "label_model.txt.gz"
MODEL_INFO_PATH = ROOT / "model" / "model_info.json"

# ============================================================================
# TILE SIZE (metres) - SET THIS NUMBER BEFORE RUNNING
#   8   = cut the floor into 8 m x 8 m tiles (one file per tile, easier to correct one at a time)
#   0   = no tiling: one labelled file for the whole floor
# Any other size can be used, e.g. 5 or 10. A value given with --tile-size overrides this one.
TILE_SIZE_M = 8
# ============================================================================

# Fixed settings: they must match the trained model (see model/model_info.json). Do not change.
FEATURE_VOXEL_M = 0.05          # labels are predicted for 5 cm cubes
NEIGHBOURS = (10, 30, 60)       # neighbourhood sizes used to describe the local shape
OUTSIDE_MARGIN_M = 0.3          # points this far below the floor / above the ceiling -> clutter
LOW_CONFIDENCE = 0.5            # reported share of points below this confidence
MIN_TILE_POINTS = 1000          # tiles with fewer points (stray edge points) are joined to a neighbour

CLASSES = ["floor", "ceiling", "wall", "door", "window", "column", "desk", "chair", "monitor",
           "partition", "cabinet_shelf", "clutter"]
CLUTTER = CLASSES.index("clutter")
# Same colours as the Point Label Corrector (used for the tile map)
COLOURS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
           "#9467bd", "#c5b0d5", "#8c564b", "#c49c94"]

OUTPUT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                         ("blue", "u1"), ("label", "u1"), ("conf", "<f4"), ("idx", "<i4")])
PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "<i2", "int16": "<i2",
             "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
             "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


# ---------------------------------------------------------------------------- reading / writing
def read_ply(path: Path):
    """Read x, y, z and colour from a binary little-endian PLY (as written by Open3D or CloudCompare)."""
    with open(path, "rb") as f:
        n, props, in_vertex, fmt = 0, [], False, ""
        while True:
            line = f.readline()
            if not line:
                raise SystemExit(f"{path.name}: not a valid PLY file (no end_header).")
            s = line.decode("ascii", "ignore").strip()
            if s.startswith("format"):
                fmt = s.split()[1]
            elif s.startswith("element"):
                in_vertex = s.split()[1] == "vertex"
                if in_vertex:
                    n = int(s.split()[2])
            elif s.startswith("property") and in_vertex:
                parts = s.split()
                if parts[1] == "list":
                    raise SystemExit(f"{path.name}: list properties in the vertex element are not supported.")
                props.append((parts[2], PLY_TYPES[parts[1]]))
            elif s == "end_header":
                break
        if fmt != "binary_little_endian":
            raise SystemExit(f"{path.name}: only binary PLY files are supported. In CloudCompare, save as "
                             "PLY with 'Binary' selected.")
        a = np.fromfile(f, dtype=np.dtype(props), count=n)
    names = a.dtype.names
    for k in ("x", "y", "z"):
        if k not in names:
            raise SystemExit(f"{path.name}: property '{k}' is missing.")
    xyz = np.c_[a["x"], a["y"], a["z"]].astype(np.float64)
    rgb_names = [k for k in (("red", "green", "blue"), ("r", "g", "b")) if all(c in names for c in k)]
    if not rgb_names:
        raise SystemExit(f"{path.name}: the point cloud has no colours (red/green/blue). Colours are needed.")
    rgb = np.c_[[a[c] for c in rgb_names[0]]].T
    rgb = (np.clip(rgb * 255.0, 0, 255) if rgb.dtype.kind == "f" and rgb.max() <= 1.0 else rgb).astype(np.uint8)
    return xyz, rgb


def write_labelled_ply(path: Path, xyz, rgb, label, conf, idx) -> None:
    a = np.empty(len(idx), dtype=OUTPUT_DTYPE)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    a["label"], a["conf"], a["idx"] = label, conf, idx
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"comment starting labels from label_transfer.py; classes: {' '.join(CLASSES)}\n"
              f"element vertex {len(a)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property uchar label\nproperty float conf\nproperty int idx\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        a.tofile(f)


def write_png(path: Path, img) -> None:
    """Write an RGB uint8 image as PNG using only the standard library."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


# ---------------------------------------------------------------------------- features (shared with training)
def storey_levels(z):
    """Floor and ceiling height from the two strongest peaks of the height histogram."""
    h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.02))
    c = (e[:-1] + e[1:]) / 2
    low = c < np.median(z)
    floor_z = c[low][np.argmax(h[low])]
    high = c > floor_z + 1.5
    if not high.any():
        return float(floor_z), float(z.max())
    return float(floor_z), float(c[high & (h > 0.3 * h[high].max())].max())


def voxel_representatives(xyz, voxel):
    """Index of the first point in every occupied cube, and the cube of every point."""
    k = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    return first, inverse.ravel()


def compute_features(xyz, rgb, chunk=100_000, log=print):
    """Per point: standardised colour, shape descriptors at 3 neighbourhood sizes, height above floor,
    depth below ceiling and neighbourhood radius (27 values)."""
    tree = cKDTree(xyz)
    n, kmax = len(xyz), max(NEIGHBOURS)
    shape = np.empty((n, 7 * len(NEIGHBOURS)), np.float32)
    radius = np.empty(n, np.float32)
    t0 = time.time()
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d, nn = tree.query(xyz[s:e], k=kmax, workers=-1)
        radius[s:e] = d[:, -1]
        for j, k in enumerate(NEIGHBOURS):
            P = xyz[nn[:, :k]]
            P = P - P.mean(1, keepdims=True)
            C = np.einsum("nki,nkj->nij", P, P) / k
            w, v = np.linalg.eigh(C)
            w = np.clip(w, 1e-12, None)
            l3, l2, l1 = w[:, 0], w[:, 1], w[:, 2]
            normal = v[:, :, 0]
            shape[s:e, 7 * j:7 * j + 7] = np.c_[(l1 - l2) / l1, (l2 - l3) / l1, l3 / l1,
                                               np.cbrt(l1 * l2 * l3) / l1, l3 / (l1 + l2 + l3),
                                               1 - np.abs(normal[:, 2]), np.sqrt(l1)]
        if (s // chunk) % 5 == 4 or e == n:
            log(f"    described {e:,} of {n:,} cubes ({time.time() - t0:.0f} s)")
    floor_z, ceil_z = storey_levels(xyz[:, 2])
    rgb01 = rgb.astype(np.float32) / 255.0
    rgb_std = (rgb01 - rgb01.mean(0)) / (rgb01.std(0) + 1e-6)
    heights = np.c_[xyz[:, 2] - floor_z, ceil_z - xyz[:, 2]].astype(np.float32)
    return np.c_[rgb_std, shape, heights, radius], (floor_z, ceil_z)


def load_model():
    import lightgbm as lgb

    if not MODEL_PATH.exists():
        raise SystemExit(f"Trained model not found: {MODEL_PATH}\nDownload the complete tool folder, "
                         "including the 'model' folder.")
    with gzip.open(MODEL_PATH, "rt", encoding="utf-8") as f:
        booster = lgb.Booster(model_str=f.read())
    info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8")) if MODEL_INFO_PATH.exists() else {}
    return booster, info


# ---------------------------------------------------------------------------- tiling
def tile_assignment(xyz, size):
    """Row/column of every point on a square grid; stray edge points are clipped into the edge tiles."""
    lo = np.percentile(xyz[:, :2], 0.5, axis=0) - 0.5
    hi = np.percentile(xyz[:, :2], 99.5, axis=0) + 0.5
    ncol = max(1, int(np.ceil((hi[0] - lo[0]) / size)))
    nrow = max(1, int(np.ceil((hi[1] - lo[1]) / size)))
    col = np.clip(((xyz[:, 0] - lo[0]) // size).astype(int), 0, ncol - 1)
    row = np.clip(((xyz[:, 1] - lo[1]) // size).astype(int), 0, nrow - 1)
    return row, col, nrow, ncol, lo


def draw_tile_map(path: Path, xyz, label, row, col, nrow, ncol, lo, size, kept):
    """Top view coloured by starting label, with the tile grid; kept tiles are outlined in white."""
    px = 0.05
    W = int(np.ceil(ncol * size / px)) + 1
    H = int(np.ceil(nrow * size / px)) + 1
    img = np.full((H, W, 3), 20, np.uint8)
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in COLOURS], np.uint8)
    keep = label != CLASSES.index("ceiling")
    order = np.argsort(xyz[keep, 2])                                  # higher points drawn last
    u = np.clip(((xyz[keep, 0] - lo[0]) / px).astype(int), 0, W - 1)[order]
    v = np.clip(((xyz[keep, 1] - lo[1]) / px).astype(int), 0, H - 1)[order]
    img[H - 1 - v, u] = pal[label[keep][order]]
    for r in range(nrow + 1):
        y = H - 1 - min(H - 1, int(r * size / px))
        img[max(0, y - 1):y + 1, :] = (90, 90, 90)
    for c in range(ncol + 1):
        x = min(W - 1, int(c * size / px))
        img[:, max(0, x - 1):x + 1] = (90, 90, 90)
    for (r, c) in kept:
        x0, x1 = int(c * size / px), min(W - 1, int((c + 1) * size / px))
        y1, y0 = H - 1 - int(r * size / px), max(0, H - 1 - int((r + 1) * size / px))
        img[y0:y0 + 3, x0:x1] = img[max(0, y1 - 3):y1, x0:x1] = 255
        img[y0:y1, x0:x0 + 3] = img[y0:y1, max(0, x1 - 3):x1] = 255
    if max(H, W) > 4000:                                               # keep the image a sensible size
        step = int(np.ceil(max(H, W) / 4000))
        img = img[::step, ::step]
    write_png(path, np.ascontiguousarray(img))


# ---------------------------------------------------------------------------- main
def find_input(arg):
    if arg:
        p = Path(arg)
        p = p if p.is_absolute() else ROOT / p
        if not p.exists():
            raise SystemExit(f"Input file not found: {p}")
        return p
    for default in (ROOT / "preprocessed.ply", ROOT.parent / "1_preprocessing" / "preprocessed.ply"):
        if default.exists():
            return default
    cands = [p for p in ROOT.glob("*.ply")
             if not p.name.startswith(("labelled", "tile_")) and p.name != "original.ply"]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise SystemExit("No point cloud found. Copy preprocessed.ply into this folder, "
                         "or name the file with --input.")
    raise SystemExit("More than one .ply file in this folder: " + ", ".join(p.name for p in cands) +
                     "\nName the one to use, e.g.  python label_transfer.py --input " + cands[0].name)


def main():
    ap = argparse.ArgumentParser(description="Give starting labels to a cleaned point cloud and cut it into tiles.")
    ap.add_argument("--input", help="cleaned point cloud (.ply); default: preprocessed.ply in this folder")
    ap.add_argument("--tile-size", type=float, default=None,
                    help="tile size in metres for this run (0 = no tiling); default: TILE_SIZE_M")
    ap.add_argument("--output-folder", default="labelled", help="folder for the results (default: labelled)")
    args = ap.parse_args()
    tile = TILE_SIZE_M if args.tile_size is None else args.tile_size
    if tile < 0:
        raise SystemExit("The tile size must be 0 (no tiling) or a positive number of metres.")

    src = find_input(args.input)
    out = ROOT / args.output_folder
    out.mkdir(exist_ok=True)
    log_lines = []

    def log(msg=""):
        print(msg, flush=True)
        log_lines.append(msg)

    t_all = time.time()
    log(f"Label transfer - {datetime.now():%Y-%m-%d %H:%M}")
    log(f"Input: {src.name}")

    t = time.time()
    xyz, rgb = read_ply(src)
    log(f"[Step a] Read point cloud  |  {len(xyz):,} points  ({time.time() - t:.0f} s)")

    t = time.time()
    first, inverse = voxel_representatives(xyz, FEATURE_VOXEL_M)
    log(f"[Step b] Group into {FEATURE_VOXEL_M * 100:.0f} cm cubes  |  {len(first):,} cubes; describing shape "
        "and colour around each cube")
    X, (floor_z, ceil_z) = compute_features(xyz[first], rgb[first], log=log)
    log(f"    floor level {floor_z:.2f} m, ceiling level {ceil_z:.2f} m  ({time.time() - t:.0f} s)")

    t = time.time()
    booster, info = load_model()
    proba = booster.predict(X)
    v_label = proba.argmax(1).astype(np.uint8)
    v_conf = proba.max(1).astype(np.float32)
    zr = xyz[first, 2]
    outside = (zr < floor_z - OUTSIDE_MARGIN_M) | (zr > ceil_z + OUTSIDE_MARGIN_M)
    v_label[outside], v_conf[outside] = CLUTTER, 1.0
    log(f"[Step c] Predict starting labels with the trained model  |  model: "
        f"{info.get('trained_on', MODEL_PATH.name)}  ({time.time() - t:.0f} s)")

    label, conf = v_label[inverse], v_conf[inverse]
    idx = np.arange(len(xyz), dtype=np.int32)
    log(f"[Step d] Give every point the label of its cube  |  {len(xyz):,} points labelled")
    log("")
    log(f"    {'class':15s}{'share':>8s}{'mean confidence':>18s}")
    cnt = np.bincount(label, minlength=len(CLASSES))
    for i, c in enumerate(CLASSES):
        mc = conf[label == i].mean() if cnt[i] else float("nan")
        log(f"    {c:15s}{100 * cnt[i] / len(label):7.1f}%{mc:18.2f}")
    log(f"    points with confidence below {LOW_CONFIDENCE}: {100 * (conf < LOW_CONFIDENCE).mean():.1f}%")
    log("")

    t = time.time()
    written = []
    if tile == 0:
        p = out / "labelled_floor.ply"
        write_labelled_ply(p, xyz, rgb, label, conf, idx)
        written.append((p.name, len(xyz)))
        log(f"[Step e] No tiling (tile size 0)  |  wrote {p.name}")
    else:
        row, col, nrow, ncol, lo = tile_assignment(xyz, tile)
        tid = row * ncol + col
        counts = np.bincount(tid, minlength=nrow * ncol)
        small = [t_ for t_ in np.nonzero(counts)[0] if counts[t_] < MIN_TILE_POINTS]
        big = [t_ for t_ in np.nonzero(counts)[0] if counts[t_] >= MIN_TILE_POINTS]
        if small and big:                                           # join stray edge points to the nearest tile
            br, bc = np.array(big) // ncol, np.array(big) % ncol
            for s_ in small:
                j = np.argmin((br - s_ // ncol) ** 2 + (bc - s_ % ncol) ** 2)
                tid[tid == s_] = big[j]
        order = np.argsort(tid, kind="stable")
        bounds = np.searchsorted(tid[order], np.arange(nrow * ncol + 1))
        kept = []
        for t_ in range(nrow * ncol):
            sel = order[bounds[t_]:bounds[t_ + 1]]
            if len(sel) == 0:
                continue
            r, c = divmod(t_, ncol)
            p = out / f"tile_r{r:02d}_c{c:02d}.ply"
            write_labelled_ply(p, xyz[sel], rgb[sel], label[sel], conf[sel], idx[sel])
            written.append((p.name, len(sel)))
            kept.append((r, c))
        draw_tile_map(out / "tile_map.png", xyz, label, row, col, nrow, ncol, lo, tile, kept)
        log(f"[Step e] Cut into {tile:g} m x {tile:g} m tiles  |  grid {nrow} rows x {ncol} columns, "
            f"{len(written)} tiles with points; tile map: tile_map.png")
        log("    Tile names give the row and column: row 00 is at the bottom of tile_map.png, column 00 at the left.")
    for name, n in written:
        log(f"    {name:28s}{n:>12,} points")
    log(f"    ({time.time() - t:.0f} s)")
    log("")
    log(f"Results written to: {out}")
    log(f"Total time: {(time.time() - t_all) / 60:.1f} min")
    log("Next: correct the labels in ../3_label_correction/point_label_corrector.html.")
    (out / "label_transfer_report.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
