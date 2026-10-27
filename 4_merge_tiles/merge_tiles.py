"""Merge manually corrected tiles into one labelled floor.

Joins all files saved by the Point Label Corrector into one file. The corrected files
are only read (checksums verified before and after); labels are copied unchanged.

Usage:  python merge_tiles.py                         (corrected files in the folder 'corrected')
        python merge_tiles.py --input-folder D:/my_tiles
Output: merged/floor_merged.ply  (x y z red green blue label conf idx tile)
        label 0-11 = classes, 255 = unknown; tile = index into the file list in the header comment.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def ensure_packages() -> None:
    """Install numpy on the first run if it is missing."""
    import importlib.util
    import subprocess

    if importlib.util.find_spec("numpy") is None:
        print("Missing Python library: numpy. Installing it now (one time only)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        importlib.invalidate_caches()


ensure_packages()

import numpy as np

ROOT = Path(__file__).resolve().parent
CLASSES = ["floor", "ceiling", "wall", "door", "window", "column", "desk", "chair", "monitor",
           "partition", "cabinet_shelf", "clutter"]
UNK = 255
OUT_DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                   ("label", "u1"), ("conf", "<f4"), ("idx", "<i4"), ("tile", "u1")])
PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "<i2", "int16": "<i2",
             "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
             "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def md5(fp):
    h = hashlib.md5()
    with open(fp, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def read_labelled(fp: Path):
    with open(fp, "rb") as f:
        props, n, inv, fmt = [], 0, False, ""
        while True:
            line = f.readline()
            if not line:
                raise SystemExit(f"{fp.name}: not a valid PLY file.")
            s = line.decode("ascii", "ignore").strip()
            if s.startswith("format"):
                fmt = s.split()[1]
            elif s.startswith("element"):
                inv = s.split()[1] == "vertex"
                n = int(s.split()[2]) if inv else n
            elif s.startswith("property") and inv:
                props.append((s.split()[2], PLY_TYPES[s.split()[1]]))
            elif s == "end_header":
                break
        if fmt != "binary_little_endian":
            raise SystemExit(f"{fp.name}: only binary PLY files are supported.")
        a = np.fromfile(f, dtype=np.dtype(props), count=n)
    if len(a) != n:
        raise SystemExit(f"{fp.name}: the file is incomplete ({len(a):,} of {n:,} points).")
    for k in ("x", "y", "z", "red", "green", "blue", "label"):
        if k not in a.dtype.names:
            raise SystemExit(f"{fp.name}: property '{k}' is missing; is this a file saved by the Point Label Corrector?")
    return a


def main():
    ap = argparse.ArgumentParser(description="Merge manually corrected tiles into one file.")
    ap.add_argument("--input-folder", default="corrected",
                    help="folder with the corrected .ply files (default: 'corrected' next to this script)")
    ap.add_argument("--output-folder", default="merged", help="folder for the result (default: merged)")
    args = ap.parse_args()
    src = Path(args.input_folder)
    src = src if src.is_absolute() else ROOT / src
    out = Path(args.output_folder)
    out = out if out.is_absolute() else ROOT / out
    if not src.is_dir():
        raise SystemExit(f"Folder not found: {src}\nCreate it and put the corrected files in it, "
                         "or name the folder with --input-folder.")
    files = sorted(src.glob("*.ply"))
    if not files:
        raise SystemExit(f"No .ply files in {src}.")
    if len(files) > 255:
        raise SystemExit("More than 255 files; use larger tiles in label_transfer.py.")
    out.mkdir(parents=True, exist_ok=True)

    before = {fp: md5(fp) for fp in files}
    parts = []
    for t, fp in enumerate(files):
        a = read_labelled(fp)
        b = np.zeros(len(a), dtype=OUT_DT)
        for k in ("x", "y", "z", "red", "green", "blue", "label"):
            b[k] = a[k]
        b["conf"] = a["conf"] if "conf" in a.dtype.names else 1.0
        b["idx"] = a["idx"] if "idx" in a.dtype.names else -1
        b["tile"] = t
        parts.append(b)
        print(f"[{t:2d}] {fp.name}: {len(a):,} points, unknown {100 * (a['label'] == UNK).mean():.1f}%")
    m = np.concatenate(parts)

    # each point should come from exactly one tile
    xyz = np.ascontiguousarray(np.c_[m["x"], m["y"], m["z"]]).view([("", "<f4", 3)]).ravel()
    dup = len(m) - len(np.unique(xyz))
    print(f"\npoints that appear more than once (same position): {dup:,}"
          + ("" if dup == 0 else "  <- check that no tile was copied twice"))

    dst = out / "floor_merged.ply"
    with open(dst, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 + "comment tiles: " + " ".join(f"{i}={p.stem}" for i, p in enumerate(files)) + "\n"
                 + "comment label 0-11: " + " ".join(f"{i}={c}" for i, c in enumerate(CLASSES)) + " ; 255=unknown\n"
                 + f"element vertex {len(m)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "property uchar label\nproperty float conf\nproperty int idx\nproperty uchar tile\nend_header\n").encode())
        m.tofile(f)

    # verify: corrected files untouched and labels copied unchanged
    assert before == {fp: md5(fp) for fp in files}, "a corrected file changed!"
    assert np.array_equal(np.concatenate([read_labelled(fp)["label"] for fp in files]), m["label"])
    cnt = np.bincount(m["label"], minlength=256)
    print(f"merged: {len(m):,} points from {len(files)} files -> {dst}")
    print("labels copied unchanged: yes; corrected files unchanged (MD5 before = after): yes")
    print("class shares: " + ", ".join(f"{c} {100 * cnt[i] / len(m):.1f}%" for i, c in enumerate(CLASSES))
          + f", unknown {100 * cnt[UNK] / len(m):.1f}%")


if __name__ == "__main__":
    main()
