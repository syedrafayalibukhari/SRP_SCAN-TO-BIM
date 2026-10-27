"""Generate IFC models from a labelled point cloud and the scanner positions.

Usage:
    python run_modeling.py
    python run_modeling.py --input floor_merged.ply --positions scan_positions.json --project-name "Building A, level 3"

Without arguments the inputs are looked up next to this script first and then in the output folders of the
earlier steps (../4_merge_tiles/merged/floor_merged.ply and ../1_preprocessing/scan_positions.json).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("prepare", "Estimate normals and prepare the floor"),
    ("structure", "Levels, wall and glazing faces"),
    ("rooms", "Rooms"),
    ("openings", "Doors and windows"),
    ("columns_partitions", "Columns and partitions"),
    ("ifc_export", "Building structure to IFC"),
    ("furniture", "Furniture and equipment"),
    ("components_describe", "Parametric component description"),
    ("components_ifc", "IFC model from the component description"),
]
REQUIRED = {"numpy": "numpy", "scipy": "scipy", "open3d": "open3d", "shapely": "shapely", "networkx": "networkx",
            "skimage": "scikit-image", "sklearn": "scikit-learn", "cv2": "opencv-python", "matplotlib": "matplotlib",
            "ifcopenshell": "ifcopenshell"}


def ensure_packages() -> None:
    import importlib.util

    missing = [pip_name for module, pip_name in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    req = ROOT / "requirements.txt"
    cmd = [sys.executable, "-m", "pip", "install"] + (["-r", str(req)] if req.exists() else missing)
    print(f"Missing Python libraries: {', '.join(missing)}. Installing them now (one time only)...")
    try:
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, OSError):
        raise SystemExit(f"Automatic installation failed. Install the libraries manually with:\n"
                         f"    {sys.executable} -m pip install -r requirements.txt")


def locate(given: str | None, name: str, fallback: Path) -> Path:
    if given:
        path = Path(given)
        path = path if path.is_absolute() else Path.cwd() / path
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        return path
    for path in (ROOT / name, fallback):
        if path.exists():
            return path
    raise SystemExit(f"{name} not found. Copy it next to run_modeling.py or give its path "
                     f"(see python run_modeling.py --help).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate IFC models from a labelled point cloud.")
    ap.add_argument("--input", help="labelled point cloud (default: floor_merged.ply)")
    ap.add_argument("--positions", help="scanner positions (default: scan_positions.json)")
    ap.add_argument("--output-folder", default="output", help="folder for the results (default: output)")
    ap.add_argument("--project-name", default="Scanned floor", help="project and building name written to the IFC files")
    args = ap.parse_args()

    ensure_packages()
    src = locate(args.input, "floor_merged.ply", ROOT.parent / "4_merge_tiles" / "merged" / "floor_merged.ply")
    positions = locate(args.positions, "scan_positions.json", ROOT.parent / "1_preprocessing" / "scan_positions.json")
    out = Path(args.output_folder)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, SCAN2BIM_INPUT=str(src), SCAN2BIM_POSITIONS=str(positions), SCAN2BIM_OUTPUT=str(out),
               SCAN2BIM_PROJECT=args.project_name, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    print(f"Input point cloud:  {src}")
    print(f"Scanner positions:  {positions}")
    print(f"Output folder:      {out}")
    t_all = time.time()
    for i, (script, title) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {title}", flush=True)
        t = time.time()
        if subprocess.call([sys.executable, "-u", str(ROOT / f"{script}.py")], cwd=ROOT, env=env) != 0:
            raise SystemExit(f"Step '{script}' failed; see the messages above.")
        print(f"    done in {time.time() - t:.0f} s", flush=True)
    print(f"\nFinished in {(time.time() - t_all) / 60:.1f} min. IFC models:")
    for name in ("model_structure.ifc", "model_with_furniture.ifc", "model_components.ifc"):
        print(f"    {out / name}")


if __name__ == "__main__":
    main()
