"""Shared paths, class definitions and helpers for the modelling steps."""
import json
import os
from pathlib import Path

import numpy as np

OUT = os.environ.get("SCAN2BIM_OUTPUT", str(Path(__file__).resolve().parent / "output"))
SCAN = f"{OUT}/floor.ply"
META = f"{OUT}/meta.json"
PROJECT_NAME = os.environ.get("SCAN2BIM_PROJECT", "Scanned floor")
IFC_STRUCTURE = f"{OUT}/model_structure.ifc"
IFC_FURNITURE = f"{OUT}/model_with_furniture.ifc"
IFC_COMPONENTS = f"{OUT}/model_components.ifc"

NAMES = ["floor", "ceiling", "wall", "door", "window", "column", "desk", "chair", "monitor", "partition", "cabinet_shelf", "clutter"]
FLOOR, CEIL, WALL, DOOR, WIN, COL, PART = 0, 1, 2, 3, 4, 5, 9
DT = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
               ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('lab', 'u1'), ('inst', '<i4')])


def load_ply(path):
    with open(path, "rb") as f:
        while f.readline().strip() != b"end_header":
            pass
        return np.fromfile(f, dtype=DT)


def stations():
    m = json.load(open(META))
    return np.array([s['xyz'] for s in m['stations']], float)


class Log:
    def __init__(self, path):
        self.path, self.lines = path, []

    def __call__(self, s=""):
        print(s, flush=True)
        self.lines.append(str(s))
        open(self.path, "w", encoding="utf-8").write("\n".join(self.lines))
