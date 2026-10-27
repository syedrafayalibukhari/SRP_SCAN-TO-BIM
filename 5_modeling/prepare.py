"""Prepare the labelled floor for modelling.

Reads the labelled point cloud and the scanner positions, estimates a normal for every point (PCA of the
20 nearest neighbours, oriented towards the nearest scanner position) and writes floor.ply
(x y z rgb nx ny nz label instance) and meta.json to the output folder.
"""
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from common import OUT, PROJECT_NAME

PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "<i2", "int16": "<i2",
             "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
             "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def read_labelled_ply(path):
    with open(path, "rb") as f:
        props, n, in_vertex = [], 0, False
        while True:
            s = f.readline().decode("ascii", "ignore").strip()
            if s.startswith("element"):
                in_vertex = s.split()[1] == "vertex"
                n = int(s.split()[2]) if in_vertex else n
            elif s.startswith("property") and in_vertex:
                props.append((s.split()[2], PLY_TYPES[s.split()[1]]))
            elif s == "end_header":
                break
        a = np.fromfile(f, dtype=np.dtype(props), count=n)
    label = next((k for k in ("label", "semantic_label", "classification") if k in a.dtype.names), None)
    if label is None:
        raise SystemExit(f"{path}: no 'label' property found.")
    return a, label


def main():
    src, positions = os.environ["SCAN2BIM_INPUT"], os.environ["SCAN2BIM_POSITIONS"]
    A, label_field = read_labelled_ply(src)
    P = np.c_[A['x'], A['y'], A['z']].astype(float)
    L = A[label_field].astype(np.uint8)
    crop = [float(P[:, 0].min() - 1), float(P[:, 1].min() - 1), float(P[:, 0].max() + 1), float(P[:, 1].max() + 1)]
    ST = json.load(open(positions))['stations']
    VP = np.array([s['xyz'] for s in ST])
    print(f"{len(A):,} points, {len(VP)} scanner positions")

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    N = np.asarray(pc.normals).copy()
    del pc
    near = VP[cKDTree(VP).query(P)[1]]
    N[np.einsum("ij,ij->i", N, near - P) < 0] *= -1
    N[(L == 0) & (N[:, 2] < 0)] *= -1
    N[(L == 1) & (N[:, 2] > 0)] *= -1

    B = np.empty(len(A), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                                ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('lab', 'u1'), ('inst', '<i4')])
    B['x'], B['y'], B['z'] = P[:, 0], P[:, 1], P[:, 2]
    B['r'], B['g'], B['b'] = A['red'], A['green'], A['blue']
    B['nx'], B['ny'], B['nz'] = N[:, 0], N[:, 1], N[:, 2]
    B['lab'] = L
    B['inst'] = -1
    with open(f"{OUT}/floor.ply", "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(B)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "property float nx\nproperty float ny\nproperty float nz\n"
                 "property uchar semantic_label\nproperty int instance_id\nend_header\n").encode())
        B.tofile(f)
    json.dump(dict(scene=PROJECT_NAME, source=os.path.basename(src), crop=crop,
                   stations=[dict(xyz=s['xyz'], scan=s['scan']) for s in ST]),
              open(f"{OUT}/meta.json", "w"), indent=1)
    print("written floor.ply, meta.json")


if __name__ == "__main__":
    main()
