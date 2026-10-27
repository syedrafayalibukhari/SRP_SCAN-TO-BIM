"""Train the starting-label model used by label_transfer.py.

Reads one or more manually labelled point clouds (PLY with a 'label' property, e.g. floor_merged.ply
from ../4_merge_tiles), describes every 5 cm cube exactly as label_transfer.py does, checks the accuracy by leaving
out one part of the floor at a time, then trains on everything and saves model/label_model.txt.gz.
The labelled files are only read, never changed.

Usage:  python train_label_model.py floor_labelled_2cm_final.ply [more_labelled_floors.ply ...]
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from datetime import datetime
from pathlib import Path

from label_transfer import (CLASSES, FEATURE_VOXEL_M, MODEL_INFO_PATH, MODEL_PATH, NEIGHBOURS, PLY_TYPES,
                            compute_features, ensure_packages, voxel_representatives)

ensure_packages()

import lightgbm as lgb
import numpy as np

IGNORE = 255                    # 'unknown' in the Point Label Corrector
PER_CLASS = 40_000              # training cubes per class and floor (cap)
N_BLOCKS = 4                    # accuracy check: floor split into this many strips, one left out at a time
PARAMS = dict(objective="multiclass", num_class=len(CLASSES), learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              seed=0, num_threads=0, verbose=-1)
N_TREES = 400


def read_labelled(path: Path):
    with open(path, "rb") as f:
        props, n, inv = [], 0, False
        while True:
            s = f.readline().decode("ascii", "ignore").strip()
            if s.startswith("element"):
                inv = s.split()[1] == "vertex"
                n = int(s.split()[2]) if inv else n
            elif s.startswith("property") and inv:
                props.append((s.split()[2], PLY_TYPES[s.split()[1]]))
            elif s == "end_header":
                break
        a = np.fromfile(f, dtype=np.dtype(props), count=n)
    xyz = np.c_[a["x"], a["y"], a["z"]].astype(np.float64)
    rgb = np.c_[a["red"], a["green"], a["blue"]].astype(np.uint8)
    return xyz, rgb, a["label"].astype(np.uint8)


def miou(y, p):
    out = {}
    for k in range(len(CLASSES)):
        t, q = y == k, p == k
        if t.sum():
            out[CLASSES[k]] = float((t & q).sum() / (t | q).sum())
    return float(np.mean(list(out.values()))), out


def sample(lab, rng, mask=None):
    ii = np.arange(len(lab)) if mask is None else np.nonzero(mask)[0]
    ii = ii[lab[ii] != IGNORE]
    return np.concatenate([rng.permutation(ii[lab[ii] == k])[:PER_CLASS] for k in range(len(CLASSES))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labelled", nargs="+", help="manually labelled PLY file(s)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    floors = []
    for p in map(Path, args.labelled):
        t = time.time()
        xyz, rgb, lab = read_labelled(p)
        first, _ = voxel_representatives(xyz, FEATURE_VOXEL_M)
        X, levels = compute_features(xyz[first], rgb[first])
        floors.append(dict(name=p.name, X=X, y=lab[first], x=xyz[first, 0], n=len(xyz)))
        print(f"{p.name}: {len(xyz):,} points -> {len(first):,} cubes, unknown "
              f"{100 * (lab[first] == IGNORE).mean():.1f}%  ({time.time() - t:.0f} s)", flush=True)

    # accuracy check: leave one strip of each floor out, train on the rest
    cv = []
    for b in range(N_BLOCKS):
        Xtr, ytr, Xte, yte = [], [], [], []
        for fl in floors:
            edges = np.quantile(fl["x"], np.linspace(0, 1, N_BLOCKS + 1))
            blk = np.clip(np.searchsorted(edges, fl["x"], side="right") - 1, 0, N_BLOCKS - 1)
            tr = sample(fl["y"], rng, blk != b)
            te = (blk == b) & (fl["y"] != IGNORE)
            Xtr.append(fl["X"][tr]); ytr.append(fl["y"][tr]); Xte.append(fl["X"][te]); yte.append(fl["y"][te])
        bst = lgb.train(PARAMS, lgb.Dataset(np.vstack(Xtr), np.concatenate(ytr)), N_TREES)
        m, per = miou(np.concatenate(yte), bst.predict(np.vstack(Xte)).argmax(1))
        cv.append((m, per))
        print(f"left-out strip {b + 1}/{N_BLOCKS}: mIoU {m:.3f}", flush=True)
    mean_cv = float(np.mean([m for m, _ in cv]))
    per_class = {c: float(np.mean([p[c] for _, p in cv if c in p])) for c in CLASSES
                 if any(c in p for _, p in cv)}
    print(f"mean mIoU over left-out strips: {mean_cv:.3f}")

    # final model on everything
    picks = [sample(fl["y"], rng) for fl in floors]
    X = np.vstack([fl["X"][ii] for fl, ii in zip(floors, picks)])
    y = np.concatenate([fl["y"][ii] for fl, ii in zip(floors, picks)])
    bst = lgb.train(PARAMS, lgb.Dataset(X, y), N_TREES)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with gzip.open(MODEL_PATH, "wt", encoding="utf-8") as f:
        f.write(bst.model_to_string())
    info = dict(trained_on=" + ".join(fl["name"] for fl in floors),
                created=f"{datetime.now():%Y-%m-%d}", classes=CLASSES, feature_voxel_m=FEATURE_VOXEL_M,
                neighbours=list(NEIGHBOURS), training_cubes=int(len(y)), trees=N_TREES, lightgbm=lgb.__version__,
                check=dict(method=f"floor split into {N_BLOCKS} strips along x; each strip left out once",
                           mean_miou=round(mean_cv, 3), per_class_iou={k: round(v, 3) for k, v in per_class.items()}))
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB) and {MODEL_INFO_PATH.name}")


if __name__ == "__main__":
    main()
