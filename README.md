# Scan-to-BIM

Turns a terrestrial laser scan of an existing building floor (`.e57`) into IFC4 BIM models.
The model contains walls, glass walls, doors, windows, columns, partitions, rooms, slabs, ceilings and furniture.

The workflow has five steps. Each step is in its own folder, and each step's output is the next step's input:

| Folder | Step | Input | Output |
|---|---|---|---|
| `1_preprocessing` | Clean and voxel-downsample the scan | `.e57` scan | `preprocessed.ply`, `scan_positions.json` |
| `2_label_transfer` | Give every point a starting label with a trained model; optionally cut into tiles | `preprocessed.ply` | `labelled/` (tiles or one file) |
| `3_label_correction` | Correct the labels by hand in the browser | files from `labelled/` | `..._corrected.ply` |
| `4_merge_tiles` | Join the corrected files into one floor | `..._corrected.ply` | `merged/floor_merged.ply` |
| `5_modeling` | Reconstruct the building and export IFC | `floor_merged.ply`, `scan_positions.json` | `output/*.ifc` |

Each folder contains a `HOW_TO_RUN.txt` with step-by-step instructions.

## Requirements

- Windows, macOS or Linux.
- Python 3.9 to 3.11 ([python.org](https://www.python.org/downloads/)).
- Google Chrome or Microsoft Edge, for step 3.

Each script installs its missing Python libraries automatically on the first run. The libraries are listed in the `requirements.txt` file of each folder.

## Quick start

```bash
cd 1_preprocessing
python preprocess_pointcloud.py            # copy the .e57 scan into this folder first

cd ../2_label_transfer
python label_transfer.py                   # reads ../1_preprocessing/preprocessed.ply

# step 3: open 3_label_correction/point_label_corrector.html, correct each file in 2_label_transfer/labelled,
#         and copy the saved ..._corrected.ply files into 4_merge_tiles/corrected

cd ../4_merge_tiles
python merge_tiles.py

cd ../5_modeling
python run_modeling.py --project-name "Building A, level 3"
```

Without arguments, `label_transfer.py` and `run_modeling.py` find their inputs in the output locations of the earlier steps.
Each script accepts `--help` for its options.

## Main settings

| Setting | File | Default | Meaning |
|---|---|---|---|
| `VOXEL_SIZE_M` | `1_preprocessing/preprocess_pointcloud.py` | `0.02` | point spacing after downsampling (m) |
| `TILE_SIZE_M` | `2_label_transfer/label_transfer.py` | `8` | tile size for manual labelling (m); `0` = one file, no tiling |

## Label classes

`0` floor, `1` ceiling, `2` wall, `3` door, `4` window, `5` column, `6` desk, `7` chair, `8` monitor, `9` partition, `10` cabinet_shelf, `11` clutter, `255` unknown.

## Results of step 5

| File | Content |
|---|---|
| `model_structure.ifc` | walls, glass walls, doors, windows, columns, partitions, spaces, slab, ceilings |
| `model_with_furniture.ifc` | the structure plus desks, chairs, monitors, cabinets and shelves as typed objects |
| `model_components.ifc` | the same model rebuilt from its parametric component description (`components.gml`) |
| `stage*_log.txt`, `stage*.png` | log and plan image of each modelling step |

The IFC files open in any IFC viewer, for example BIMvision, Solibri Anywhere, BlenderBIM or Autodesk Revit.

## Data

Scans, point clouds and results are large and are not stored in this repository (see `.gitignore`).
