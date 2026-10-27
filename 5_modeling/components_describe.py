"""Parametric component description of the reconstructed floor (Tang et al. 2022, section 3.4), written as
components.gml (XML) and components.json (same content).

Every component is a TYPE (shared parameters, like a Revit family type / IFC type object) plus OCCURRENCES (instance
parameters only: where, how long, which host). Types are found by grouping measured instances.

  Building / Storey      elevation, storey height
  WallType               id, function (exterior | interior | partition), thickness, material, evidence
  Wall                   id, typeID, centre line (start, end), base offset, height, spaceSide (left/right/both of start->end)
  DoorType / WindowType  id, width, height (sill for windows), operation
  Door / Window          id, typeID, wallID, offset of the centre from the wall start; instance width/height/sill
                         (instance parameters, override the type size like Revit instance parameters)
  FloorType / Floor      thickness, material / boundary polygon (with holes)
  CeilingType / Ceiling  thickness, material / boundary polygon, height above floor
  Space                  id, name, boundary polygon (+ holes), height
  ColumnType / Column    width, depth, shape / location, rotation, height
  FurnitureType          category (desk, chair, monitor, laptop, pedestal, shelf), width, depth, height, geometry template
  Furniture              typeID, location, rotation, elevation, spaceID

Inputs: stage1-4 and stage7 JSON (the cleaned walls are recomputed with the clean-up code of ifc_export.py).
"""
import collections
import json
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import shape, Point, Polygon, mapping
from shapely.ops import unary_union

from common import OUT, PROJECT_NAME

HERE = Path(__file__).resolve().parent

# re-use the clean-up of ifc_export.py (walls, partitions, columns) without writing its IFC or log
src = (HERE / "ifc_export.py").read_text(encoding="utf-8")
src = src[:src.index("# ---------------------------------------------------------------- IFC")]
src = src.replace('Log(f"{OUT}/stage5_log.txt")', 'Log(LOGPATH)')
ns = {"__name__": "stage5_cleanup", "LOGPATH": f"{OUT}/describe_log.txt"}
exec(compile(src, "ifc_export.py (clean-up part)", "exec"), ns)
ACC, S1, S2, S3, ZF, ZC, rect, log = ns['ACC'], ns['S1'], ns['S2'], ns['S3'], ns['ZF'], ns['ZC'], ns['rect'], ns['log']
S4 = json.load(open(f"{OUT}/stage4.json")); S7 = json.load(open(f"{OUT}/stage7.json"))
r3 = lambda v: round(float(v), 3)


def group(values, tol):
    """greedy grouping of parameter vectors: returns list of (mean vector, member indices)"""
    G = []
    for i in np.argsort([-v[0] for v in values]):
        v = np.asarray(values[i], float)
        for g in G:
            if np.all(np.abs(np.mean(g[1], 0) - v) <= tol):
                g[1].append(v); g[2].append(i); break
        else:
            G.append([None, [v], [i]])
    return [(np.mean(g[1], 0), g[2]) for g in G]


D = dict(Building=dict(id="B01", name=PROJECT_NAME),
         Storey=dict(id="L00", name="Level 0", elevation=r3(ZF), height=r3(ZC - ZF)))

# ---------------------------------------------------------------- walls
walls = [x for x in ACC if x['kind'] in ("wall", "glazing", "partition")]
for x in walls:
    x['function'] = x['kind'] if x['kind'] in ("partition", "glazing") else ("interior" if "two faces" in x['prov'] else "exterior")
WT = []
for fn in ("exterior", "interior", "glazing", "partition"):
    xs = [x for x in walls if x['function'] == fn]
    for mean, idx in group([[x['T'], x['H']] for x in xs], np.array([0.01, 0.05])):
        tid = f"WT{len(WT) + 1:02d}"
        unseen = any("unseen" in xs[i]['prov'] for i in idx)
        WT.append(dict(id=tid, function=fn, thickness=r3(mean[0]), height=r3(mean[1]),
                       material={"exterior": "Exterior wall (material not observable)", "interior": "Interior wall", "glazing": "Glass",
                                 "partition": "Workstation panel"}[fn],
                       evidence="assumed thickness (only one face scanned)" if unseen else ("glass panel, 6 cm frame depth assumed" if fn == "glazing" else "measured from two faces")))
        for i in idx:
            xs[i]['typeID'] = tid
WALLS = []
for k, x in enumerate(sorted(walls, key=lambda x: (x['function'], -(x['t1'] - x['t0'])))):
    a = x['d'] * x['t0'] + x['n'] * x['o']; b = x['d'] * x['t1'] + x['n'] * x['o']
    x['id'] = f"W{k + 1:03d}"
    left = np.array([-(b - a)[1], (b - a)[0]]) / max(np.linalg.norm(b - a), 1e-9)      # left of start -> end
    mid = (a + b) / 2; probe = lambda sg: any(shape(r['poly']).contains(Point(mid + sg * left * (x['T'] / 2 + 0.3))) for r in S2['ROOMS'])
    side = "both" if probe(1) and probe(-1) else ("left" if probe(1) else ("right" if probe(-1) else "none"))
    WALLS.append(dict(id=x['id'], typeID=x['typeID'], start=[r3(a[0]), r3(a[1])], end=[r3(b[0]), r3(b[1])], baseOffset=0.0, height=r3(x['H']),
                      spaceSide=side))
log(f"walls: {len(WALLS)} occurrences of {len(WT)} wall types")

# ---------------------------------------------------------------- doors / windows (hosted by wallID, offset along the wall)
OPN = []
for o in S3['OPEN']:
    sl = S2['SLABS'][o['slab']]; d, n = np.array(sl['d']), np.array(sl['n'])
    cxy = d * (o['t0'] + o['t1']) / 2 + n * (sl['o0'] + sl['o1']) / 2
    host = min((x for x in walls if x['function'] != "partition" and abs(x['d'] @ d) > 0.99),
               key=lambda x: abs(cxy @ x['n'] - x['o']) + max(0, x['t0'] - cxy @ x['d'], cxy @ x['d'] - x['t1']))
    OPN.append(dict(kind=o['kind'], wall=host, offset=float(cxy @ host['d'] - host['t0']), W=o['width'], H=o['height'], sill=o['z0']))
TYPES_O = {}
for kind in ("door", "window"):
    xs = [o for o in OPN if o['kind'] == kind]
    TYPES_O[kind] = []
    for mean, idx in group([[o['W'], o['H'], o['sill']] for o in xs], np.array([0.10, 0.10, 0.10])):
        tid = f"{'DT' if kind == 'door' else 'NT'}{len(TYPES_O[kind]) + 1:02d}"
        TYPES_O[kind].append(dict(id=tid, width=r3(round(mean[0] / 0.05) * 0.05), height=r3(round(mean[1] / 0.05) * 0.05),
                                  sill=r3(round(mean[2] / 0.05) * 0.05), count=len(idx),
                                  operation="SINGLE_SWING_LEFT" if kind == "door" and mean[0] < 1.3 else
                                  ("DOUBLE_DOOR_SINGLE_SWING" if kind == "door" else "SINGLE_PANEL")))
        for i in idx:
            xs[i]['typeID'] = tid
DOORS, WINDOWS = [], []
for o in OPN:
    rec = dict(typeID=o['typeID'], wallID=o['wall']['id'], offset=r3(o['offset']), width=r3(o['W']), height=r3(o['H']), sill=r3(o['sill']))
    (DOORS if o['kind'] == "door" else WINDOWS).append(rec)
for i, r in enumerate(DOORS):
    r['id'] = f"D{i + 1:03d}"
for i, r in enumerate(WINDOWS):
    r['id'] = f"N{i + 1:03d}"
log(f"doors: {len(DOORS)} of {len(TYPES_O['door'])} types; windows: {len(WINDOWS)} of {len(TYPES_O['window'])} types")

# ---------------------------------------------------------------- spaces, floor, ceilings
ring = lambda r: [[r3(x), r3(y)] for x, y in r.coords]
SPACES, CEILS = [], []
for i, r in enumerate(S2['ROOMS']):
    p = shape(r['poly'])
    holes = [ring(h) for h in p.interiors]
    SPACES.append(dict(id=f"S{i + 1:02d}", name=r['name'], boundary=ring(p.exterior), holes=holes, height=r3(r['z_ceil'] - ZF),
                       clippedByTile=bool(r.get('clipped', False))))
    CEILS.append(dict(id=f"C{i + 1:02d}", typeID="CT01", spaceID=f"S{i + 1:02d}", height=r3(r['z_ceil'] - ZF), boundary=ring(p.exterior), holes=holes))
foot = unary_union([shape(r['poly']) for r in S2['ROOMS']] + [rect(x['d'], x['n'], x['t0'], x['t1'], x['o'], x['T']) for x in walls if x['function'] != "partition"])
foot = foot.buffer(0.02, join_style=2).buffer(-0.02, join_style=2)
FLOORS = [dict(id=f"F{k + 1:02d}", typeID="SL01", boundary=ring(g.exterior), holes=[ring(h) for h in g.interiors])
          for k, g in enumerate(foot.geoms if hasattr(foot, "geoms") else [foot])]

# ---------------------------------------------------------------- columns
cols = [x for x in ACC if x['kind'] == "column"]
CT = []
for mean, idx in group([sorted([x['t1'] - x['t0'], x['T']], reverse=True) for x in cols], np.array([0.03, 0.03])):
    tid = f"KT{len(CT) + 1:02d}"; CT.append(dict(id=tid, shape="rectangular", width=r3(mean[0]), depth=r3(mean[1])))
    for i in idx:
        cols[i]['typeID'] = tid
COLUMNS = []
for i, x in enumerate(cols):
    c = x['d'] * (x['t0'] + x['t1']) / 2 + x['n'] * x['o']
    COLUMNS.append(dict(id=f"K{i + 1:02d}", typeID=x['typeID'], location=[r3(c[0]), r3(c[1])],
                        rotation=r3(np.degrees(np.arctan2(x['d'][1], x['d'][0]))), height=r3(x['H'])))

# ---------------------------------------------------------------- furniture: types by category, small groups folded into the nearest type
FT, FURN = [], []
# per-dimension tolerances (width, depth, height); a monitor's depth is mostly its stand and varies with visibility
TOL = {"desk": (0.10, 0.10, 0.10), "chair": (0.10, 0.10, 0.10), "monitor": (0.10, 0.40, 0.15), "laptop": (0.10, 0.10, 0.10),
       "pedestal": (0.10, 0.10, 0.10), "shelf": (0.15, 0.15, 0.15)}
spaces_poly = [(s['id'], Polygon(s['boundary'], s['holes'])) for s in SPACES]
for cat in ("desk", "chair", "monitor", "laptop", "pedestal", "shelf"):
    xs = [o for o in S7 if o['kind'] == cat]
    if not xs:
        continue
    G = group([[o['W'], o['D'], o['H']] for o in xs], np.array(TOL[cat]))
    big = [g for g in G if len(g[1]) >= 2] or G
    for mean, idx in G:
        if len(idx) < 2 and big is not G:           # one-off: fold into the nearest common type if within 30 cm per dimension
            near = min(big, key=lambda g: np.abs(g[0] - mean).max())
            if np.abs(near[0] - mean).max() <= 0.30:
                near[1].extend(idx); continue
            big.append((mean, idx))
    for mean, idx in big:
        tid = f"FT{len(FT) + 1:02d}"
        FT.append(dict(id=tid, category=cat, width=r3(mean[0]), depth=r3(mean[1]), height=r3(mean[2]), template=cat, count=len(idx)))
        for i in idx:
            o = xs[i]; c = np.array(o['c']); ax = np.array(o['ax'])
            sp = next((sid for sid, p in spaces_poly if p.contains(Point(c))), None)
            FURN.append(dict(typeID=tid, location=[r3(c[0]), r3(c[1])], rotation=r3(np.degrees(np.arctan2(ax[1], ax[0]))),
                             elevation=r3(o['z0'] - ZF if cat in ("monitor", "laptop") else 0.0), spaceID=sp,
                             measured=[r3(o['W']), r3(o['D']), r3(o['H'])]))
for i, f in enumerate(FURN):
    f['id'] = f"FU{i + 1:03d}"
log(f"furniture: {len(FURN)} occurrences of {len(FT)} types " + str(collections.Counter(t['category'] for t in FT)))

DESC = dict(Building=D['Building'], Storey=D['Storey'],
            WallTypes=WT, Walls=WALLS, DoorTypes=TYPES_O['door'], Doors=DOORS, WindowTypes=TYPES_O['window'], Windows=WINDOWS,
            FloorTypes=[dict(id="SL01", thickness=0.2, material="Floor slab (assumed)")], Floors=FLOORS,
            CeilingTypes=[dict(id="CT01", thickness=0.02, material="Suspended ceiling (assumed)")], Ceilings=CEILS,
            Spaces=SPACES, ColumnTypes=CT, Columns=COLUMNS, FurnitureTypes=FT, Furniture=FURN)
json.dump(DESC, open(f"{OUT}/components.json", "w"), indent=1)

# ---------------------------------------------------------------- GML-style XML
root = ET.Element("IndoorBIM", {"version": "1.0", "source": "scan-to-BIM reconstruction (Tang et al. 2022 style description)"})
b = ET.SubElement(root, "Building", {k: str(v) for k, v in D['Building'].items()})
st = ET.SubElement(b, "Storey", {k: str(v) for k, v in D['Storey'].items()})
pos = lambda pts: " ".join(f"{x} {y}" for x, y in pts)
for tag, items in (("WallTypes", WT), ("DoorTypes", TYPES_O['door']), ("WindowTypes", TYPES_O['window']), ("FloorTypes", DESC['FloorTypes']),
                   ("CeilingTypes", DESC['CeilingTypes']), ("ColumnTypes", CT), ("FurnitureTypes", FT)):
    sec = ET.SubElement(st, tag)
    for it in items:
        ET.SubElement(sec, tag[:-1], {k: str(v) for k, v in it.items()})
sec = ET.SubElement(st, "Walls")
for w in WALLS:
    e = ET.SubElement(sec, "Wall", {"id": w['id'], "typeID": w['typeID'], "baseOffset": str(w['baseOffset']), "height": str(w['height']),
                                    "spaceSide": w['spaceSide']})
    ET.SubElement(e, "CentreLine").text = pos([w['start'], w['end']])
for tag, items in (("Doors", DOORS), ("Windows", WINDOWS)):
    sec = ET.SubElement(st, tag)
    for it in items:
        ET.SubElement(sec, tag[:-1], {k: str(v) for k, v in it.items()})
for tag, items in (("Spaces", SPACES), ("Floors", FLOORS), ("Ceilings", CEILS)):
    sec = ET.SubElement(st, tag)
    for it in items:
        e = ET.SubElement(sec, tag[:-1], {k: str(v) for k, v in it.items() if k not in ("boundary", "holes")})
        ET.SubElement(e, "Boundary").text = pos(it['boundary'])
        for h in it.get('holes', []):
            ET.SubElement(e, "Hole").text = pos(h)
for tag, items in (("Columns", COLUMNS), ("Furniture", FURN)):
    sec = ET.SubElement(st, tag)
    for it in items:
        ET.SubElement(sec, "Item" if tag == "Furniture" else "Column", {k: (" ".join(map(str, v)) if isinstance(v, list) else str(v)) for k, v in it.items()})
ET.indent(root)
ET.ElementTree(root).write(f"{OUT}/components.gml", encoding="utf-8", xml_declaration=True)
log("written components.gml / components.json")
for t in WT:
    log(f"  {t['id']} {t['function']:<9s} thickness {t['thickness']:.3f} height {t['height']:.3f}  {t['evidence']}")
for k in ("door", "window"):
    for t in TYPES_O[k]:
        log(f"  {t['id']} {k:<6s} {t['width']:.2f} x {t['height']:.2f} sill {t['sill']:.2f}  x{t['count']}")
for t in CT:
    log(f"  {t['id']} column {t['width']:.3f} x {t['depth']:.3f}")
for t in FT:
    log(f"  {t['id']} {t['category']:<8s} {t['width']:.2f} x {t['depth']:.2f} x {t['height']:.2f}  x{t['count']}")
