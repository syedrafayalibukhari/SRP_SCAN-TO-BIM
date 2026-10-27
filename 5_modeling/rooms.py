"""Rooms.
  visibility (Cui 2019 / Previtali 2018): 2D raster 5 cm; blockers = wall/column/door/window points at 0.3-2.4 m plus the
      detected wall faces; from every scanner station 2880 rays until a blocker -> visible pixel set
  room clustering (Cui 2019): overlap ratio |Vi n Vj| / min(|Vi|, |Vj|) between stations, Otsu threshold, union-find
  cell complex (Oesau 2014 / Cui 2019): wall-face lines extended across the bounding box, polygonized; cells between the
      two faces of a paired wall are fixed "solid"
  ceiling occupancy (Previtali 2018): cells less than half covered by the (closed) ceiling-point map are outside
  labelling: multi-label graph cut (alpha-expansion, Potts) with data term = cell area x (1 - share of the cell seen by
      the room's stations) and outside cost = cell area x (largest share seen by any room);
      smoothness = LAMBDA x edge length x (1 - wall-face support of the edge) (Oesau 2014)
  room heights: median z of ceiling points inside each room (Wei 2023 style height reference)
Output: stage2.json, stage2_rooms.png"""
import json
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Polygon, Point, box, mapping, shape
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree
from matplotlib.path import Path as MPath
from skimage.filters import threshold_otsu
from common import *

log = Log(f"{OUT}/stage2_log.txt")
S1 = json.load(open(f"{OUT}/stage1.json"))
S = load_ply(SCAN)
P = np.c_[S['x'], S['y'], S['z']].astype(float); L = S['lab']
ST = stations()
PIX, LAMBDA, EXT_T = 0.05, 0.6, 0.25
X0, Y0 = P[:, 0].min() - 1.0, P[:, 1].min() - 1.0
NX, NY = int((P[:, 0].max() + 1.0 - X0) / PIX) + 1, int((P[:, 1].max() + 1.0 - Y0) / PIX) + 1
to_px = lambda xy: np.floor((np.asarray(xy) - [X0, Y0]) / PIX).astype(int)

# ---------------------------------------------------------------- blockers
blk = np.zeros((NX, NY), bool)
m = np.isin(L, [WALL, COL, DOOR, WIN]) & (P[:, 2] > 0.3) & (P[:, 2] < 2.4)
ij = to_px(P[m, :2]); blk[ij[:, 0], ij[:, 1]] = True
FACES = [S1['FACES'][i] for sl in S1['SLABS'] for i in sl['faces']]       # faces of the kept slabs only (artefact faces dropped)
for f in FACES:
    n, d = np.array(f['n']), np.array(f['d'])
    t = np.arange(f['t0'], f['t1'], PIX / 2)
    ij = to_px(n * f['o'] + np.outer(t, d)); blk[ij[:, 0], ij[:, 1]] = True
from scipy.ndimage import binary_dilation
blk = binary_dilation(blk, iterations=1)

# ---------------------------------------------------------------- visibility per station
ANG = np.linspace(0, 2 * np.pi, 2880, endpoint=False)
R = np.arange(0.2, 30, PIX / 2)                 # start 20 cm out: a scan position may stand next to a wall
VIS = []
for s in ST:
    xy = s[:2] + np.outer(np.cos(ANG), R)[..., None] * [1, 0] + np.outer(np.sin(ANG), R)[..., None] * [0, 1]
    ij = to_px(xy.reshape(-1, 2)).reshape(len(ANG), len(R), 2)
    ok = (ij[..., 0] >= 0) & (ij[..., 0] < NX) & (ij[..., 1] >= 0) & (ij[..., 1] < NY)
    ijc = np.clip(ij, 0, [NX - 1, NY - 1])
    hit = blk[ijc[..., 0], ijc[..., 1]] | ~ok
    first = np.where(hit.any(1), hit.argmax(1), len(R))
    keep = np.arange(len(R))[None, :] < first[:, None]
    flat = np.unique(ijc[keep][:, 0] * NY + ijc[keep][:, 1])
    VIS.append(flat)
log(f"visibility: {len(ST)} stations, median visible area {np.median([len(v) for v in VIS]) * PIX * PIX:.1f} m2")

# ---------------------------------------------------------------- station clustering
ns = len(ST); OV = np.zeros((ns, ns))
sets = [set(v.tolist()) for v in VIS]
for i in range(ns):
    for j in range(i + 1, ns):
        inter = len(np.intersect1d(VIS[i], VIS[j], assume_unique=True))
        OV[i, j] = OV[j, i] = inter / max(1, min(len(VIS[i]), len(VIS[j])))
vals = OV[np.triu_indices(ns, 1)]
TH = float(threshold_otsu(vals[vals > 0])) if (vals > 0).any() else 0.5
parent = list(range(ns))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i in range(ns):
    for j in range(i + 1, ns):
        if OV[i, j] > TH:
            parent[find(i)] = find(j)
roots = sorted({find(i) for i in range(ns)}, key=lambda r: -sum(find(i) == r for i in range(ns)))
CL = np.array([roots.index(find(i)) for i in range(ns)])
log(f"station clustering: Otsu threshold on overlap ratio {TH:.2f} -> {len(roots)} clusters, sizes {np.bincount(CL).tolist()}")

# visibility share per cluster as rasters
VISK = np.zeros((len(roots), NX * NY), bool)
for i, v in enumerate(VIS):
    VISK[CL[i], v] = True

# ---------------------------------------------------------------- cell complex
bb = box(X0 + 0.5, Y0 + 0.5, X0 + NX * PIX - 0.5, Y0 + NY * PIX - 0.5)
lines = {}
for f in FACES:
    key = (round(f['fam'], 3), round(f['n'][0], 3), round(f['n'][1], 3), round(f['o'], 2))
    lines[key] = f
segs = []
for f in lines.values():
    n, d = np.array(f['n']), np.array(f['d'])
    segs.append(LineString([n * f['o'] - d * 100, n * f['o'] + d * 100]).intersection(bb))
CELLS = [c for c in polygonize(unary_union(segs + [bb.exterior])) if c.area > 1e-4]
SLABS = S1['SLABS']
slab_poly = []
for sl in SLABS:
    if sl['paired']:
        n, d = np.array(sl['n']), np.array(sl['d'])
        slab_poly.append(Polygon([n * sl['o0'] + d * sl['t0'], n * sl['o0'] + d * sl['t1'], n * sl['o1'] + d * sl['t1'], n * sl['o1'] + d * sl['t0']]))
solid = [any(sp.contains(c.representative_point()) for sp in slab_poly) for c in CELLS]
cx = X0 + (np.arange(NX) + 0.5) * PIX; cy = Y0 + (np.arange(NY) + 0.5) * PIX
K = len(roots); OUTL = K
D = np.zeros((len(CELLS), K + 1)); CSH = np.zeros(len(CELLS))
# ceiling occupancy map: ceiling points above 2.3 m, gaps up to 0.3 m closed (lamps, diffusers, dropouts)
from scipy.ndimage import binary_closing, binary_fill_holes
CEILOCC = np.zeros((NX, NY), bool)
_c = P[(L == CEIL) & (P[:, 2] > 2.3)]
_ij = to_px(_c[:, :2]); CEILOCC[_ij[:, 0], _ij[:, 1]] = True
CEILOCC = binary_fill_holes(binary_closing(CEILOCC, iterations=6))
for k_, c in enumerate(CELLS):
    xa, ya, xb, yb = c.bounds
    i0, i1 = np.searchsorted(cx, [xa, xb]); j0, j1 = np.searchsorted(cy, [ya, yb])
    if i1 <= i0 or j1 <= j0:
        D[k_, :K] = c.area; continue
    Xg, Yg = np.meshgrid(cx[i0:i1], cy[j0:j1], indexing="ij")
    ins = MPath(np.array(c.exterior.coords)).contains_points(np.c_[Xg.ravel(), Yg.ravel()])
    if not ins.any():
        D[k_, :K] = c.area; continue
    idx = (np.repeat(np.arange(i0, i1), j1 - j0) * NY + np.tile(np.arange(j0, j1), i1 - i0))[ins]
    share = VISK[:, idx].mean(1)
    D[k_, :K] = c.area * (1 - share); D[k_, OUTL] = c.area * share.max()
    CSH[k_] = CEILOCC.ravel()[idx].mean()
BIG = 1e6
for k_ in range(len(CELLS)):            # Previtali 2018: no ceiling above -> not a room (rays escape through glass)
    if CSH[k_] < 0.5:
        D[k_, :K] = BIG; D[k_, OUTL] = 0
for k_ in range(len(CELLS)):
    if solid[k_]:
        D[k_, :] = BIG; D[k_, OUTL] = 0          # wall interiors are not part of any room

# edges + face support
facepolys = [LineString([np.array(f['n']) * f['o'] + np.array(f['d']) * f['t0'], np.array(f['n']) * f['o'] + np.array(f['d']) * f['t1']]).buffer(0.02) for f in FACES]
fu = unary_union(facepolys)
tr = STRtree(CELLS); EDGES = []
for i, c in enumerate(CELLS):
    for j in tr.query(c):
        j = int(j)
        if j <= i:
            continue
        e = c.boundary.intersection(CELLS[j].boundary)
        if e.length < 1e-4:
            continue
        sup = e.intersection(fu).length / e.length
        EDGES.append((i, j, LAMBDA * e.length * (1 - sup)))


def energy(lab):
    return D[np.arange(len(lab)), lab].sum() + sum(w for i, j, w in EDGES if lab[i] != lab[j])


def expand(lab, a):
    G = nx.DiGraph(); s, t = "s", "t"
    for i in range(len(lab)):
        # source side = keep current label, sink side = take label a  (cut cost of s->i is paid if i takes a)
        G.add_edge(s, i, capacity=float(D[i, a]))
        G.add_edge(i, t, capacity=float(D[i, lab[i]]) if lab[i] != a else BIG * 10)
    for i, j, w in EDGES:
        if w <= 0:
            continue
        li, lj = lab[i], lab[j]
        if li == lj:
            if li != a:                               # both keep (0) / both a (0) / one changes (w)
                G.add_edge(i, j, capacity=w); G.add_edge(j, i, capacity=w)
        else:
            # Potts with different current labels: auxiliary node
            q = ("aux", i, j)
            G.add_edge(i, q, capacity=w); G.add_edge(q, i, capacity=w)
            G.add_edge(j, q, capacity=w); G.add_edge(q, j, capacity=w)
            G.add_edge(q, t, capacity=w)
    _, (S_, T_) = nx.minimum_cut(G, s, t)
    new = lab.copy()
    for i in range(len(lab)):
        if i in T_:
            new[i] = a
    return new


lab = D.argmin(1); E0 = energy(lab)
for sweep in range(4):
    changed = False
    for a in range(K + 1):
        new = expand(lab, a)
        if energy(new) < energy(lab) - 1e-9:
            lab, changed = new, True
    if not changed:
        break
log(f"cell complex: {len(CELLS)} cells ({sum(solid)} inside paired walls), {len(EDGES)} edges; "
    f"alpha-expansion energy {E0:.1f} -> {energy(lab):.1f} in {sweep + 1} sweeps")

CROP = box(*json.load(open(META))['crop'])                 # whole floor: data extent + 1 m (does not cut rooms)
ROOMS = []
for k in range(K):
    cells = [CELLS[i] for i in np.where(lab == k)[0]]
    if not cells:
        continue
    poly = unary_union(cells).buffer(1e-6).buffer(-1e-6).intersection(CROP)
    for p in (poly.geoms if hasattr(poly, "geoms") else [poly]):
        if p.area < 1.0:
            continue
        p = p.simplify(0.005)
        ins = MPath(np.array(p.exterior.coords)).contains_points(P[L == CEIL, :2])
        zc = float(np.median(P[L == CEIL][ins, 2])) if ins.sum() > 100 else float(S1['CEILS'][-1])
        ROOMS.append(dict(cluster=k, poly=mapping(p), area=p.area, z_ceil=zc, stations=int((CL == k).sum()),
                          clipped=bool(p.boundary.intersection(CROP.exterior).length > 0.5)))
ROOMS.sort(key=lambda r: -r['area'])
for i, r in enumerate(ROOMS):
    r['name'] = f"Space {i + 1:02d}"
log(f"rooms: {len(ROOMS)}; " + ", ".join(f"{r['name']} {r['area']:.1f} m2 (h {r['z_ceil'] - S1['Z_FLOOR']:.2f}{', touches data extent' if r['clipped'] else ''})" for r in ROOMS))

# single-face walls: exterior if the side behind the face is outside
inside = unary_union([shape(r['poly']) for r in ROOMS])
for sl in SLABS:
    if sl['paired']:
        sl['side'] = "interior"; continue
    n, d = np.array(sl['n']), np.array(sl['d']); tm = (sl['t0'] + sl['t1']) / 2
    behind = Point(n * (sl['o1'] - 0.3) + d * tm)
    sl['side'] = "exterior" if not inside.contains(behind) else "interior (one face seen)"
    sl['o0'] = sl['o1'] - (0.06 if sl.get('kind') == "glazing" else EXT_T)     # glass seen from one side: 6 cm, walls: default
log(f"single-face walls: {sum(s['side'] == 'exterior' for s in SLABS)} bounded by outside/unseen space -> thickness {EXT_T} m (not observable), "
    f"{sum(s['side'] == 'interior (one face seen)' for s in SLABS)} interior")
json.dump(dict(ROOMS=ROOMS, SLABS=SLABS, CL=CL.tolist(), TH=TH, EXT_T=EXT_T), open(f"{OUT}/stage2.json", "w"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MP
fig, ax = plt.subplots(1, 3, figsize=(30, 8))
cm = plt.get_cmap("tab20")
img = np.zeros((NX, NY)); img[blk] = -1
for k in range(K):
    img.ravel()[VISK[k]] = k + 1
ax[0].imshow(np.ma.masked_where(img == 0, img).T, origin="lower", extent=[X0, X0 + NX * PIX, Y0, Y0 + NY * PIX], cmap="tab20", interpolation="nearest")
ax[0].scatter(ST[:, 0], ST[:, 1], c=[cm(c % 20) for c in CL], edgecolors="k", s=40)
ax[0].set_title(f"station visibility, {K} station clusters")
for i, c in enumerate(CELLS):
    ax[1].add_patch(MP(np.array(c.exterior.coords), fc="0.3" if solid[i] else ("white" if lab[i] == OUTL else cm(lab[i] % 20)), ec="0.5", lw=0.3))
ax[1].set_title(f"graph-cut labelling of {len(CELLS)} cells (grey = inside paired walls)")
for i, r in enumerate(ROOMS):
    p = shape(r['poly']); ax[2].add_patch(MP(np.array(p.exterior.coords), fc=cm(i % 20), alpha=0.5, ec="k"))
    q = p.representative_point(); ax[2].text(q.x, q.y, f"{r['name']}\n{r['area']:.1f} m²", ha="center", fontsize=8)
ax[2].set_title(f"rooms: {len(ROOMS)}")
for a in ax:
    a.set_xlim(X0, X0 + NX * PIX); a.set_ylim(Y0, Y0 + NY * PIX); a.set_aspect("equal")
plt.tight_layout(); plt.savefig(f"{OUT}/stage2_rooms.png", dpi=60)
