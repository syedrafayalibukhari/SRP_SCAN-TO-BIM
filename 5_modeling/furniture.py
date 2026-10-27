"""Furniture and equipment as typed, parametric BIM objects.
  instances (Wang 2022 / Mahmoud 2024): per class, points projected to a 2 cm grid, morphological erosion, connected
      components, points re-assigned to the nearest component (dilation back); clusters below a minimum size dropped
  geometry: oriented box from the 1st-99th percentiles (desks/cabinets snapped to the building directions within 10 deg,
      chairs free); front direction: chair = side of the backrest points; desk/monitor = towards the nearest chair
  types (Tang 2022 parametric description / Mahmoud 2024 family placement): instances of one class with dimensions
      within TYPE_TOL are one type; each type is an IFC type object with a parametric RepresentationMap built from its mean
      dimensions (desk = top + end panels, chair = base + column + seat + back, monitor = screen + neck + foot,
      laptop = base + lid, cabinet/shelf = box); occurrences use IfcMappedItem + their own placement, keep measured
      dimensions in a property set and are contained in the IfcSpace they stand in
  IFC classes: IfcFurniture DESK / CHAIR / FILECABINET / SHELF, IfcAudioVisualAppliance DISPLAY (monitor),
      IfcCommunicationsAppliance COMPUTER (laptop)
  clutter (papers, cups, bags, bins, plants ...) is not modelled
Input: model_structure.ifc (unchanged). Output: model_with_furniture.ifc, stage7.json, stage7_furniture.png"""
import json, collections
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from common import *

log = Log(f"{OUT}/stage7_log.txt")
S1 = json.load(open(f"{OUT}/stage1.json")); S2 = json.load(open(f"{OUT}/stage2.json"))
S = load_ply(SCAN)
P = np.c_[S['x'], S['y'], S['z']].astype(float); N = np.c_[S['nx'], S['ny'], S['nz']].astype(float); L = S['lab']
ZF = S1['Z_FLOOR']; FAM = S1['FAM']
DESK, CHAIR, MON, CAB = 6, 7, 8, 10
GRID = 0.02
# class: (closing px, erosion px, min points, min footprint m2, snap to building directions, z-band used for the footprint)
CFG = {DESK: (2, 1, 800, 0.25, True, (0.0, 0.9)), CHAIR: (1, 1, 300, 0.05, False, (0.0, 1.3)),
       MON: (2, 0, 120, 0.01, True, (0.6, 1.4)), CAB: (2, 1, 80, 0.05, True, (0.0, 2.6))}
SPLIT = {MON}               # classes whose merged neighbours are split with the scene's own size prior
S4 = json.load(open(f"{OUT}/stage4.json"))
# separators: modelled partitions and walls (furniture never crosses them) -> cut lines for the instance raster
SEP = []
for p_ in S4['PARTS']:
    SEP.append((np.array(p_['d']), np.array(p_['n']), p_['t0'], p_['t1'], p_['o'], p_['T']))
for sl in S2['SLABS']:
    SEP.append((np.array(sl['d']), np.array(sl['n']), sl['t0'], sl['t1'], (sl['o0'] + sl['o1']) / 2, sl['o1'] - sl['o0']))


def separator_mask(x0, y0, shape_, dil=1):
    m = np.zeros(shape_, bool)
    for d, n, t0, t1, o, T in SEP:
        t = np.arange(t0, t1, GRID / 2); w = np.arange(-T / 2, T / 2 + 1e-9, GRID / 2)
        q = (d[None, None, :] * t[:, None, None] + n[None, None, :] * (o + w[None, :, None])).reshape(-1, 2)
        ij = ((q - [x0, y0]) / GRID).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape_[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape_[1])
        m[ij[ok, 0], ij[ok, 1]] = True
    return ndimage.binary_dilation(m, iterations=dil) if dil else m


def split_by_size(comps, lid):
    W = []
    for pid in comps:
        b = obox(pid, CFG[lid][4]); W.append(b['W'])
    W = np.array(W); ref = np.median(W[W < 1.5 * np.median(W)])
    out = []
    for pid, w in zip(comps, W):
        k = int(round(w / ref))
        if k >= 2 and abs(w / k - ref) <= 0.25 * ref:
            b = obox(pid, CFG[lid][4]); u = P[pid, :2] @ b['d']
            cen = np.linspace(u.min() + w / (2 * k), u.max() - w / (2 * k), k)
            for _ in range(10):                                   # 1-D k-means along the long side
                a = np.argmin(np.abs(u[:, None] - cen[None, :]), 1)
                cen = np.array([u[a == j].mean() if (a == j).any() else cen[j] for j in range(k)])
            out += [pid[a == j] for j in range(k) if (a == j).sum() > 20]
        else:
            out.append(pid)
    return out, ref
TYPE_TOL = 0.06


def instances(lid):
    cl_, er, minp, mina, snap, (za, zb) = CFG[lid]
    idx = np.where(L == lid)[0]
    band = idx[(P[idx, 2] - ZF > za) & (P[idx, 2] - ZF < zb)]
    x0, y0 = P[band, :2].min(0) - 0.1
    ij = ((P[band, :2] - [x0, y0]) / GRID).astype(int)
    img = np.zeros(ij.max(0) + 3, bool); img[ij[:, 0], ij[:, 1]] = True
    img = ndimage.binary_closing(img, iterations=cl_)        # bridges holes left by occluding objects on top
    if lid in (DESK, CAB):
        img &= ~separator_mask(x0, y0, img.shape, 1 if lid == DESK else 0)
    core = ndimage.binary_erosion(img, iterations=er) if er else img
    lab, n = ndimage.label(core)
    if n == 0:
        return []
    cells = np.argwhere(lab > 0); tr = cKDTree(cells)
    ijall = ((P[idx, :2] - [x0, y0]) / GRID).astype(int)
    dd, nn = tr.query(ijall, distance_upper_bound=er + 2)
    ok = np.isfinite(dd)
    comp = np.full(len(idx), -1); comp[ok] = lab[cells[nn[ok], 0], cells[nn[ok], 1]]
    out = []
    for k in range(1, n + 1):
        pid = idx[comp == k]
        if len(pid) >= minp:
            out.append(pid)
    return out


def obox(pid, snap):
    xy = P[pid, :2]
    lo, hi = np.percentile(xy, [1, 99], axis=0); xy = xy[np.all((xy >= lo - 0.02) & (xy <= hi + 0.02), 1)]
    (cx, cy), (w, h), a = cv2.minAreaRect(xy.astype(np.float32))
    ang = np.radians(a)
    if snap:
        best = min([f + 90 * q for f in FAM for q in range(4)], key=lambda f: abs((np.degrees(ang) - f + 45) % 90 - 45))
        if abs((np.degrees(ang) - best + 45) % 90 - 45) <= 10:
            ang = np.radians(best)
    d = np.array([np.cos(ang), np.sin(ang)]); n = np.array([-d[1], d[0]])
    u, v = xy @ d, xy @ n
    u0, u1 = np.percentile(u, [0.5, 99.5]); v0, v1 = np.percentile(v, [0.5, 99.5])
    c = d * (u0 + u1) / 2 + n * (v0 + v1) / 2
    z = P[pid, 2]
    W, D = u1 - u0, v1 - v0
    if D > W:                                         # local x = longer side
        d, n, W, D = n, -d, D, W
    return dict(c=c, d=d, n=n, W=float(W), D=float(D), z0=float(max(ZF, np.percentile(z, 0.5))), H=float(np.percentile(z, 99.5) - ZF))


from shapely.geometry import Polygon as SPoly
from shapely.ops import unary_union
from shapely.strtree import STRtree
SEP_POLY = [SPoly([d * t0 + n * (o - T / 2), d * t1 + n * (o - T / 2), d * t1 + n * (o + T / 2), d * t0 + n * (o + T / 2)]) for d, n, t0, t1, o, T in SEP]
_c = json.load(open(META))['crop']                    # objects may not grow past the data extent
SEP_POLY += [SPoly([(_c[0] - 1, _c[1] - 1), (_c[2] + 1, _c[1] - 1), (_c[2] + 1, _c[1]), (_c[0] - 1, _c[1])]),
             SPoly([(_c[0] - 1, _c[3]), (_c[2] + 1, _c[3]), (_c[2] + 1, _c[3] + 1), (_c[0] - 1, _c[3] + 1)]),
             SPoly([(_c[0] - 1, _c[1]), (_c[0], _c[1]), (_c[0], _c[3]), (_c[0] - 1, _c[3])]),
             SPoly([(_c[2], _c[1]), (_c[2] + 1, _c[1]), (_c[2] + 1, _c[3]), (_c[2], _c[3])])]
ST = stations()


def fp(o):
    return SPoly([o['c'] + sx * o['d'] * o['W'] / 2 + sy * o['n'] * o['D'] / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def complete(objs, kind, log_):
    """context completion of partly seen boxes: grow towards the scene's reference size of the class (80th percentile of
    the observed boxes), along the long side in both directions and in depth away from the nearest scanner station,
    stopping 1 cm before walls, partitions or an already accepted object of the same class"""
    if len(objs) < 3:
        return
    Wr, Dr = np.percentile([o['W'] for o in objs], 80), np.percentile([o['D'] for o in objs], 80)
    acc, grown = [], 0
    for o in sorted(objs, key=lambda o: -o['W'] * o['D']):
        blockers = STRtree(SEP_POLY + acc)
        free = lambda q: not any(q.buffer(-0.01).intersects(blockers.geometries[int(i)]) for i in blockers.query(q.buffer(-0.01)))
        st = ST[np.argmin(np.linalg.norm(ST[:, :2] - o['c'], axis=1)), :2]
        away = o['n'] if (o['c'] - st) @ o['n'] > 0 else -o['n']
        steps = 0
        while o['W'] < Wr and steps < 200:
            moved = False
            for sg in (1, -1):
                q = dict(o, W=o['W'] + 0.01, c=o['c'] + sg * o['d'] * 0.005)
                if o['W'] < Wr and free(fp(q)):
                    o.update(W=q['W'], c=q['c']); moved = True
            steps += 1
            if not moved:
                break
        while o['D'] < Dr and steps < 400:
            q = dict(o, D=o['D'] + 0.01, c=o['c'] + away * 0.005)
            if not free(fp(q)):
                break
            o.update(D=q['D'], c=q['c']); steps += 1
        grown += steps > 0
        acc.append(fp(o))
    log_(f"{kind}: context completion towards {Wr:.2f} x {Dr:.2f} m grew {grown} of {len(objs)} boxes")


def max_rect(B):
    """largest all-True axis-aligned rectangle in a boolean grid (histogram method) -> (area, i0, i1, j0, j1)"""
    best = (0, 0, 0, 0, 0); h = np.zeros(B.shape[1], int)
    for i in range(B.shape[0]):
        h = np.where(B[i], h + 1, 0); st = []
        for j in range(B.shape[1] + 1):
            cur = h[j] if j < B.shape[1] else 0
            while st and h[st[-1]] >= cur:
                top = st.pop(); hh = h[top]; left = st[-1] + 1 if st else 0
                if hh * (j - left) > best[0]:
                    best = (hh * (j - left), i - hh + 1, i + 1, left, j)
            st.append(j)
    return best


def desk_rects(pid):
    """cubicle desks are L-shaped -> up to two rectangles (main top + return) from the desk-top raster
    (desk + monitor points at 0.6-1.0 m, 5 cm cells, gaps closed, holes filled) - no growing towards a typical size"""
    b = obox(pid, True); d, n = b['d'], b['n']
    q = P[pid]; top = q[(q[:, 2] > ZF + 0.6) & (q[:, 2] < ZF + 1.0)]
    zt = float(np.percentile(top[:, 2], 90)) - ZF if len(top) > 50 else b['H']
    lo = np.array([q[:, :2] @ d, q[:, :2] @ n]).min(1) - 0.1; hi = np.array([q[:, :2] @ d, q[:, :2] @ n]).max(1) + 0.1
    mon = P[(L == MON) & (P[:, 2] > ZF + 0.6) & (P[:, 2] < ZF + 1.0)]
    pts = np.vstack([top[:, :2], mon[:, :2]])
    uv = np.c_[pts @ d, pts @ n]; uv = uv[np.all((uv >= lo) & (uv <= hi), 1)]
    G = np.zeros((int((hi[0] - lo[0]) / 0.05) + 1, int((hi[1] - lo[1]) / 0.05) + 1), bool)
    ij = ((uv - lo) / 0.05).astype(int); G[ij[:, 0], ij[:, 1]] = True
    G = ndimage.binary_fill_holes(ndimage.binary_closing(G, iterations=2))
    out = []
    for _ in range(2):
        a, i0, i1, j0, j1 = max_rect(G)
        W, D = (i1 - i0) * 0.05, (j1 - j0) * 0.05
        if a * 0.0025 < (0.25 if not out else 0.15) or min(W, D) < 0.3:
            break
        c = d * (lo[0] + (i0 + i1) * 0.025) + n * (lo[1] + (j0 + j1) * 0.025)
        dd, nn = (d, n) if W >= D else (n, -d)
        out.append(dict(c=c, d=dd, n=nn, W=float(max(W, D)), D=float(min(W, D)), z0=ZF, H=zt))
        G[i0:i1, j0:j1] = False
    return out


def split_monitor(pid):
    """two screens side by side are one component -> split by screen direction (dual screens are angled),
    else a component wider than 0.9 m is cut into equal screens at the 1-D k-means centres"""
    q = N[pid]; m = np.abs(q[:, 2]) < 0.5
    a = np.arctan2(q[m, 1], q[m, 0]) * 2                    # doubled angle: screen front and back are the same direction
    if m.sum() > 80:
        X = np.c_[np.cos(a), np.sin(a)]
        c1, c2 = X[np.argmax(X @ X.mean(0))], -X.mean(0) / (np.linalg.norm(X.mean(0)) + 1e-9)
        for _ in range(10):
            lab = (X @ c1) < (X @ c2); c1, c2 = X[~lab].mean(0), X[lab].mean(0) if lab.any() else c2
        sep = np.degrees(np.arccos(np.clip(c1 @ c2 / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-9), -1, 1))) / 2
        if sep > 8 and min(lab.sum(), (~lab).sum()) > 0.2 * m.sum():
            g = np.full(len(pid), -1); g[np.where(m)[0]] = lab.astype(int)
            # points without a usable normal go to the nearer group
            from scipy.spatial import cKDTree as _T
            t = _T(P[pid[g >= 0], :2]); _, k = t.query(P[pid[g < 0], :2]); g[g < 0] = g[g >= 0][k]
            return [pid[g == 0], pid[g == 1]]
    b = obox(pid, True)
    if b['W'] > 0.9:
        k = int(round(b['W'] / 0.6)); u = P[pid, :2] @ b['d']; cen = np.linspace(u.min(), u.max(), 2 * k + 1)[1::2]
        for _ in range(10):
            g = np.argmin(np.abs(u[:, None] - cen[None, :]), 1); cen = np.array([u[g == j].mean() if (g == j).any() else cen[j] for j in range(k)])
        return [pid[g == j] for j in range(k) if (g == j).sum() > 30]
    return [pid]


OBJ = []
for lid, kind in ((DESK, "desk"), (CHAIR, "chair"), (MON, "monitor"), (CAB, "cabinet")):
    comps = instances(lid)
    if lid == MON:
        n0 = len(comps); comps = [q for pid in comps for q in split_monitor(pid)]
        log(f"monitor: {n0} components -> {len(comps)} screens (split by screen direction / width)")
    for pid in comps:
        if lid == DESK:
            for r in desk_rects(pid):
                r.update(kind=kind, npts=int(len(pid)), pid=pid); OBJ.append(r)
            continue
        b = obox(pid, CFG[lid][4]); b.update(kind=kind, npts=int(len(pid)), pid=pid)
        OBJ.append(b)
    cur = [o for o in OBJ if o['kind'] == kind]
    if lid == DESK:
        log(f"desk: {len(comps)} desk components -> {len(cur)} desk tops (L-shaped desks as two rectangles)")
    if lid == CAB:
        complete([o for o in cur if o['H'] < 1.2], "low cabinet", log)
    OBJ = [o for o in OBJ if o['kind'] != kind or o['W'] * o['D'] >= CFG[lid][3]]
log("instances: " + str(collections.Counter(o['kind'] for o in OBJ)))

# sub-types and front directions
chairs = np.array([o['c'] for o in OBJ if o['kind'] == "chair"])
for o in OBJ:
    if o['kind'] == "monitor":
        top = o['H']                                         # top above the floor
        under = [x for x in OBJ if x['kind'] == "desk" and abs((o['c'] - x['c']) @ x['d']) <= x['W'] / 2 + 0.05
                 and abs((o['c'] - x['c']) @ x['n']) <= x['D'] / 2 + 0.10]
        if under:                                            # the monitor foot is hidden -> stand it on the desk top
            o['z0'] = ZF + max(x['H'] for x in under)
        o['H'] = top - (o['z0'] - ZF)                        # stands on the desk: height above its own base
        o['kind'] = "laptop" if o['W'] < 0.42 and o['H'] < 0.32 else "monitor"
    if o['kind'] == "cabinet":
        o['kind'] = "shelf" if o['H'] > 1.2 else "pedestal"
    if o['kind'] == "chair":
        # backrest: points higher than seat + 0.25 m; the front points away from them
        q = P[o['pid']]; top = q[q[:, 2] > ZF + 0.75]
        v = (top[:, :2].mean(0) - o['c']) if len(top) > 20 else -o['n']
        o['front'] = -v / (np.linalg.norm(v) + 1e-9)
    else:
        if len(chairs):
            v = chairs[np.argmin(np.linalg.norm(chairs - o['c'], axis=1))] - o['c']
            o['front'] = o['n'] if v @ o['n'] >= 0 else -o['n']
        else:
            o['front'] = o['n']
    # local frame: y = front, x = y rotated -90 deg
    f = o['front']; o['ax'] = np.array([f[1], -f[0]])
    if o['kind'] == "chair":
        # chair box dims in its own front frame
        q = P[o['pid'], :2] - o['c']
        o['W'] = float(np.ptp(np.percentile(q @ o['ax'], [1, 99]))); o['D'] = float(np.ptp(np.percentile(q @ f, [1, 99])))
    elif abs(o['ax'] @ o['d']) < 0.5:
        o['W'], o['D'] = o['D'], o['W']


def types_for(kind):
    objs = [o for o in OBJ if o['kind'] == kind]
    T = []
    for o in sorted(objs, key=lambda o: -o['W']):
        dims = np.array([o['W'], o['D'], o['H']])
        for t in T:
            if np.all(np.abs(np.mean(t['dims'], 0) - dims) <= TYPE_TOL):
                t['dims'].append(dims); t['objs'].append(o); break
        else:
            T.append(dict(kind=kind, dims=[dims], objs=[o]))
    for t in T:
        t['mean'] = np.round(np.mean(t['dims'], 0), 2)
    return T


TYPES = [t for k in ("desk", "chair", "monitor", "laptop", "pedestal", "shelf") for t in types_for(k)]
log(f"types: {len(TYPES)}")
for t in TYPES:
    log(f"  {t['kind']:<9s} {t['mean'][0] * 1000:5.0f} x {t['mean'][1] * 1000:4.0f} x {t['mean'][2] * 1000:4.0f} mm  occurrences {len(t['objs'])}")

# ---------------------------------------------------------------- IFC
import ifcopenshell, ifcopenshell.guid
F = ifcopenshell.open(IFC_STRUCTURE); GID = ifcopenshell.guid.new
BODY = next(c for c in F.by_type("IfcGeometricRepresentationSubContext") if c.ContextIdentifier == "Body")
STPL = F.by_type("IfcBuildingStorey")[0].ObjectPlacement
p3 = lambda x, y, z: F.createIfcCartesianPoint((float(x), float(y), float(z)))
p2 = lambda x, y: F.createIfcCartesianPoint((float(x), float(y)))
d3 = lambda x, y, z: F.createIfcDirection((float(x), float(y), float(z)))
ax3 = lambda o, x=(1, 0): F.createIfcAxis2Placement3D(p3(*o), d3(0, 0, 1), d3(x[0], x[1], 0))


def blk(cx, cy, z, w, d, h):            # box centred at (cx, cy) in the type frame, from z to z + h
    pr = F.createIfcRectangleProfileDef("AREA", None, F.createIfcAxis2Placement2D(p2(cx, cy), None), float(w), float(d))
    return F.createIfcExtrudedAreaSolid(pr, ax3((0, 0, z)), d3(0, 0, 1), float(h))


def cyl(r, z, h):
    pr = F.createIfcCircleProfileDef("AREA", None, F.createIfcAxis2Placement2D(p2(0, 0), None), float(r))
    return F.createIfcExtrudedAreaSolid(pr, ax3((0, 0, z)), d3(0, 0, 1), float(h))


def parts(kind, W, D, H):
    """parametric geometry in the type frame: origin at the footprint centre on the floor, +y = front"""
    if kind == "desk":
        t = 0.03
        return [blk(0, 0, H - t, W, D, t), blk(-W / 2 + 0.015, 0, 0, 0.03, D - 0.06, H - t), blk(W / 2 - 0.015, 0, 0, 0.03, D - 0.06, H - t),
                blk(0, -D / 2 + 0.1, 0.3, W - 0.06, 0.02, H - t - 0.3)]
    if kind == "chair":
        seat = min(0.48, H * 0.47)
        return [cyl(min(W, D) * 0.45, 0, 0.06), cyl(0.03, 0.06, seat - 0.06), blk(0, 0.02, seat, W * 0.8, D * 0.75, 0.07),
                blk(0, -D / 2 + 0.08, seat + 0.1, W * 0.75, 0.05, H - seat - 0.1)]
    if kind == "monitor":
        sh = min(H * 0.7, H - 0.12)
        return [blk(0, 0, 0, min(0.25, W * 0.5), min(0.2, D), 0.02), blk(0, -0.02, 0.02, 0.05, 0.03, H - sh), blk(0, 0.0, H - sh, W, 0.03, sh)]
    if kind == "laptop":
        return [blk(0, 0.02, 0, W, D * 0.9, 0.02), blk(0, -D / 2 + 0.01, 0.02, W, 0.01, H - 0.02)]
    return [blk(0, 0, 0, W, D, H)]


CLS = {"desk": ("IfcFurniture", "IfcFurnitureType", "DESK"), "chair": ("IfcFurniture", "IfcFurnitureType", "CHAIR"),
       "pedestal": ("IfcFurniture", "IfcFurnitureType", "FILECABINET"), "shelf": ("IfcFurniture", "IfcFurnitureType", "SHELF"),
       "monitor": ("IfcAudioVisualAppliance", "IfcAudioVisualApplianceType", "DISPLAY"),
       "laptop": ("IfcCommunicationsAppliance", "IfcCommunicationsApplianceType", "COMPUTER")}
spaces = [(sp, shape(r['poly'])) for sp in F.by_type("IfcSpace") for r in S2['ROOMS'] if r['name'] == sp.Name]
in_space = collections.defaultdict(list); cnt = collections.Counter()
for ti, t in enumerate(TYPES):
    W, D, H = t['mean']; occ, tcls, pre = CLS[t['kind']]
    nm = f"{t['kind'].title()} {W * 1000:.0f}x{D * 1000:.0f}x{H * 1000:.0f}"
    rmap = F.createIfcRepresentationMap(ax3((0, 0, 0)), F.createIfcShapeRepresentation(BODY, "Body", "SweptSolid", parts(t['kind'], W, D, H)))
    kw = dict(GlobalId=GID(), Name=nm, Description=f"Type from {len(t['objs'])} scanned instances", RepresentationMaps=[rmap], PredefinedType=pre)
    if tcls == "IfcFurnitureType":
        kw['AssemblyPlace'] = "NOTDEFINED"
    T_ = F.create_entity(tcls, **kw)
    occs = []
    for o in t['objs']:
        cnt[t['kind']] += 1
        pl = F.createIfcLocalPlacement(STPL, ax3((o['c'][0], o['c'][1], o['z0'] - ZF if t['kind'] in ("monitor", "laptop") else 0.0), o['ax']))
        mi = F.createIfcMappedItem(rmap, F.createIfcCartesianTransformationOperator3D(None, None, p3(0, 0, 0), None, None))
        shp = F.createIfcProductDefinitionShape(None, None, [F.createIfcShapeRepresentation(BODY, "Body", "MappedRepresentation", [mi])])
        e = F.create_entity(occ, GlobalId=GID(), Name=f"{t['kind'].title()} {cnt[t['kind']]:03d}", ObjectType=nm, ObjectPlacement=pl,
                            Representation=shp, PredefinedType=pre)
        v = [F.createIfcPropertySingleValue(k, None, F.create_entity(ty, x), None) for k, ty, x in
             (("MeasuredWidth_m", "IfcReal", round(o['W'], 3)), ("MeasuredDepth_m", "IfcReal", round(o['D'], 3)),
              ("MeasuredHeight_m", "IfcReal", round(o['H'], 3)),
              ("Points", "IfcInteger", o['npts']), ("Source", "IfcLabel", "class points -> grid components -> oriented box"))]
        F.createIfcRelDefinesByProperties(GID(), None, None, None, [e], F.createIfcPropertySet(GID(), None, "Scan_Furniture", None, v))
        occs.append(e); o['ent'] = e
        sp = next((s for s, poly in spaces if poly.contains(Point(o['c']))), None)
        in_space[sp].append(e)
    F.createIfcRelDefinesByType(GID(), None, None, None, occs, T_)
F.createIfcRelDeclares(GID(), None, None, None, F.by_type("IfcProject")[0], list(F.by_type("IfcTypeObject")))
for sp, els in in_space.items():
    if sp is None:
        F.by_type("IfcRelContainedInSpatialStructure")[0].RelatedElements = list(F.by_type("IfcRelContainedInSpatialStructure")[0].RelatedElements) + els
    else:
        F.createIfcRelContainedInSpatialStructure(GID(), None, None, None, els, sp)
OUT_IFC = IFC_FURNITURE; F.write(OUT_IFC)
log(f"written {OUT_IFC}: " + str({k: len(F.by_type(k)) for k in ["IfcFurniture", "IfcAudioVisualAppliance", "IfcCommunicationsAppliance",
                                                                    "IfcFurnitureType", "IfcAudioVisualApplianceType", "IfcCommunicationsApplianceType"]}))
json.dump([dict(kind=o['kind'], c=o['c'].tolist(), ax=o['ax'].tolist(), W=o['W'], D=o['D'], H=o['H'], z0=o['z0'], npts=o['npts'],
                pid_sample=o['pid'][:2000].tolist()) for o in OBJ], open(f"{OUT}/stage7.json", "w"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MP
fig, ax = plt.subplots(figsize=(20, 13))
col = {"desk": "#8c564b", "chair": "#1f77b4", "monitor": "#9467bd", "laptop": "#e377c2", "pedestal": "#7f7f7f", "shelf": "#2ca02c"}
s = np.where(np.isin(L, [DESK, CHAIR, MON, CAB]))[0][::15]
ax.scatter(P[s, 0], P[s, 1], s=0.05, c="0.75")
for o in OBJ:
    a, f = o['ax'], o['front']
    q = [o['c'] + sx * a * o['W'] / 2 + sy * f * o['D'] / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ax.add_patch(MP(np.array(q), fc="none", ec=col[o['kind']], lw=1.2))
    ax.arrow(o['c'][0], o['c'][1], f[0] * 0.25, f[1] * 0.25, head_width=0.06, color=col[o['kind']])
for k, c in col.items():
    ax.plot([], [], c=c, lw=3, label=k)
ax.legend(loc="lower right"); ax.set_aspect("equal"); ax.autoscale_view(); ax.set_title("stage 7: furniture instances (boxes) and front directions (arrows)")
plt.tight_layout(); plt.savefig(f"{OUT}/stage7_furniture.png", dpi=70)
