"""Element clean-up and IFC4 export of the building structure.
  clean-up: L/T junction joining and overlap resolution of walls, partitions and columns (rules in
            junctions.py + priority-based trimming; column > wall > partition)
  IFC: IfcWall (STANDARD, measured or default thickness, floor to ceiling), IfcOpeningElement + IfcDoor / IfcWindow
       hosted by IfcRelVoidsElement / IfcRelFillsElement, IfcWall PARTITIONING (snapped height), IfcColumn,
       IfcSpace per room (graph-cut polygon, measured height), IfcSlab floor, IfcCovering ceiling per room;
       property set Scan_Provenance on every element (which evidence produced it)
Output: model_structure.ifc, stage5_plan.png, elements.csv"""
import csv, json, sys, collections
import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from common import *
import junctions as JN

log = Log(f"{OUT}/stage5_log.txt")
S1 = json.load(open(f"{OUT}/stage1.json")); S2 = json.load(open(f"{OUT}/stage2.json"))
S3 = json.load(open(f"{OUT}/stage3.json")); S4 = json.load(open(f"{OUT}/stage4.json"))
ZF = S1['Z_FLOOR']; ZC = max(r['z_ceil'] for r in S2['ROOMS'])
PRIO = {"column": 0, "wall": 1, "glazing": 2, "partition": 4}
GAP, MIN_LEN, PAR_ANG = 0.001, 0.10, 10.0

EL = []
for i, sl in enumerate(S2['SLABS']):
    n, d = np.array(sl['n']), np.array(sl['d'])
    gl = sl.get('kind') == "glazing"
    EL.append(dict(kind="glazing" if gl else "wall", src=("slab", i), d=d, n=n, o=(sl['o0'] + sl['o1']) / 2, T=sl['o1'] - sl['o0'], t0=sl['t0'], t1=sl['t1'],
                   H=ZC - ZF, prov=("glass wall (window points), 6 cm" if gl else "two faces (measured thickness)") if sl['paired']
                   else f"one face, other side unseen (default {S2['EXT_T']} m)"))
for i, p in enumerate(S4['PARTS']):
    EL.append(dict(kind="partition", src=("part", i), d=np.array(p['d']), n=np.array(p['n']), o=p['o'], T=p['T'], t0=p['t0'], t1=p['t1'],
                   H=p['H'], prov="two faces" if p['paired'] else "one face (median thickness)"))
for i, c in enumerate(S4['COLS']):
    d = np.array(c['d']); n = np.array([-d[1], d[0]]); cc = np.array(c['c'])
    EL.append(dict(kind="column", src=("col", i), d=d, n=n, o=float(cc @ n), T=c['b'], t0=float(cc @ d - c['a'] / 2), t1=float(cc @ d + c['a'] / 2),
                   H=ZC - ZF, prov="column points, outer box"))
for x in EL:
    x['e'] = x['src']


def rect(d, n, t0, t1, o, T):
    return Polygon([d * t0 + n * (o - T / 2), d * t1 + n * (o - T / 2), d * t1 + n * (o + T / 2), d * t0 + n * (o + T / 2)])


def resolve(EL, tag):
    ACC, ACC_POLY, stats = [], [], collections.Counter()
    for x in sorted(EL, key=lambda x: (PRIO[x['kind']], -(x['t1'] - x['t0']))):
        if x['kind'] == "column":
            ACC.append(dict(x)); ACC_POLY.append(rect(x['d'], x['n'], x['t0'], x['t1'], x['o'], x['T'])); continue
        R = rect(x['d'], x['n'], x['t0'], x['t1'], x['o'], x['T'])
        b0, b1 = x['o'] - x['T'] / 2, x['o'] + x['T'] / 2; side = []
        for P_, A_ in zip(ACC_POLY, ACC):
            I = R.intersection(P_)
            if I.area < 1e-6 or abs(A_['d'] @ x['d']) < np.cos(np.radians(PAR_ANG)):
                continue
            q = np.vstack([np.array(g.exterior.coords) for g in (I.geoms if hasattr(I, "geoms") else [I]) if g.geom_type == "Polygon"])
            on = q @ x['n']
            if on.max() - on.min() < 0.5 * x['T'] - 1e-9:
                side.append(on)
                if on.mean() > x['o']:
                    b1 = min(b1, on.min() - GAP)
                else:
                    b0 = max(b0, on.max() + GAP)
        if side and b1 - b0 >= max(0.02, 0.4 * x['T']):
            x['o'], x['T'] = (b0 + b1) / 2, b1 - b0; stats['thinned'] += 1
        R = rect(x['d'], x['n'], x['t0'], x['t1'], x['o'], x['T'])
        hit = [P_.buffer(GAP, join_style=2) for P_ in ACC_POLY if R.intersection(P_).area > 1e-6]
        pieces = [(x['t0'], x['t1'])]
        if hit:
            I = R.intersection(unary_union(hit))
            for g in (I.geoms if hasattr(I, "geoms") else [I]):
                if g.geom_type != "Polygon" or g.area < 1e-7:
                    continue
                t = np.array(g.exterior.coords) @ x['d']; a, b = t.min(), t.max(); nxt = []
                for s0, s1 in pieces:
                    if b <= s0 or a >= s1:
                        nxt.append((s0, s1))
                    else:
                        if a > s0:
                            nxt.append((s0, a))
                        if b < s1:
                            nxt.append((b, s1))
                pieces = nxt
            stats['trimmed'] += 1
        kept = [(a, b) for a, b in pieces if b - a >= MIN_LEN]
        stats['split'] += len(kept) > 1; stats[f"removed {x['kind']}"] += not kept
        for a, b in kept:
            ACC.append(dict(x, t0=a, t1=b)); ACC_POLY.append(rect(x['d'], x['n'], a, b, x['o'], x['T']))
    log(f"overlap resolution ({tag}): {dict(stats)}")
    return ACC


# a glass wall ending within 1 m of a perpendicular wall (or the data extent) is extended to it when the gap is
# full of points (>= 500 per metre at 0.3-2.0 m: junction clutter / mullion artefacts); an empty gap (doorway) stays open
_S = load_ply(SCAN); _P = np.c_[_S['x'], _S['y'], _S['z']].astype(float)
_P = _P[(_P[:, 2] > ZF + 0.3) & (_P[:, 2] < ZF + 2.0)]
CROPB = json.load(open(META))['crop']
for g in [x for x in EL if x['kind'] == "glazing"]:
    for end in ("t0", "t1"):
        sg = -1 if end == "t0" else 1; te = g[end]
        stops = []
        for w in (x for x in EL if x['kind'] == "wall" and abs(x['d'] @ g['d']) < 0.1):
            # wall face nearest to the glass end, measured along the glass axis
            for face in (w['o'] - w['T'] / 2, w['o'] + w['T'] / 2):
                tw = (w['n'] * face) @ g['d'] if abs(w['n'] @ g['d']) > 0.9 else None
                cross = g['o'] - (w['d'] * w['t0'] + w['n'] * w['o']) @ g['n'], g['o'] - (w['d'] * w['t1'] + w['n'] * w['o']) @ g['n']
                if tw is not None and min(cross) <= 0.2 and max(cross) >= -0.2 and 0 < sg * (tw - te) <= 1.0:
                    stops.append(tw)
        for c in (CROPB[0], CROPB[2]) if abs(g['d'][0]) > 0.9 else (CROPB[1], CROPB[3]):
            tc = c * np.sign(g['d'][0] if abs(g['d'][0]) > 0.9 else g['d'][1])
            if 0 < sg * (tc - te) <= 1.0:
                stops.append(tc)
        if not stops:
            continue
        tn = min(stops, key=lambda t: abs(t - te))
        a, b = sorted((te, tn))
        q = _P[:, :2]; inband = (np.abs(q @ g['n'] - g['o']) < 0.15) & (q @ g['d'] > a) & (q @ g['d'] < b)
        dens = inband.sum() / max(b - a, 1e-3)
        if dens >= 500:
            log(f"glass wall {end} extended {abs(tn - te):.2f} m to {'data extent' if tn in [c * 1 for c in CROPB] or abs(abs(tn) - CROPB[2]) < 1e-6 or abs(abs(tn) - CROPB[0]) < 1e-6 else 'wall'} (gap holds {dens:.0f} points/m)")
            g[end] = tn
        else:
            log(f"glass wall {end}: gap of {abs(tn - te):.2f} m left open ({dens:.0f} points/m) -> possible doorway")
del _S, _P
log(f"elements in: {collections.Counter(x['kind'] for x in EL)}")
ACC = resolve(EL, "pass 1")
for it in (2, 3):
    ACC = JN.remove_stubs(ACC, log)
    JN.join(ACC, PRIO, log)
    for x in ACC:
        x['t1'] = max(x['t1'], x['t0'] + 1e-3)
    ACC = resolve(ACC, f"pass {it}")
# partition pieces shorter than 0.30 m are panel end faces or leftovers between crossing panels
_short = [x for x in ACC if x['kind'] == "partition" and x['t1'] - x['t0'] < 0.30]
ACC = [x for x in ACC if not (x['kind'] == "partition" and x['t1'] - x['t0'] < 0.30)]
log(f"partition pieces < 0.30 m removed: {len(_short)}")
log(f"elements out: {collections.Counter(x['kind'] for x in ACC)}")

# ---------------------------------------------------------------- IFC
import ifcopenshell, ifcopenshell.guid
F = ifcopenshell.file(schema="IFC4"); GID = ifcopenshell.guid.new
p3 = lambda x, y, z: F.createIfcCartesianPoint((float(x), float(y), float(z)))
p2 = lambda x, y: F.createIfcCartesianPoint((float(x), float(y)))
d3 = lambda x, y, z: F.createIfcDirection((float(x), float(y), float(z)))
ax3 = lambda o, x=(1, 0): F.createIfcAxis2Placement3D(p3(*o), d3(0, 0, 1), d3(x[0], x[1], 0))
units = F.createIfcUnitAssignment([F.createIfcSIUnit(None, u, None, n) for u, n in
                                   (("LENGTHUNIT", "METRE"), ("AREAUNIT", "SQUARE_METRE"), ("VOLUMEUNIT", "CUBIC_METRE"), ("PLANEANGLEUNIT", "RADIAN"))])
ctx = F.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, ax3((0, 0, 0)), None)
body = F.createIfcGeometricRepresentationSubContext("Body", "Model", None, None, None, None, ctx, None, "MODEL_VIEW", None)
proj = F.createIfcProject(GID(), None, PROJECT_NAME, "Scan-to-BIM model generated from a labelled laser scan", None, None, None, [ctx], units)
spl = F.createIfcLocalPlacement(None, ax3((0, 0, 0)))
site = F.createIfcSite(GID(), None, "Site", None, None, spl, None, None, "ELEMENT", None, None, None, None, None)
bpl = F.createIfcLocalPlacement(spl, ax3((0, 0, 0)))
bld = F.createIfcBuilding(GID(), None, PROJECT_NAME, None, None, bpl, None, None, "ELEMENT", None, None, None)
stpl = F.createIfcLocalPlacement(bpl, ax3((0, 0, ZF)))
sto = F.createIfcBuildingStorey(GID(), None, "Level 0", None, None, stpl, None, None, "ELEMENT", ZF)
for a_, b_ in ((proj, site), (site, bld), (bld, sto)):
    F.createIfcRelAggregates(GID(), None, None, None, a_, [b_])
rep = lambda s: F.createIfcProductDefinitionShape(None, None, [F.createIfcShapeRepresentation(body, "Body", "SweptSolid", [s])])


def box_solid(Lx, Ly, H):
    pr = F.createIfcRectangleProfileDef("AREA", None, F.createIfcAxis2Placement2D(p2(0, 0), None), float(Lx), float(Ly))
    return F.createIfcExtrudedAreaSolid(pr, ax3((0, 0, 0)), d3(0, 0, 1), float(H))


def poly_solid(poly, H):
    pl_ = lambda ring: F.createIfcPolyline([p2(x, y) for x, y in ring.coords])
    if len(poly.interiors):
        pr = F.createIfcArbitraryProfileDefWithVoids("AREA", None, pl_(poly.exterior), [pl_(r) for r in poly.interiors])
    else:
        pr = F.createIfcArbitraryClosedProfileDef("AREA", None, pl_(poly.exterior))
    return F.createIfcExtrudedAreaSolid(pr, ax3((0, 0, 0)), d3(0, 0, 1), float(H))


def pset(o, props):
    v = [F.createIfcPropertySingleValue(k, None, F.create_entity("IfcReal" if isinstance(x, float) else "IfcLabel", x), None) for k, x in props.items()]
    F.createIfcRelDefinesByProperties(GID(), None, None, None, [o], F.createIfcPropertySet(GID(), None, "Scan_Provenance", None, v))


CONT, ROWS, WALLENT = [], [], []
cnt = collections.Counter()
for x in ACC:
    Lx = x['t1'] - x['t0']; c = x['d'] * (x['t0'] + x['t1']) / 2 + x['n'] * x['o']
    pl = F.createIfcLocalPlacement(stpl, ax3((c[0], c[1], 0), x['d']))
    cnt[x['kind']] += 1; nm = f"{x['kind'].title()} {cnt[x['kind']]:03d}"
    if x['kind'] == "column":
        e = F.createIfcColumn(GID(), None, nm, None, None, pl, rep(box_solid(Lx, x['T'], x['H'])), None, "COLUMN")
    elif x['kind'] == "glazing":
        e = F.createIfcCurtainWall(GID(), None, nm, x['prov'], None, pl, rep(box_solid(Lx, x['T'], x['H'])), None, "NOTDEFINED")
    else:
        e = F.createIfcWall(GID(), None, nm, x['prov'], None, pl, rep(box_solid(Lx, x['T'], x['H'])), None,
                            "PARTITIONING" if x['kind'] == "partition" else "STANDARD")
    pset(e, {"Source": x['prov'], "Length_m": round(float(Lx), 3), "Thickness_m": round(float(x['T']), 3), "Height_m": round(float(x['H']), 3)})
    CONT.append(e); x['ent'] = e; x['pl'] = pl
    ROWS.append([nm, e.is_a(), x['kind'], round(float(c[0]), 3), round(float(c[1]), 3), round(float(Lx), 3), round(float(x['T']), 3), round(float(x['H']), 3), x['prov']])
nop = collections.Counter()
for o in S3['OPEN']:
    sl = S2['SLABS'][o['slab']]; d, n = np.array(sl['d']), np.array(sl['n'])
    tm = (o['t0'] + o['t1']) / 2; cxy = d * tm + n * (sl['o0'] + sl['o1']) / 2
    host = min((x for x in ACC if x['kind'] == "wall"), key=lambda x: abs(cxy @ x['n'] - x['o']) + max(0, x['t0'] - cxy @ x['d'], cxy @ x['d'] - x['t1'])
               if abs(x['d'] @ d) > 0.99 else 1e9)
    sg = 1 if host['d'] @ d > 0 else -1
    tl = sg * tm - (host['t0'] + host['t1']) / 2
    opl = F.createIfcLocalPlacement(host['pl'], ax3((tl, 0, o['z0'])))
    nop[o['kind']] += 1; nm = f"{o['kind'].title()} {nop[o['kind']]:03d}"
    op = F.createIfcOpeningElement(GID(), None, f"Opening for {nm}", None, None, opl, rep(box_solid(o['width'], host['T'] + 0.1, o['height'])), None, "OPENING")
    F.createIfcRelVoidsElement(GID(), None, None, None, host['ent'], op)
    # leaf / glass plane: windows in one-face (exterior/core) walls at the observed interior face, everything else at
    # the wall centre. (The scan's window points are not used for this: 'return'-mode glass echoes lie ~14 cm behind the pane.)
    yax = np.array([-host['d'][1], host['d'][0]])                  # local y axis of the host placement
    chost = host['d'] * (host['t0'] + host['t1']) / 2 + host['n'] * host['o']
    if o['kind'] == "window" and not sl['paired']:
        ly = float((sl['o1'] - 0.025) * (yax @ n) - chost @ yax)
    else:
        ly = 0.0
    fpl = F.createIfcLocalPlacement(opl, ax3((0, ly, 0)))
    ev = o['evidence']; src = "labels" if ev['door_px'] + ev['window_px'] > ev['empty_px'] else "ray casting (seen empty)"
    if o['kind'] == "door":
        el = F.createIfcDoor(GID(), None, nm, src, None, fpl, rep(box_solid(o['width'], 0.05, o['height'])), None, float(o['height']), float(o['width']), "DOOR", None, None)
    else:
        el = F.createIfcWindow(GID(), None, nm, src, None, fpl, rep(box_solid(o['width'], 0.05, o['height'])), None, float(o['height']), float(o['width']), "WINDOW", None, None)
    pset(el, {"Source": src, "Width_m": round(o['width'], 3), "Height_m": round(o['height'], 3), "Sill_m": round(o['z0'], 3)})
    F.createIfcRelFillsElement(GID(), None, None, None, op, el); CONT.append(el)
    ROWS.append([nm, el.is_a(), o['kind'], round(float(cxy[0]), 3), round(float(cxy[1]), 3), round(o['width'], 3), "", round(o['height'], 3), src])
SP = []
foot = unary_union([shape(r['poly']) for r in S2['ROOMS']] + [rect(x['d'], x['n'], x['t0'], x['t1'], x['o'], x['T']) for x in ACC if x['kind'] in ('wall', 'glazing')]).buffer(0.02, join_style=2).buffer(-0.02, join_style=2)
for r in S2['ROOMS']:
    p = shape(r['poly']); H = r['z_ceil'] - ZF
    sp = F.createIfcSpace(GID(), None, r['name'], None, None, F.createIfcLocalPlacement(stpl, ax3((0, 0, 0))), rep(poly_solid(p, H)),
                          f"{r['area']:.1f} m2", "ELEMENT", "INTERNAL", 0.0)
    pset(sp, {"Source": "station-visibility clusters + graph cut", "Area_m2": round(r['area'], 2), "Height_m": round(H, 3), "Stations": str(r['stations'])})
    SP.append(sp)
    CONT.append(F.createIfcCovering(GID(), None, f"Ceiling {r['name']}", None, None, F.createIfcLocalPlacement(stpl, ax3((0, 0, H))),
                                    rep(poly_solid(p, 0.03)), None, "CEILING"))
    ROWS.append([r['name'], "IfcSpace", "space", round(p.centroid.x, 3), round(p.centroid.y, 3), round(r['area'], 2), "", round(H, 3), "graph cut"])
for k, g in enumerate(foot.geoms if hasattr(foot, "geoms") else [foot]):
    CONT.append(F.createIfcSlab(GID(), None, f"Floor slab {k + 1}", None, None, F.createIfcLocalPlacement(stpl, ax3((0, 0, -0.2))),
                                rep(poly_solid(g, 0.2)), None, "FLOOR"))
F.createIfcRelContainedInSpatialStructure(GID(), None, None, None, CONT, sto)
F.createIfcRelAggregates(GID(), None, None, None, sto, SP)
IFC = IFC_STRUCTURE; F.write(IFC)
log(f"IFC written {IFC}: " + str({t: len(F.by_type(t)) for t in ["IfcWall", "IfcCurtainWall", "IfcDoor", "IfcWindow", "IfcOpeningElement", "IfcColumn", "IfcSpace", "IfcSlab", "IfcCovering"]}))
with open(f"{OUT}/elements.csv", "w", newline="") as fo:
    wr = csv.writer(fo); wr.writerow(["name", "ifc", "kind", "cx", "cy", "length_or_width_or_area", "thickness", "height", "source"]); wr.writerows(ROWS)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MP
fig, ax = plt.subplots(figsize=(18, 12)); cm = plt.get_cmap("tab20")
for i, r in enumerate(S2['ROOMS']):
    p = shape(r['poly']); ax.add_patch(MP(np.array(p.exterior.coords), fc=cm(i % 20), alpha=0.25, ec="none"))
    q = p.representative_point(); ax.text(q.x, q.y, f"{r['name']}\n{r['area']:.1f} m²", ha="center", fontsize=9)
col = {"wall": "#333333", "glazing": "#1f9ed1", "partition": "#2ca02c", "column": "#d62728"}
for x in ACC:
    ax.add_patch(MP(np.array(rect(x['d'], x['n'], x['t0'], x['t1'], x['o'], x['T']).exterior.coords), fc=col[x['kind']], ec="none"))
for o in S3['OPEN']:
    sl = S2['SLABS'][o['slab']]; d, n = np.array(sl['d']), np.array(sl['n']); om = (sl['o0'] + sl['o1']) / 2
    a, b = d * o['t0'] + n * om, d * o['t1'] + n * om
    ax.plot([a[0], b[0]], [a[1], b[1]], c="#ff7f0e" if o['kind'] == "door" else "#1f77b4", lw=5, solid_capstyle="butt")
ax.plot([], [], c="#ff7f0e", lw=5, label="door"); ax.plot([], [], c="#1f77b4", lw=5, label="window")
ax.legend(loc="lower right"); ax.set_aspect("equal"); ax.autoscale_view(); ax.set_title("Model (plan)")
plt.tight_layout(); plt.savefig(f"{OUT}/stage5_plan.png", dpi=70)
