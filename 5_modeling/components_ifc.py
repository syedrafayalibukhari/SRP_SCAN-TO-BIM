"""Build the BIM model only from the parametric component description components.gml (Tang et al. 2022).
Nothing from the point cloud is read here - the description is the complete interface, so it could equally be fed to a
Revit/Dynamo script (Tang used a toolkit to Revit; Mahmoud 2024 used Dynamo).

Component -> IFC4 mapping
  WallType            IfcWallType + IfcMaterialLayerSet (one layer = type thickness, type material);
                      function 'glazing' -> IfcCurtainWallType / IfcCurtainWall (glass material)
  Wall                IfcWall (STANDARD | PARTITIONING), placement at the centre-line start, 'Axis' polyline + 'Body' extrusion,
                      IfcMaterialLayerSetUsage centred on the axis, Pset_WallCommon.IsExternal
  DoorType/WindowType IfcDoorType / IfcWindowType with a parametric RepresentationMap (lining + leaf / frame + glass)
  Door/Window         IfcOpeningElement cut into the host wall (wallID, offset along the axis, sill) + IfcDoor / IfcWindow
                      filling it, geometry = mapped type geometry, stretched (non-uniform transformation) when the
                      occurrence carries its own width/height (instance parameters); windows in walls with space on one side only sit at the
                      space-side face, all others on the wall axis
  FloorType/Floor     IfcSlabType FLOOR + IfcSlab (boundary with holes, extruded downwards by the type thickness)
  CeilingType/Ceiling IfcCoveringType CEILING + IfcCovering at the ceiling height
  Space               IfcSpace (boundary x height), aggregated into the storey
  ColumnType/Column   IfcColumnType (mapped box) + IfcColumn (location, rotation)
  FurnitureType       IfcFurnitureType DESK | CHAIR | FILECABINET | SHELF, IfcAudioVisualApplianceType DISPLAY,
                      IfcCommunicationsApplianceType COMPUTER, each with a parametric template geometry
  Furniture           occurrence of its type (mapped geometry), contained in its IfcSpace
"""
import xml.etree.ElementTree as ET
import numpy as np
import ifcopenshell, ifcopenshell.guid
from shapely.geometry import Polygon

from common import IFC_COMPONENTS, OUT

root = ET.parse(f"{OUT}/components.gml").getroot()
stg = root.find("Building/Storey")
num = lambda e, k: float(e.get(k))
pts = lambda txt: np.array(txt.split(), float).reshape(-1, 2)

F = ifcopenshell.file(schema="IFC4"); GID = ifcopenshell.guid.new
p3 = lambda x, y, z: F.createIfcCartesianPoint((float(x), float(y), float(z)))
p2 = lambda x, y: F.createIfcCartesianPoint((float(x), float(y)))
d3 = lambda x, y, z: F.createIfcDirection((float(x), float(y), float(z)))
ax3 = lambda o, x=(1, 0): F.createIfcAxis2Placement3D(p3(*o), d3(0, 0, 1), d3(x[0], x[1], 0))
units = F.createIfcUnitAssignment([F.createIfcSIUnit(None, u, None, n) for u, n in
                                   (("LENGTHUNIT", "METRE"), ("AREAUNIT", "SQUARE_METRE"), ("VOLUMEUNIT", "CUBIC_METRE"), ("PLANEANGLEUNIT", "RADIAN"))])
ctx = F.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, ax3((0, 0, 0)), None)
BODY = F.createIfcGeometricRepresentationSubContext("Body", "Model", None, None, None, None, ctx, None, "MODEL_VIEW", None)
AXIS = F.createIfcGeometricRepresentationSubContext("Axis", "Model", None, None, None, None, ctx, None, "GRAPH_VIEW", None)
b_ = root.find("Building")
proj = F.createIfcProject(GID(), None, b_.get("name"), "built from the parametric component description (Tang et al. 2022 style)", None, None, None, [ctx], units)
spl = F.createIfcLocalPlacement(None, ax3((0, 0, 0)))
site = F.createIfcSite(GID(), None, "Site", None, None, spl, None, None, "ELEMENT", None, None, None, None, None)
bpl = F.createIfcLocalPlacement(spl, ax3((0, 0, 0)))
bld = F.createIfcBuilding(GID(), None, b_.get("name"), None, None, bpl, None, None, "ELEMENT", None, None, None)
ZF = num(stg, "elevation")
stpl = F.createIfcLocalPlacement(bpl, ax3((0, 0, ZF)))
sto = F.createIfcBuildingStorey(GID(), None, stg.get("name"), None, None, stpl, None, None, "ELEMENT", ZF)
for a_, c_ in ((proj, site), (site, bld), (bld, sto)):
    F.createIfcRelAggregates(GID(), None, None, None, a_, [c_])


def blk(cx, cy, z, w, d, h):
    pr = F.createIfcRectangleProfileDef("AREA", None, F.createIfcAxis2Placement2D(p2(cx, cy), None), float(w), float(d))
    return F.createIfcExtrudedAreaSolid(pr, ax3((0, 0, z)), d3(0, 0, 1), float(h))


def cyl(r, z, h):
    pr = F.createIfcCircleProfileDef("AREA", None, F.createIfcAxis2Placement2D(p2(0, 0), None), float(r))
    return F.createIfcExtrudedAreaSolid(pr, ax3((0, 0, z)), d3(0, 0, 1), float(h))


def area_solid(outer, holes, z, h):
    pl = lambda r: F.createIfcPolyline([p2(x, y) for x, y in r])
    pr = (F.createIfcArbitraryProfileDefWithVoids("AREA", None, pl(outer), [pl(r) for r in holes]) if holes
          else F.createIfcArbitraryClosedProfileDef("AREA", None, pl(outer)))
    return F.createIfcExtrudedAreaSolid(pr, ax3((0, 0, z)), d3(0, 0, 1), float(h))


body_rep = lambda items: F.createIfcProductDefinitionShape(None, None, [F.createIfcShapeRepresentation(BODY, "Body", "SweptSolid", items)])
rmap = lambda items: F.createIfcRepresentationMap(ax3((0, 0, 0)), F.createIfcShapeRepresentation(BODY, "Body", "SweptSolid", items))
mapped = lambda rm: F.createIfcProductDefinitionShape(None, None, [F.createIfcShapeRepresentation(
    BODY, "Body", "MappedRepresentation", [F.createIfcMappedItem(rm, F.createIfcCartesianTransformationOperator3D(None, None, p3(0, 0, 0), None, None))])])


def pset(obj, name, props):
    v = []
    for k, x in props.items():
        val = F.create_entity("IfcBoolean", x) if isinstance(x, bool) else (F.create_entity("IfcReal", float(x)) if isinstance(x, (int, float)) else F.create_entity("IfcLabel", str(x)))
        v.append(F.createIfcPropertySingleValue(k, None, val, None))
    F.createIfcRelDefinesByProperties(GID(), None, None, None, [obj], F.createIfcPropertySet(GID(), None, name, None, v))


TYPES, OCC_BY_TYPE, CONTAIN, IN_SPACE = {}, {}, [], {}
MAT = {}


def material(name):
    if name not in MAT:
        MAT[name] = F.createIfcMaterial(name, None, None)
    return MAT[name]


# ---------------------------------------------------------------- types
for t in stg.find("WallTypes"):
    th = num(t, "thickness")
    ls = F.createIfcMaterialLayerSet([F.createIfcMaterialLayer(material(t.get("material")), th, None, None, None, None, None)], f"{t.get('id')} {th * 1000:.0f} mm", None)
    if t.get("function") == "glazing":                  # glass wall component -> curtain wall type
        wt = F.createIfcCurtainWallType(GID(), None, f"{t.get('id')} glazing {th * 1000:.0f} mm", t.get("evidence"), None, None, None, None, None, "NOTDEFINED")
    else:
        wt = F.createIfcWallType(GID(), None, f"{t.get('id')} {t.get('function')} {th * 1000:.0f} mm", t.get("evidence"), None, None, None, None, None,
                                 "PARTITIONING" if t.get("function") == "partition" else "STANDARD")
    F.createIfcRelAssociatesMaterial(GID(), None, None, None, [wt], ls)
    TYPES[t.get("id")] = dict(ent=wt, thickness=th, function=t.get("function"), layers=ls)


def door_geom(W, H):
    f = 0.05
    return [blk(-W / 2 + f / 2, 0, 0, f, 0.10, H), blk(W / 2 - f / 2, 0, 0, f, 0.10, H), blk(0, 0, H - f, W, 0.10, f),
            blk(0, 0, 0, W - 2 * f, 0.04, H - f)]


def window_geom(W, H):
    f = 0.06
    return [blk(0, 0, 0, W, 0.06, f), blk(0, 0, H - f, W, 0.06, f), blk(-W / 2 + f / 2, 0, f, f, 0.06, H - 2 * f),
            blk(W / 2 - f / 2, 0, f, f, 0.06, H - 2 * f), blk(0, 0, f, W - 2 * f, 0.01, H - 2 * f)]


for t in stg.find("DoorTypes"):
    W, H = num(t, "width"), num(t, "height"); rm = rmap(door_geom(W, H))
    TYPES[t.get("id")] = dict(ent=F.createIfcDoorType(GID(), None, f"{t.get('id')} door {W * 1000:.0f}x{H * 1000:.0f}", None, None, None, [rm], None, None,
                                                      "DOOR", t.get("operation"), False, None), W=W, H=H, rm=rm)
for t in stg.find("WindowTypes"):
    W, H = num(t, "width"), num(t, "height"); rm = rmap(window_geom(W, H))
    TYPES[t.get("id")] = dict(ent=F.createIfcWindowType(GID(), None, f"{t.get('id')} window {W * 1000:.0f}x{H * 1000:.0f}", None, None, None, [rm], None, None,
                                                        "WINDOW", t.get("operation"), False, None), W=W, H=H, sill=num(t, "sill"), rm=rm)
for t in stg.find("FloorTypes"):
    th = num(t, "thickness")
    st_ = F.createIfcSlabType(GID(), None, f"{t.get('id')} floor {th * 1000:.0f} mm", None, None, None, None, None, None, "FLOOR")
    F.createIfcRelAssociatesMaterial(GID(), None, None, None, [st_], F.createIfcMaterialLayerSet([F.createIfcMaterialLayer(material(t.get("material")), th, None, None, None, None, None)], None, None))
    TYPES[t.get("id")] = dict(ent=st_, thickness=th)
for t in stg.find("CeilingTypes"):
    th = num(t, "thickness")
    ct = F.createIfcCoveringType(GID(), None, f"{t.get('id')} ceiling {th * 1000:.0f} mm", None, None, None, None, None, None, "CEILING")
    F.createIfcRelAssociatesMaterial(GID(), None, None, None, [ct], material(t.get("material")))
    TYPES[t.get("id")] = dict(ent=ct, thickness=th)
H_STOREY = num(stg, "height")
for t in stg.find("ColumnTypes"):
    W, Dp = num(t, "width"), num(t, "depth"); rm = rmap([blk(0, 0, 0, W, Dp, H_STOREY)])
    TYPES[t.get("id")] = dict(ent=F.createIfcColumnType(GID(), None, f"{t.get('id')} column {W * 1000:.0f}x{Dp * 1000:.0f}", None, None, None, [rm], None, None, "COLUMN"), rm=rm)


def furniture_geom(cat, W, D_, H):
    """parametric templates, type frame: origin at the footprint centre on its base, +y = front"""
    if cat == "desk":
        t = 0.03
        return [blk(0, 0, H - t, W, D_, t), blk(-W / 2 + 0.015, 0, 0, 0.03, D_ - 0.06, H - t), blk(W / 2 - 0.015, 0, 0, 0.03, D_ - 0.06, H - t),
                blk(0, -D_ / 2 + 0.1, 0.3, W - 0.06, 0.02, H - t - 0.3)]
    if cat == "chair":
        seat = min(0.48, H * 0.47)
        return [cyl(min(W, D_) * 0.45, 0, 0.06), cyl(0.03, 0.06, seat - 0.06), blk(0, 0.02, seat, W * 0.8, D_ * 0.75, 0.07),
                blk(0, -D_ / 2 + 0.08, seat + 0.1, W * 0.75, 0.05, H - seat - 0.1)]
    if cat == "monitor":
        sh = min(H * 0.7, H - 0.12)
        return [blk(0, 0, 0, min(0.25, W * 0.5), min(0.2, D_), 0.02), blk(0, -0.02, 0.02, 0.05, 0.03, H - sh), blk(0, 0, H - sh, W, 0.03, sh)]
    if cat == "laptop":
        return [blk(0, 0.02, 0, W, D_ * 0.9, 0.02), blk(0, -D_ / 2 + 0.01, 0.02, W, 0.01, H - 0.02)]
    return [blk(0, 0, 0, W, D_, H)]


FCLS = {"desk": ("IfcFurniture", "IfcFurnitureType", "DESK"), "chair": ("IfcFurniture", "IfcFurnitureType", "CHAIR"),
        "pedestal": ("IfcFurniture", "IfcFurnitureType", "FILECABINET"), "shelf": ("IfcFurniture", "IfcFurnitureType", "SHELF"),
        "monitor": ("IfcAudioVisualAppliance", "IfcAudioVisualApplianceType", "DISPLAY"),
        "laptop": ("IfcCommunicationsAppliance", "IfcCommunicationsApplianceType", "COMPUTER")}
for t in stg.find("FurnitureTypes"):
    cat = t.get("category"); W, Dp, H = num(t, "width"), num(t, "depth"), num(t, "height")
    occ, tcls, pre = FCLS[cat]; rm = rmap(furniture_geom(cat, W, Dp, H))
    kw = dict(GlobalId=GID(), Name=f"{t.get('id')} {cat} {W * 1000:.0f}x{Dp * 1000:.0f}x{H * 1000:.0f}", RepresentationMaps=[rm], PredefinedType=pre)
    if tcls == "IfcFurnitureType":
        kw['AssemblyPlace'] = "NOTDEFINED"
    TYPES[t.get("id")] = dict(ent=F.create_entity(tcls, **kw), rm=rm, occ=occ, pre=pre, cat=cat)
F.createIfcRelDeclares(GID(), None, None, None, proj, [v['ent'] for v in TYPES.values()])


def typed(occ, tid):
    OCC_BY_TYPE.setdefault(tid, []).append(occ)


# ---------------------------------------------------------------- spaces
SPACE = {}
for s in stg.find("Spaces"):
    b = pts(s.find("Boundary").text); H = num(s, "height")
    sp = F.createIfcSpace(GID(), None, s.get("name"), None, None, F.createIfcLocalPlacement(stpl, ax3((0, 0, 0))),
                          body_rep([area_solid(b, [pts(h.text) for h in s.findall("Hole")], 0, H)]),
                          None, "ELEMENT", "INTERNAL", 0.0)
    pset(sp, "Pset_SpaceCommon", {"Reference": s.get("id")})
    if s.get("clippedByTile") == "True":
        sp.Description = "boundary touches the data extent"
    SPACE[s.get("id")] = sp
F.createIfcRelAggregates(GID(), None, None, None, sto, list(SPACE.values()))

# ---------------------------------------------------------------- walls
WALL = {}
for w in stg.find("Walls"):
    T = TYPES[w.get("typeID")]; a, b = pts(w.find("CentreLine").text)
    L = float(np.linalg.norm(b - a)); d = (b - a) / L; H = num(w, "height")
    pl = F.createIfcLocalPlacement(stpl, ax3((a[0], a[1], num(w, "baseOffset")), d))
    axis = F.createIfcShapeRepresentation(AXIS, "Axis", "Curve2D", [F.createIfcPolyline([p2(0, 0), p2(L, 0)])])
    body = F.createIfcShapeRepresentation(BODY, "Body", "SweptSolid", [blk(L / 2, 0, 0, L, T['thickness'], H)])
    if T['function'] == "glazing":
        e = F.createIfcCurtainWall(GID(), None, w.get("id"), None, None, pl, F.createIfcProductDefinitionShape(None, None, [axis, body]), None, "NOTDEFINED")
        pset(e, "Pset_CurtainWallCommon", {"Reference": w.get("typeID"), "IsExternal": False})
    else:
        e = F.createIfcWall(GID(), None, w.get("id"), None, None, pl, F.createIfcProductDefinitionShape(None, None, [axis, body]), None,
                            "PARTITIONING" if T['function'] == "partition" else "STANDARD")
        F.createIfcRelAssociatesMaterial(GID(), None, None, None, [e], F.createIfcMaterialLayerSetUsage(T['layers'], "AXIS2", "POSITIVE", -T['thickness'] / 2, None))
        pset(e, "Pset_WallCommon", {"Reference": w.get("typeID"), "IsExternal": T['function'] == "exterior"})
    typed(e, w.get("typeID")); CONTAIN.append(e)
    WALL[w.get("id")] = dict(ent=e, pl=pl, a=a, d=d, L=L, T=T['thickness'], side=w.get("spaceSide"), H=H)

# ---------------------------------------------------------------- doors / windows
for tag, cls in (("Doors", "IfcDoor"), ("Windows", "IfcWindow")):
    for o in stg.find(tag):
        T = TYPES[o.get("typeID")]; w = WALL[o.get("wallID")]; off = num(o, "offset")
        # instance parameters override the type size; the type geometry is stretched to them
        W_, H_ = (num(o, "width"), num(o, "height")) if o.get("width") else (T['W'], T['H'])
        sill = num(o, "sill") if o.get("sill") else T.get('sill', 0.0)
        opl = F.createIfcLocalPlacement(w['pl'], ax3((off, 0, sill)))
        op = F.createIfcOpeningElement(GID(), None, f"Opening {o.get('id')}", None, None, opl, body_rep([blk(0, 0, 0, W_, w['T'] + 0.1, H_)]), None, "OPENING")
        F.createIfcRelVoidsElement(GID(), None, None, None, w['ent'], op)
        y = 0.0
        if cls == "IfcWindow" and w['side'] in ("left", "right"):          # glass at the face of the space side
            y = (w['T'] / 2 - 0.03) * (1 if w['side'] == "left" else -1)
        fpl = F.createIfcLocalPlacement(opl, ax3((0, y, 0)))
        if abs(W_ - T['W']) > 1e-3 or abs(H_ - T['H']) > 1e-3:
            op3 = F.createIfcCartesianTransformationOperator3DnonUniform(None, None, p3(0, 0, 0), W_ / T['W'], None, 1.0, H_ / T['H'])
            rep_ = F.createIfcProductDefinitionShape(None, None, [F.createIfcShapeRepresentation(BODY, "Body", "MappedRepresentation",
                                                                                                [F.createIfcMappedItem(T['rm'], op3)])])
        else:
            rep_ = mapped(T['rm'])
        e = F.create_entity(cls, GlobalId=GID(), Name=o.get("id"), ObjectPlacement=fpl, Representation=rep_,
                            OverallHeight=H_, OverallWidth=W_, PredefinedType="DOOR" if cls == "IfcDoor" else "WINDOW")
        F.createIfcRelFillsElement(GID(), None, None, None, op, e)
        pset(e, "Pset_DoorCommon" if cls == "IfcDoor" else "Pset_WindowCommon", {"Reference": o.get("typeID"), "IsExternal": TYPES and w['side'] in ("left", "right")})
        typed(e, o.get("typeID")); CONTAIN.append(e)

# ---------------------------------------------------------------- floors, ceilings, columns
for f in stg.find("Floors"):
    T = TYPES[f.get("typeID")]
    e = F.createIfcSlab(GID(), None, f.get("id"), None, None, F.createIfcLocalPlacement(stpl, ax3((0, 0, -T['thickness']))),
                        body_rep([area_solid(pts(f.find("Boundary").text), [pts(h.text) for h in f.findall("Hole")], 0, T['thickness'])]), None, "FLOOR")
    typed(e, f.get("typeID")); CONTAIN.append(e)
for c in stg.find("Ceilings"):
    T = TYPES[c.get("typeID")]
    e = F.createIfcCovering(GID(), None, c.get("id"), None, None, F.createIfcLocalPlacement(stpl, ax3((0, 0, num(c, "height")))),
                            body_rep([area_solid(pts(c.find("Boundary").text), [pts(h.text) for h in c.findall("Hole")], 0, T['thickness'])]), None, "CEILING")
    typed(e, c.get("typeID")); IN_SPACE.setdefault(c.get("spaceID"), []).append(e)
for k in stg.find("Columns"):
    T = TYPES[k.get("typeID")]; x, y = map(float, k.get("location").split()); r = np.radians(num(k, "rotation"))
    e = F.createIfcColumn(GID(), None, k.get("id"), None, None, F.createIfcLocalPlacement(stpl, ax3((x, y, 0), (np.cos(r), np.sin(r)))), mapped(T['rm']), None, "COLUMN")
    typed(e, k.get("typeID")); CONTAIN.append(e)

# ---------------------------------------------------------------- furniture
for it in stg.find("Furniture"):
    T = TYPES[it.get("typeID")]; x, y = map(float, it.get("location").split()); r = np.radians(num(it, "rotation"))
    pl = F.createIfcLocalPlacement(stpl, ax3((x, y, num(it, "elevation")), (np.cos(r), np.sin(r))))
    e = F.create_entity(T['occ'], GlobalId=GID(), Name=it.get("id"), ObjectType=T['ent'].Name, ObjectPlacement=pl, Representation=mapped(T['rm']),
                        PredefinedType=T['pre'])
    W, Dp, H = map(float, it.get("measured").split())
    pset(e, "Scan_Measured", {"Reference": it.get("typeID"), "MeasuredWidth_m": W, "MeasuredDepth_m": Dp, "MeasuredHeight_m": H})
    typed(e, it.get("typeID"))
    sid = it.get("spaceID")
    if sid in SPACE:
        IN_SPACE.setdefault(sid, []).append(e)
    else:
        CONTAIN.append(e)

for tid, occ in OCC_BY_TYPE.items():
    F.createIfcRelDefinesByType(GID(), None, None, None, occ, TYPES[tid]['ent'])
F.createIfcRelContainedInSpatialStructure(GID(), None, None, None, CONTAIN, sto)
for sid, els in IN_SPACE.items():
    F.createIfcRelContainedInSpatialStructure(GID(), None, None, None, els, SPACE[sid])
F.write(IFC_COMPONENTS)
cnt = {t: len(F.by_type(t)) for t in ["IfcWallType", "IfcWall", "IfcCurtainWallType", "IfcCurtainWall", "IfcDoorType", "IfcDoor", "IfcWindowType", "IfcWindow", "IfcSlabType", "IfcSlab",
                                      "IfcCoveringType", "IfcCovering", "IfcSpace", "IfcColumnType", "IfcColumn", "IfcFurnitureType", "IfcFurniture",
                                      "IfcAudioVisualApplianceType", "IfcAudioVisualAppliance", "IfcCommunicationsApplianceType", "IfcCommunicationsAppliance"]}
print(f"written {IFC_COMPONENTS}\n" + "\n".join(f"  {k:<32s} {v}" for k, v in cnt.items()))
