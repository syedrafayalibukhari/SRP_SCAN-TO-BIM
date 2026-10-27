"""Doors and windows.
  per-wall image (Previtali 2018), 5 cm pixels in (t along the wall, z): each pixel is
     wall      - wall points within the slab (+-3 cm of its faces)
     door/win  - door / window labelled points within 0.3 m / 0.15 m of the slab (semantic evidence, Tang 2022, Mahmoud 2024)
     empty     - no point, but a scanner station has a free line of sight to the pixel (ray through a 5 cm voxel grid of all
                 other points): the scanner looked there and got no wall return -> opening or glass without return
     occluded  - no point and every station's ray is blocked -> treated as wall
  openings = connected components of (empty | door | window) after a 2-pixel closing; rectangle = bounding box
  type: door if it reaches the floor (bottom <= 0.15 m) and is >= 1.8 m high, else window if bottom >= 0.3 m;
        remaining small/irregular components are rejected (width 0.5-3.0 m, height >= 0.4 m)
Output: stage3.json, stage3_wall_images.png"""
import json
import numpy as np
from scipy import ndimage
from common import *

log = Log(f"{OUT}/stage3_log.txt")
S1 = json.load(open(f"{OUT}/stage1.json")); S2 = json.load(open(f"{OUT}/stage2.json"))
S = load_ply(SCAN)
P = np.c_[S['x'], S['y'], S['z']].astype(float); L = S['lab']
ST = stations(); ZF = S1['Z_FLOOR']
PX, VOX = 0.05, 0.05
lo = P.min(0) - 0.5
G = np.zeros(tuple(((P.max(0) + 0.5 - lo) / VOX).astype(int) + 1), bool)
SLABS = S2['SLABS']
OPEN = []; IMGS = []
for si, sl in enumerate(SLABS):
    if sl.get('kind') == "glazing":         # glass walls are curtain walls, not hosts of openings
        continue
    n, d = np.array(sl['n']), np.array(sl['d'])
    o0, o1 = sl['o0'], sl['o1']                         # slab between offsets o0 < o1 along n
    zc = max(r['z_ceil'] for r in S2['ROOMS'])
    t0, t1 = sl['t0'], sl['t1']
    nt, nz = int((t1 - t0) / PX) + 1, int((zc - ZF) / PX) + 1
    off = P[:, :2] @ n; tt = P[:, :2] @ d
    near = (off > o0 - 0.35) & (off < o1 + 0.35) & (tt > t0 - 0.05) & (tt < t1 + 0.05) & (P[:, 2] > ZF - 0.05) & (P[:, 2] < zc + 0.05)
    idx = np.where(near)[0]
    it = np.clip(((tt[idx] - t0) / PX).astype(int), 0, nt - 1); iz = np.clip(((P[idx, 2] - ZF) / PX).astype(int), 0, nz - 1)
    o_ = off[idx]
    inslab = (o_ > o0 - 0.03) & (o_ < o1 + 0.03)
    W = np.zeros((nt, nz), bool); Dm = np.zeros((nt, nz), bool); Wn = np.zeros((nt, nz), bool)
    m = inslab & (L[idx] == WALL); W[it[m], iz[m]] = True
    m = (L[idx] == DOOR) & (o_ > o0 - 0.3) & (o_ < o1 + 0.3); Dm[it[m], iz[m]] = True
    m = (L[idx] == WIN) & (o_ > o0 - 0.15) & (o_ < o1 + 0.15); Wn[it[m], iz[m]] = True
    # occupancy grid without this wall's own neighbourhood
    G[:] = False
    keep = np.ones(len(P), bool); keep[idx[(o_ > o0 - 0.12) & (o_ < o1 + 0.12)]] = False
    v = ((P[keep] - lo) / VOX).astype(int); G[v[:, 0], v[:, 1], v[:, 2]] = True
    # candidate pixels: no evidence at all
    cand = np.argwhere(~(W | Dm | Wn))
    EMPTY = np.zeros((nt, nz), bool); OCC = np.zeros((nt, nz), bool)
    if len(cand):
        # stations on either side, nearest 12 per side
        so = ST[:, :2] @ n
        sts = [s for side in (so < o0, so > o1) for s in ST[side][np.argsort(np.abs((ST[side][:, :2] @ d) - (t0 + t1) / 2) + np.abs(so[side] - o1))[:12]]]
        ctr = np.c_[t0 + (cand[:, 0] + 0.5) * PX, ZF + (cand[:, 1] + 0.5) * PX]
        for k0 in range(0, len(cand), 4000):
            c = ctr[k0:k0 + 4000]; seen = np.zeros(len(c), bool)
            for s in sts:
                side_o = o0 - 0.12 if s[:2] @ n < o0 else o1 + 0.12           # stop the ray 12 cm before the slab
                tgt = np.c_[(c[:, 0:1] * d + side_o * n), c[:, 1]]
                vec = tgt - s; Lr = np.linalg.norm(vec, axis=1)
                ns_ = int(Lr.max() / (VOX / 2)) + 1
                fr = np.linspace(0, 1, ns_)
                pts = s + vec[:, None, :] * fr[None, :, None]
                pts = pts[:, int(0.3 / (VOX / 2) / max(ns_, 1) * ns_):]                     # skip first 30 cm (scanner)
                vi = ((pts - lo) / VOX).astype(int)
                vi = np.clip(vi, 0, np.array(G.shape) - 1)
                blocked = G[vi[..., 0], vi[..., 1], vi[..., 2]].any(1)
                seen |= ~blocked
            EMPTY[cand[k0:k0 + 4000, 0][seen], cand[k0:k0 + 4000, 1][seen]] = True
            OCC[cand[k0:k0 + 4000, 0][~seen], cand[k0:k0 + 4000, 1][~seen]] = True
    OPENMASK = ndimage.binary_closing(EMPTY | Dm | Wn, iterations=2) & ~ndimage.binary_erosion(W, iterations=1)
    lab_, nlab = ndimage.label(OPENMASK)
    for k in range(1, nlab + 1):
        ii, zz = np.where(lab_ == k)
        a, b = t0 + ii.min() * PX, t0 + (ii.max() + 1) * PX; zb, zt = ii.min(), 0
        zb, zt = zz.min() * PX, (zz.max() + 1) * PX
        w_, h_ = b - a, zt - zb
        nd, nw, ne = Dm[ii, zz].sum(), Wn[ii, zz].sum(), EMPTY[ii, zz].sum()
        fill = len(ii) / ((ii.max() - ii.min() + 1) * (zz.max() - zz.min() + 1))
        # door labels dominate and everything below the component is occluded / unexplained -> extend to the floor
        below = (OCC | Dm | EMPTY)[ii.min():ii.max() + 1, :zz.min()]
        if nd > nw and zb > 0.15 and below.size and below.mean() >= 0.8:
            zb, h_ = 0.0, zt
        if zb <= 0.15:
            zb, h_ = 0.0, zt                           # openings that reach the floor start at the floor
        if zb == 0.0 and h_ >= 1.8 and zt < zc - ZF - 0.1:
            kind = "door"
        elif zb >= 0.3 and h_ >= 0.4 and zt < zc - ZF - 0.1:
            kind = "window"
        else:
            continue
        if not (0.5 <= w_ <= 3.0) or fill < 0.6:
            continue
        if kind == "window" and nw + nd < ne and h_ < 0.8:     # low ray-cast-only gaps are shadows behind desks/clutter
            continue
        inc = (lab_[it, iz] == k) & np.isin(L[idx], [WIN, DOOR]) & (o_ > o0 - 0.3) & (o_ < o1 + 0.3)
        leaf_o = float(np.median(o_[inc])) if inc.sum() >= 50 else None      # observed glass / leaf plane
        OPEN.append(dict(slab=si, kind=kind, t0=a, t1=b, z0=zb, z1=zt, width=w_, height=h_, fill=fill, leaf_o=leaf_o,
                         evidence=dict(door_px=int(nd), window_px=int(nw), empty_px=int(ne))))
    IMGS.append((si, W, Dm, Wn, EMPTY, OCC, t0, ZF))
log(f"openings: {sum(o['kind'] == 'door' for o in OPEN)} doors, {sum(o['kind'] == 'window' for o in OPEN)} windows on {len(SLABS)} walls")
for o in OPEN:
    e = o['evidence']; src = "labels" if e['door_px'] + e['window_px'] > e['empty_px'] else "ray casting (empty)"
    log(f"  wall {o['slab']:2d} {o['kind']:<6s} t {o['t0']:6.2f}-{o['t1']:6.2f}  z {o['z0']:.2f}-{o['z1']:.2f}  {o['width']:.2f} x {o['height']:.2f} m  "
        f"fill {o['fill']:.2f}  mainly from {src}")
json.dump(dict(OPEN=OPEN), open(f"{OUT}/stage3.json", "w"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
big = [x for x in IMGS if x[1].shape[0] * PX > 1]
if not big:
    raise SystemExit(0)
fig, ax = plt.subplots(len(big), 1, figsize=(18, 1.6 * len(big)))
for a, (si, W, Dm, Wn, EM, OC, t0, ZF_) in zip(np.atleast_1d(ax), big):
    img = np.zeros(W.shape + (3,)) + 1
    img[OC] = [0.75, 0.75, 0.75]; img[W] = [0.35, 0.35, 0.35]; img[EM] = [1.0, 0.85, 0.3]; img[Wn] = [0.3, 0.6, 1.0]; img[Dm] = [0.2, 0.7, 0.2]
    a.imshow(img.transpose(1, 0, 2), origin="lower", extent=[t0, t0 + W.shape[0] * PX, 0, W.shape[1] * PX], aspect="equal")
    for o in OPEN:
        if o['slab'] == si:
            a.add_patch(Rectangle((o['t0'], o['z0']), o['width'], o['height'], fc="none", ec="r" if o['kind'] == "door" else "m", lw=2))
    a.set_ylabel(f"wall {si}"); a.set_yticks([])
plt.suptitle("wall images: dark = wall points, light grey = occluded, yellow = seen empty, blue = window labels, green = door labels; boxes = openings")
plt.tight_layout(); plt.savefig(f"{OUT}/stage3_wall_images.png", dpi=60)
