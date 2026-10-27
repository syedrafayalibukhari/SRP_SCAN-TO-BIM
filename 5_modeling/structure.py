"""Structure: levels, dominant directions, wall and glazing faces, slabs.
  faces are detected separately on wall points and on window points (glazing = interior glass walls
  and glass facade); a glazing face pair may be only a few cm apart (glass panel), a lone glazing face gets 6 cm
  levels (Oesau 2014 / Cui 2019): z-histogram of horizontal-normal floor/ceiling points, 1 cm bins, peaks
  directions (Cui 2019, weak-Manhattan as Wei 2023): length-weighted histogram of horizontal wall normals mod 90 deg
  faces: for each direction family and each of its 4 normal directions, 1 cm histogram of the offset along the normal
         (Hough-style accumulation, cf. global clustering in Oesau 2014) -> peaks -> inlier band +-3 cm
         -> along-axis runs (split at gaps > GAP_SPLIT) -> faces with z-extent and support
  slabs: a face with normal -n at p and a face with normal +n at q (q - p in [0.04, 0.45] m, overlapping) form one
         wall [p, q] (measured thickness, unlike the fixed 0.22 m of Cui 2019); unpaired faces are single-sided walls
Output: stage1.json (levels, frame, faces, slabs), stage1_faces.png"""
import json
import numpy as np
from scipy.ndimage import maximum_filter1d, gaussian_filter1d
from common import *

log = Log(f"{OUT}/stage1_log.txt")
S = load_ply(SCAN)
P = np.c_[S['x'], S['y'], S['z']].astype(float); N = np.c_[S['nx'], S['ny'], S['nz']].astype(float); L = S['lab']
log(f"scan {len(S):,} points")

# ---------------------------------------------------------------- levels
def peaks_z(z, bin_=0.01, min_share=0.05):
    h, e = np.histogram(z, bins=np.arange(z.min() - 0.05, z.max() + 0.05, bin_))
    hs = gaussian_filter1d(h.astype(float), 1.5)
    pk = np.where((hs == maximum_filter1d(hs, 15)) & (hs > min_share * hs.max()))[0]
    return [(float(e[k] + bin_ / 2), float(hs[k])) for k in pk]


hor = np.abs(N[:, 2]) > 0.95
fl = peaks_z(P[hor & (L == FLOOR), 2]); ce = peaks_z(P[hor & (L == CEIL), 2])
Z_FLOOR = max(fl, key=lambda t: t[1])[0]
CEILS = sorted(z for z, _ in ce)
log(f"floor level {Z_FLOOR:.3f} m; ceiling levels {[round(z, 3) for z in CEILS]} m")

# ---------------------------------------------------------------- dominant directions
W = (L == WALL) & (np.abs(N[:, 2]) < 0.2)
WG = (L == WIN) & (np.abs(N[:, 2]) < 0.2)
# whole floor: directions from wall, window and partition points (the 28 deg wing is mostly glass and panels).
# A family = a local maximum of the 1-degree histogram (mod 90, circular) that is >= 25 % above the lowest bin within
# +-10 deg and holds >= 0.5 % of the points per degree (the wing is a small but distinct peak next to the main axis)
WD = (np.isin(L, [WALL, WIN, PART])) & (np.abs(N[:, 2]) < 0.2)
ang = np.degrees(np.arctan2(N[WD, 1], N[WD, 0])) % 90
h, _ = np.histogram(ang, bins=90, range=(0, 90)); h = h / WD.sum()
FAM = []
for k in np.argsort(-h):
    win = [h[(k + j) % 90] for j in range(-8, 9)]
    low = min(h[(k + j) % 90] for j in range(-10, 11))
    if h[k] < 0.005:
        break
    if h[k] < max(win) or h[k] < 1.25 * low:
        continue
    if all(min(abs(k + 0.5 - f), 90 - abs(k + 0.5 - f)) > 10 for f in FAM):
        c0 = k + 0.5; d = (ang - c0 + 45) % 90 - 45; m = np.abs(d) < 2
        FAM.append(float((c0 + d[m].mean()) % 90))
log(f"dominant directions (mod 90): {[round(f, 2) for f in FAM]} deg")

# ---------------------------------------------------------------- faces
GAP_SPLIT, BAND, MIN_LEN, MIN_PTS = 0.5, 0.03, 0.3, 300


# normals come from PCA and are oriented only roughly (virtual viewpoints), so faces are found on UNSIGNED
# offsets and each face's viewing side is decided by line of sight from the viewpoints (2 cm plan raster of wall,
# window and column points at 0.5-2.4 m)
VP = stations()[:, :2]
RPIX = 0.02
RX0, RY0 = P[:, 0].min() - 0.5, P[:, 1].min() - 0.5
BLK = np.zeros((int((P[:, 0].max() + 0.5 - RX0) / RPIX) + 1, int((P[:, 1].max() + 0.5 - RY0) / RPIX) + 1), bool)
_m = np.isin(L, [WALL, WIN, COL]) & (P[:, 2] > 0.5) & (P[:, 2] < 2.4)
_ij = ((P[_m, :2] - [RX0, RY0]) / RPIX).astype(int); BLK[_ij[:, 0], _ij[:, 1]] = True


def seen_from(q, n_side):
    """number of viewpoints with a free 2D line of sight to point q (which lies 15 cm in front of a face)"""
    k = 0
    for v in VP:
        if (v - q) @ n_side <= 0:
            continue
        L_ = np.linalg.norm(v - q); m_ = int(L_ / (RPIX / 2))
        t = np.linspace(0, 1, m_)[4:]                      # skip the first 4 cm next to q
        ij = ((q + np.outer(t, v - q) - [RX0, RY0]) / RPIX).astype(int)
        ij = ij[(ij[:, 0] >= 0) & (ij[:, 0] < BLK.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < BLK.shape[1])]
        k += not BLK[ij[:, 0], ij[:, 1]].any()
    return k


def detect_faces(FAM):
    FACES = []
    for kind, MASK in (("wall", W), ("glazing", WG)):
        for fam in FAM:
            for q in range(2):                                   # unsigned: two axes per family
                a = np.radians(fam + 90 * q); n = np.array([np.cos(a), np.sin(a)]); dvec = np.array([-n[1], n[0]])
                sel = MASK & (np.abs(N[:, :2] @ n) > np.cos(np.radians(15)))
                if sel.sum() < MIN_PTS:
                    continue
                off = P[sel, :2] @ n
                for o, _ in peaks_z(off, 0.01, 0.01):
                    inl = np.where(sel)[0][np.abs(off - o) < BAND]
                    if len(inl) < MIN_PTS:
                        continue
                    o = float(np.median(P[inl, :2] @ n)); t = P[inl, :2] @ dvec; order = np.argsort(t); ts = t[order]
                    cuts = np.where(np.diff(ts) > GAP_SPLIT)[0]
                    for seg in np.split(order, cuts + 1):
                        tt = t[seg]
                        if len(seg) < MIN_PTS or tt.max() - tt.min() < MIN_LEN:
                            continue
                        z = P[inl[seg], 2]; Q = P[inl[seg], :2] - P[inl[seg], :2].mean(0)
                        pca = np.linalg.svd(Q, full_matrices=False)[2][0]
                        t0, t1 = float(np.percentile(tt, 0.2)), float(np.percentile(tt, 99.8))
                        samp = [n * o + dvec * tq for tq in np.linspace(t0, t1, 7)[1:-1]]
                        vis = {sg: sum(seen_from(c + sg * 0.15 * n, sg * n) > 0 for c in samp) for sg in (1, -1)}
                        sides = [sg for sg in (1, -1) if vis[sg] >= 2] or [max(vis, key=vis.get)]
                        for sg in sides:                          # a face seen from both sides is a thin element: two faces
                            nn = sg * n
                            FACES.append(dict(fam=fam, n=nn.tolist(), d=[-nn[1], nn[0]], o=float(sg * o),
                                              t0=float(sg * t0) if sg > 0 else float(-t1), t1=float(sg * t1) if sg > 0 else float(-t0),
                                              z0=float(np.percentile(z, 1)), z1=float(np.percentile(z, 99)), npts=int(len(seg)),
                                              rms=float(np.sqrt(np.mean((P[inl[seg], :2] @ n - o) ** 2))), pca=pca.tolist(), kind=kind,
                                              seen=dict(plus=vis[1], minus=vis[-1])))
    return FACES


FACES = detect_faces(FAM)
for it in range(2):        # refine each direction family from its face inliers (PCA angle, length-weighted)
    FAM = [float((f + np.average([((np.degrees(np.arctan2(x['pca'][1], x['pca'][0])) - f + 45) % 90 - 45) for x in FACES if x['fam'] == f],
                                 weights=[x['t1'] - x['t0'] for x in FACES if x['fam'] == f])) % 90) for f in FAM]
    FACES = detect_faces(FAM)
log(f"refined directions (mod 90): {[round(f, 3) for f in FAM]} deg")
log(f"wall faces: {len(FACES)} (band +-{BAND} m, split at gaps > {GAP_SPLIT} m, length >= {MIN_LEN} m)")

# faces no scan position can see: face away from a seen parallel face 4-45 cm away
for f in FACES:
    if f['seen']['plus'] + f['seen']['minus'] > 0:
        continue
    n = np.array(f['n'])
    for g in FACES:
        if g is f or g['kind'] != f['kind'] or g['seen']['plus'] + g['seen']['minus'] == 0 or abs(np.dot(g['n'], n)) < 0.99:
            continue
        og = g['o'] * np.dot(g['n'], n)                          # g's offset along f's axis
        tg = sorted((g['t0'], g['t1'])) if np.dot(g['d'], f['d']) > 0 else sorted((-g['t1'], -g['t0']))
        if 0.04 <= abs(f['o'] - og) <= 0.45 and min(tg[1], f['t1']) - max(tg[0], f['t0']) > 0.3:
            if np.sign(f['o'] - og) != 1:                        # g lies on f's + side -> f must face -n
                f.update(n=(-n).tolist(), d=[n[1], -n[0]], o=-f['o'], t0=-f['t1'], t1=-f['t0'])
            f['seen']['inferred'] = True
            break
# ---------------------------------------------------------------- slabs (pair opposite faces)
used = set(); SLABS = []
cand = []
for i, f in enumerate(FACES):
    for j, g in enumerate(FACES):
        if i == j or f['fam'] != g['fam'] or f['kind'] != g['kind'] or np.dot(f['n'], g['n']) > -0.99:
            continue
        # f has normal -m at offset p (along m), g has normal +m at offset q, q > p
        m = np.array(g['n']); p = -f['o']; qo = g['o']
        th = qo - p
        if not ((0.04 <= th <= 0.45) if g['kind'] == "wall" else (-0.03 <= th <= 0.15)):
            continue
        # along-axis overlap in g's frame
        tf = sorted((-f['t1'], -f['t0'])) if np.dot(f['d'], g['d']) < 0 else (f['t0'], f['t1'])
        ov = min(tf[1], g['t1']) - max(tf[0], g['t0'])
        if ov < 0.5 * min(tf[1] - tf[0], g['t1'] - g['t0']):
            continue
        cand.append((th, i, j, tf, ov))
for th, i, j, tf, ov in sorted(cand):
    if i in used or j in used:
        continue
    f, g = FACES[i], FACES[j]
    used |= {i, j}
    o0, o1 = -f['o'], g['o']
    if g['kind'] == "glazing" and o1 - o0 < 0.06:          # glass: at least 6 cm (frame), centred on the two faces
        c = (o0 + o1) / 2; o0, o1 = c - 0.03, c + 0.03
    SLABS.append(dict(fam=g['fam'], n=g['n'], d=g['d'], o0=o0, o1=o1, t0=min(tf[0], g['t0']), t1=max(tf[1], g['t1']),
                      faces=[i, j], paired=True, kind=g['kind']))
for i, f in enumerate(FACES):
    if i not in used and (f['t1'] - f['t0']) >= (1.0 if f['kind'] == "glazing" else 0.5):   # short lone faces = edge artefacts
        # single face: solid lies behind the face (opposite to the normal); thickness decided in stage 2
        SLABS.append(dict(fam=f['fam'], n=f['n'], d=f['d'], o0=None, o1=f['o'], t0=f['t0'], t1=f['t1'], faces=[i], paired=False, kind=f['kind']))
# drop: lone faces lying on a face of a paired slab (second copy of a face seen from both sides), short glazing slabs
def on_paired(sl):
    n = np.array(sl['n'])
    for q in SLABS:
        if q['paired'] and abs(np.dot(q['n'], n)) > 0.99:
            sg = np.dot(q['n'], n)
            if min(abs(sl['o1'] - sg * q['o0']), abs(sl['o1'] - sg * q['o1'])) < 0.02:
                return True
    return False
SLABS = [sl for sl in SLABS if not (not sl['paired'] and on_paired(sl)) and not (sl['kind'] == "glazing" and sl['t1'] - sl['t0'] < 1.0)]
th = [s['o1'] - s['o0'] for s in SLABS if s['paired'] and s['kind'] == 'wall'] or [0.0]
log(f"wall slabs: {sum(s['paired'] for s in SLABS)} paired (thickness median {np.median(th):.3f} m, range {min(th):.3f}-{max(th):.3f}), "
    f"{sum(not s['paired'] for s in SLABS)} single-face")
json.dump(dict(Z_FLOOR=Z_FLOOR, CEILS=CEILS, FAM=FAM, FACES=FACES, SLABS=SLABS), open(f"{OUT}/stage1.json", "w"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16, 11))
s = np.random.default_rng(0).choice(len(P), 400000, replace=False); s = s[(P[s, 2] > 0.3) & (P[s, 2] < 2.2)]
ax.scatter(P[s, 0], P[s, 1], s=0.03, c="0.75")
for k, sl in enumerate(SLABS):
    n, d = np.array(sl['n']), np.array(sl['d'])
    for o in ([sl['o0'], sl['o1']] if sl['paired'] else [sl['o1']]):
        a, b = n * o + d * sl['t0'], n * o + d * sl['t1']
        ax.plot([a[0], b[0]], [a[1], b[1]], c="#d62728" if sl['paired'] else "#1f77b4", lw=1.5)
ax.plot([], [], c="#d62728", label="paired faces (wall with measured thickness)"); ax.plot([], [], c="#1f77b4", label="single face")
ax.legend(); ax.set_aspect("equal"); ax.set_title("stage 1: wall faces")
plt.tight_layout(); plt.savefig(f"{OUT}/stage1_faces.png", dpi=80)
