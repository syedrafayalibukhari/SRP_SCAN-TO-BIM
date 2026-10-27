"""Columns and partitions.
  columns (Tang 2022 / Mahmoud 2024 style): column-labelled points at 0.3-2.4 m, DBSCAN in 2D, box aligned to the nearest
      dominant direction from the 1st/99th percentiles of the points (hollow outlines -> outer box), height floor-ceiling
  partitions: same face accumulation as the walls (stage 1) on partition-labelled points, band +-2 cm, runs split at gaps
      > 0.15 m; opposite faces 0.02-0.15 m apart are paired (measured thickness), single faces get the median thickness;
      height = 99th percentile z of the run, snapped to the nearest scene height reference within 5 cm
      (reference heights = peaks of the run-height histogram, Wei 2023)
Output: stage4.json, stage4_plan.png"""
import json
import numpy as np
from scipy.ndimage import maximum_filter1d, gaussian_filter1d
from sklearn.cluster import DBSCAN
from common import *

log = Log(f"{OUT}/stage4_log.txt")
S1 = json.load(open(f"{OUT}/stage1.json")); S2 = json.load(open(f"{OUT}/stage2.json"))
S = load_ply(SCAN)
P = np.c_[S['x'], S['y'], S['z']].astype(float); N = np.c_[S['nx'], S['ny'], S['nz']].astype(float); L = S['lab']
ZF = S1['Z_FLOOR']; FAM = S1['FAM']; ZC = max(r['z_ceil'] for r in S2['ROOMS'])

# ---------------------------------------------------------------- columns
m = (L == COL) & (P[:, 2] > ZF + 0.3) & (P[:, 2] < ZF + 2.4)
C = P[m, :2]
cl = DBSCAN(eps=0.15, min_samples=20).fit_predict(C)
COLS = []
for k in sorted(set(cl) - {-1}):
    q = C[cl == k]
    if len(q) < 200:
        continue
    a = np.radians(FAM[0]); d = np.array([np.cos(a), np.sin(a)]); n = np.array([-d[1], d[0]])
    u, v = q @ d, q @ n
    u0, u1 = np.percentile(u, [1, 99]); v0, v1 = np.percentile(v, [1, 99])
    if max(u1 - u0, v1 - v0) > 2.0:
        continue
    c = d * (u0 + u1) / 2 + n * (v0 + v1) / 2
    COLS.append(dict(c=c.tolist(), d=d.tolist(), a=float(u1 - u0), b=float(v1 - v0), npts=int(len(q))))
log(f"columns: {len(COLS)}; sizes " + ", ".join(f"{c['a']:.2f}x{c['b']:.2f}" for c in COLS))


# ---------------------------------------------------------------- partitions
def peaks(x, bin_=0.01, min_share=0.01):
    h, e = np.histogram(x, bins=np.arange(x.min() - 0.05, x.max() + 0.05, bin_))
    hs = gaussian_filter1d(h.astype(float), 1.0)
    pk = np.where((hs == maximum_filter1d(hs, 5)) & (hs > min_share * hs.max()))[0]
    return [float(e[k] + bin_ / 2) for k in pk]


Wp = (L == PART) & (np.abs(N[:, 2]) < 0.3) & (P[:, 2] > ZF + 0.1)
FACES = []
for fam in FAM:
    for q in range(2):                       # normal signs unreliable -> unsigned, two axes per family
        a = np.radians(fam + 90 * q); n = np.array([np.cos(a), np.sin(a)]); d = np.array([-n[1], n[0]])
        sel = np.where(Wp & (np.abs(N[:, :2] @ n) > np.cos(np.radians(20))))[0]
        if len(sel) < 100:
            continue
        off = P[sel, :2] @ n
        for o in peaks(off):
            inl = sel[np.abs(off - o) < 0.02]
            if len(inl) < 100:
                continue
            t = P[inl, :2] @ d; order = np.argsort(t)
            for seg in np.split(order, np.where(np.diff(t[order]) > 0.15)[0] + 1):
                tt = t[seg]
                if len(seg) < 100 or tt.max() - tt.min() < 0.3:
                    continue
                pz = P[inl[seg], 2]
                FACES.append(dict(fam=fam, n=n, d=d, o=float(np.median(P[inl[seg], :2] @ n)), t0=float(np.percentile(tt, 0.5)),
                                  t1=float(np.percentile(tt, 99.5)), ztop=float(np.percentile(pz, 99)), npts=int(len(seg))))
used = set(); PARTS = []
cand = []
for i, f in enumerate(FACES):                # two faces on the same axis, 2-15 cm apart, overlapping -> one panel
    for j, g in enumerate(FACES):
        if j <= i or f['fam'] != g['fam'] or f['n'] @ g['n'] < 0.99:
            continue
        th = abs(g['o'] - f['o'])
        if not (0.02 <= th <= 0.15):
            continue
        ov = min(f['t1'], g['t1']) - max(f['t0'], g['t0'])
        if ov < 0.5 * min(f['t1'] - f['t0'], g['t1'] - g['t0']):
            continue
        cand.append((th, i, j))
for th, i, j in sorted(cand):
    if i in used or j in used:
        continue
    f, g = FACES[i], FACES[j]; used |= {i, j}
    PARTS.append(dict(fam=g['fam'], n=g['n'], d=g['d'], o=(g['o'] + f['o']) / 2, T=th, t0=min(f['t0'], g['t0']), t1=max(f['t1'], g['t1']),
                      ztop=max(f['ztop'], g['ztop']), paired=True))
Tmed = float(np.median([p['T'] for p in PARTS])) if PARTS else 0.05
for i, f in enumerate(FACES):
    if i not in used:
        PARTS.append(dict(fam=f['fam'], n=f['n'], d=f['d'], o=f['o'] - Tmed / 2, T=Tmed, t0=f['t0'], t1=f['t1'], ztop=f['ztop'], paired=False))
# heights: snap to reference heights
hz = np.array([p['ztop'] - ZF for p in PARTS]); hw = np.array([p['t1'] - p['t0'] for p in PARTS])
hh, he = np.histogram(hz, bins=np.arange(0, 3, 0.02), weights=hw)
REF = [float(he[k] + 0.01) for k in np.where((hh == maximum_filter1d(hh, 5)) & (hh > 0.1 * hh.max()))[0]]
for p in PARTS:
    h = p['ztop'] - ZF; r = min(REF, key=lambda x: abs(x - h))
    p['H'] = r if abs(r - h) <= 0.05 else h
    p['n'], p['d'] = p['n'].tolist(), p['d'].tolist()
log(f"partition faces {len(FACES)} -> panels {len(PARTS)} ({sum(p['paired'] for p in PARTS)} paired, thickness median {Tmed:.3f} m), "
    f"total length {sum(hw):.1f} m; reference heights {[round(r, 2) for r in REF]} m")
json.dump(dict(COLS=COLS, PARTS=PARTS, REF=REF), open(f"{OUT}/stage4.json", "w"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MP
fig, ax = plt.subplots(figsize=(16, 11))
s = np.where(L == PART)[0][::20]; ax.scatter(P[s, 0], P[s, 1], s=0.05, c="0.7")
for p in PARTS:
    n, d = np.array(p['n']), np.array(p['d'])
    q = [n * (p['o'] + sg * p['T'] / 2) + d * t for t, sg in ((p['t0'], -1), (p['t1'], -1), (p['t1'], 1), (p['t0'], 1))]
    ax.add_patch(MP(np.array(q), fc="#2ca02c" if p['paired'] else "#98df8a", ec="k", lw=0.3))
for c in COLS:
    ax.add_patch(MP(np.array([np.array(c['c']) + sx * np.array(c['d']) * c['a'] / 2 + sy * np.array([-c['d'][1], c['d'][0]]) * c['b'] / 2
                              for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]), fc="#d62728"))
ax.set_aspect("equal"); ax.autoscale_view(); ax.set_title(f"stage 4: {len(PARTS)} partition panels, {len(COLS)} columns")
plt.tight_layout(); plt.savefig(f"{OUT}/stage4_plan.png", dpi=80)
