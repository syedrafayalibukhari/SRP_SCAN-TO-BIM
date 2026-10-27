"""Corner clean-up applied to the element list before overlap resolution.
Elements are dicts with axis d, normal n, interval [t0, t1] along d, offset o along n, thickness T, kind.
  1. direction snapping: axes within SNAP_DEG of a dominant direction (or its perpendicular) are rotated onto it
  2. junctions between non-parallel elements, closest first, each element end used once:
       L  (intersection of the centre lines near an end of both): the higher-priority element runs to the outer face,
          the other stops at its inner face -> butt corner, no gap, no overshoot
       T  (near an end of one, inside the other): the end is moved to the near face of the through element
  3. collinear gaps: two ends of same-kind, collinear elements closer than GAP_COL are joined at the midpoint
"""
import numpy as np

SNAP_DEG = 5.0
TOL = 0.35          # m, largest extension/trim of an end
GAP_COL = 0.15      # m, collinear gaps closed
MIN_ANG = 30.0      # deg, junction only between elements at least this far from parallel


def endpoints(x):
    return x['d'] * x['t0'] + x['n'] * x['o'], x['d'] * x['t1'] + x['n'] * x['o']


def set_axis(x, d, p0, p1):
    d = d / np.linalg.norm(d); n = np.array([-d[1], d[0]])
    if (p1 - p0) @ d < 0:
        p0, p1 = p1, p0
    c = (p0 + p1) / 2; o = float(c @ n)
    x.update(d=d, n=n, o=o, t0=float(p0 @ d), t1=float(p1 @ d))


def dominant_directions(EL, min_share=0.05):
    ang = np.array([np.degrees(np.arctan2(x['d'][1], x['d'][0])) % 90 for x in EL])
    w = np.array([x['t1'] - x['t0'] for x in EL])
    h, _ = np.histogram(ang, bins=90, range=(0, 90), weights=w)
    hs = h + np.roll(h, 1) + np.roll(h, -1)                       # 3-degree window, circular mod 90
    peaks = []
    for k in np.argsort(-hs):
        if hs[k] < min_share * w.sum():
            break
        if all(min(abs(k - p), 90 - abs(k - p)) > 2 * SNAP_DEG for p in peaks):
            peaks.append(k)
    out = []
    for p in peaks:                                                 # refine: weighted mean of angles near the peak
        dd = (ang - (p + 0.5) + 45) % 90 - 45; m = np.abs(dd) < 2
        out.append(float((p + 0.5 + np.average(dd[m], weights=w[m])) % 90) if m.any() else p + 0.5)
    return out


def snap_directions(EL, doms, log):
    k = 0
    for x in EL:
        if x['kind'] == "column":
            continue
        a = np.degrees(np.arctan2(x['d'][1], x['d'][0]))
        best = None
        for D in doms:
            r = (a - D + 45) % 90 - 45
            if abs(r) <= SNAP_DEG and (best is None or abs(r) < abs(best)):
                best = r
        if best is not None and abs(best) > 1e-6:
            p0, p1 = endpoints(x); c = (p0 + p1) / 2; L = x['t1'] - x['t0']
            a2 = np.radians(a - best); d = np.array([np.cos(a2), np.sin(a2)])
            set_axis(x, d, c - d * L / 2, c + d * L / 2); k += 1
    log(f"direction snapping: dominant directions {[round(v, 1) for v in doms]} deg (mod 90); {k} elements rotated (<= {SNAP_DEG} deg)")


def line_x(a, b):
    """intersection of the centre lines of a and b -> (point, t along a, t along b, sin angle)"""
    s = a['d'][0] * b['d'][1] - a['d'][1] * b['d'][0]
    pa = a['n'] * a['o']; pb = b['n'] * b['o']
    ta = ((pb - pa)[0] * b['d'][1] - (pb - pa)[1] * b['d'][0]) / s
    X = pa + ta * a['d']
    return X, float(X @ a['d']), float(X @ b['d']), abs(s)


def join(EL, PRIO, log):
    idx = [i for i, x in enumerate(EL) if x['kind'] != "column"]
    cand = []
    for ii, i in enumerate(idx):
        a = EL[i]
        for j in idx[ii + 1:]:
            b = EL[j]
            if abs(a['d'] @ b['d']) > np.cos(np.radians(MIN_ANG)):
                continue
            X, ta, tb, s = line_x(a, b)
            ea = min((abs(ta - a['t0']), 0), (abs(ta - a['t1']), 1))   # (distance, end) for a
            eb = min((abs(tb - b['t0']), 0), (abs(tb - b['t1']), 1))
            ina = a['t0'] + TOL < ta < a['t1'] - TOL; inb = b['t0'] + TOL < tb < b['t1'] - TOL
            if ea[0] <= TOL and eb[0] <= TOL:
                cand.append((ea[0] + eb[0], "L", i, ea[1], j, eb[1], ta, tb, s))
            elif ea[0] <= TOL and inb:
                cand.append((ea[0], "T", i, ea[1], j, None, ta, tb, s))
            elif eb[0] <= TOL and ina:
                cand.append((eb[0], "T", j, eb[1], i, None, tb, ta, s))
    used = set(); nL = nT = 0
    for gap, typ, i, ei, j, ej, ti, tj, s in sorted(cand, key=lambda c: c[0]):
        if (i, ei) in used or (typ == "L" and (j, ej) in used):
            continue
        a, b = EL[i], EL[j]
        # end of a lies on the side sa (+1: t1 end, -1: t0 end); the body of a extends from X in direction -sa
        sa = 1 if ei == 1 else -1
        if typ == "T":
            new = ti - sa * (b['T'] / 2) / s                           # stop at the near face of b
            if abs(new - (a['t1'] if ei else a['t0'])) > TOL or (a['t1'] - a['t0']) < 2 * TOL and abs(new - ti) > a['t1'] - a['t0']:
                continue
            a['t1' if ei else 't0'] = new; used.add((i, ei)); nT += 1
        else:
            sb = 1 if ej == 1 else -1
            if PRIO[a['kind']] <= PRIO[b['kind']] and not (PRIO[a['kind']] == PRIO[b['kind']] and a['t1'] - a['t0'] < b['t1'] - b['t0']):
                m, sm, tm, em, k, sk, tk, ek = a, sa, ti, ei, b, sb, tj, ej
            else:
                m, sm, tm, em, k, sk, tk, ek = b, sb, tj, ej, a, sa, ti, ei
            m['t1' if em else 't0'] = tm + sm * (k['T'] / 2) / s        # master: to the outer face of the slave
            k['t1' if ek else 't0'] = tk - sk * (m['T'] / 2) / s        # slave: to the inner face of the master
            used.add((i, ei)); used.add((j, ej)); nL += 1
    # collinear gaps
    nC = 0
    for ii, i in enumerate(idx):
        a = EL[i]
        for j in idx[ii + 1:]:
            b = EL[j]
            if a['kind'] != b['kind'] or abs(a['d'] @ b['d']) < np.cos(np.radians(2)):
                continue
            sg = 1 if a['d'] @ b['d'] > 0 else -1
            if abs(a['o'] - sg * b['o']) > 0.03:
                continue
            bt = sorted((sg * b['t0'], sg * b['t1']))
            for ea, eb in ((1, 0), (0, 1)):
                ga = a['t1'] if ea else a['t0']; gb = bt[eb]
                gap = (gb - ga) if ea else (ga - gb)
                if 0 < gap <= GAP_COL and (i, ea) not in used:
                    mid = (ga + gb) / 2
                    a['t1' if ea else 't0'] = mid
                    if sg > 0:
                        b['t0' if eb == 0 else 't1'] = mid
                    else:
                        b['t1' if eb == 0 else 't0'] = -mid
                    nC += 1
    log(f"junctions: {nL} L-corners, {nT} T-junctions joined, {nC} collinear gaps closed (tolerance {TOL} m)")


def remove_stubs(EL, log, touch=0.003):
    """overshoot remnants: a piece shorter than TOL, split off the same source element, with one end face touching
    another element and the other end free, is deleted (the end that overshot a crossing element)"""
    from shapely.geometry import Polygon, LineString
    rect = lambda x: Polygon([x['d'] * t + x['n'] * (x['o'] + s * x['T'] / 2) for t, s in ((x['t0'], -1), (x['t1'], -1), (x['t1'], 1), (x['t0'], 1))])
    face = lambda x, t: LineString([x['d'] * t + x['n'] * (x['o'] - x['T'] / 2), x['d'] * t + x['n'] * (x['o'] + x['T'] / 2)])
    P = [rect(x) for x in EL]
    nsrc = {}
    for x in EL:
        nsrc[id(x['src'])] = nsrc.get(id(x['src']), 0) + 1
    drop = set()
    for i, x in enumerate(EL):
        if x['kind'] == "column" or x['t1'] - x['t0'] >= TOL or nsrc[id(x['src'])] < 2:
            continue
        t = [any(P[j].distance(face(x, te)) < touch for j in range(len(EL)) if j != i and j not in drop) for te in (x['t0'], x['t1'])]
        if t[0] != t[1]:
            drop.add(i)
    log(f"overshoot stubs removed: {len(drop)}")
    return [x for i, x in enumerate(EL) if i not in drop]
