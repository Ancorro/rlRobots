"""Scene memory for stack v2 (owner 2026-10-09: "more architecture"; re-identification must be checked: reid_bench.py).

One track per physical object, kept across looks:
  id (p1, p2, ...; never reused), name (set once by grounding), appearance (unit image embedding, running mean),
  centre (m), dims (cm), where (table / red_tray / blue_tray / shelf / gripper), status (seen / held / missing),
  last_seen (look number), misses (looks in a row unseen), history [(look, event)].
Each look: predict where every track should be (the last action: held in the gripper, placed at a place), match the
detections to the tracks (Hungarian, cost = position term + appearance term, gated), start tracks for the unmatched
detections, age the unmatched tracks: missing after MISS_LOOKS looks in a row where its spot was in view and not
behind the robot. A placed object seen far from where it was put is flagged ("drop?").
Methods (reid_bench.py compares them on the same looks): "v1" (stack v1: nearest within 4 cm, no prediction),
"position", "appearance", "both".
"""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # the perception environment has no scipy
    def linear_sum_assignment(C):
        """Hungarian algorithm (Jonker-Volgenant style potentials), rectangular C; returns (rows, cols) like scipy."""
        C = np.asarray(C, float)
        tr = C.shape[0] > C.shape[1]
        if tr:
            C = C.T
        n, m = C.shape
        u, v, p, way = np.zeros(n + 1), np.zeros(m + 1), np.zeros(m + 1, int), np.zeros(m + 1, int)
        for i in range(1, n + 1):
            p[0], j0 = i, 0
            minv, used = np.full(m + 1, np.inf), np.zeros(m + 1, bool)
            while True:
                used[j0] = True
                i0, delta, j1 = p[j0], np.inf, 0
                for j in range(1, m + 1):
                    if not used[j]:
                        cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j], way[j] = cur, j0
                        if minv[j] < delta:
                            delta, j1 = minv[j], j
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        rows = [p[j] - 1 for j in range(1, m + 1) if p[j]]
        cols = [j - 1 for j in range(1, m + 1) if p[j]]
        r, c = (np.array(cols), np.array(rows)) if tr else (np.array(rows), np.array(cols))
        o = np.argsort(r)
        return r[o], c[o]

MISS_LOOKS = 2
DROP_M = 0.04  # a placed object seen this far from its place: dropped or knocked


class Track:
    def __init__(self, tid, det, look):
        self.id, self.name = tid, det.get("name")
        self.emb = None if det.get("emb") is None else det["emb"] / np.linalg.norm(det["emb"])
        self.centre, self.dims, self.where = np.asarray(det["centre"], float), det.get("dims"), det.get("where", "table")
        self.status, self.last_seen, self.misses, self.n = "seen", look, 0, 1
        self.expect = None  # (kind, xy): "held" or "placed" at xy, from the last action
        self.history = [(look, "new")]


class SceneMemory:
    def __init__(self, method="both", pos_scale=0.05, app_scale=0.15, w_app=1.0, gate=3.0, v1_same=0.04, ema=0.3,
                 relink=True, relink_gate=1.0):
        """pos_scale: metres that cost 1; app_scale: (1 - cosine) that costs 1; gate: highest cost still matched;
        relink_gate: highest appearance-only cost for the second pass (lost track to unexplained detection)."""
        self.method, self.pos_scale, self.app_scale, self.w_app, self.gate = method, pos_scale, app_scale, w_app, gate
        self.relink, self.relink_gate = relink, relink_gate
        self.v1_same, self.ema = v1_same, ema
        self.tracks, self.look, self.next_id = {}, 0, 1

    # --- actions tell the memory what should have happened --------------------------------------------------------------
    def picked(self, tid, hand_xy=None):
        t = self.tracks[tid]
        t.status, t.where, t.expect = "held", "gripper", ("held", None if hand_xy is None else np.asarray(hand_xy, float))
        t.history.append((self.look, "picked"))

    def placed(self, tid, xy, where):
        t = self.tracks[tid]
        t.status, t.where, t.expect = "seen", where, ("placed", np.asarray(xy, float))
        t.centre = np.array([xy[0], xy[1], t.centre[2]])
        t.history.append((self.look, f"placed {where}"))

    # --- one look ------------------------------------------------------------------------------------------------------
    def predicted(self, t):
        if t.expect and t.expect[1] is not None:
            return np.array([*t.expect[1][:2], t.centre[2]])
        return t.centre

    def cost(self, t, d):
        pos = np.linalg.norm(self.predicted(t)[:2] - np.asarray(d["centre"])[:2]) / self.pos_scale
        app = 0.0
        if t.emb is not None and d.get("emb") is not None:
            app = (1 - float(t.emb @ (d["emb"] / np.linalg.norm(d["emb"])))) / self.app_scale
        if self.method == "position":
            return pos
        if self.method == "appearance":
            return app
        return pos + self.w_app * app

    def update(self, dets, in_view=None):
        """dets: [{centre, dims, where, emb, name?}]. in_view(xy) -> True when that spot was visible this look (not
        behind the robot, inside both cameras' view); None = always. Returns {det index: track id}, events."""
        self.look += 1
        live = [t for t in self.tracks.values() if t.status != "missing"]
        assign, events = {}, []
        if self.method == "v1":  # stack v1: nearest perceived object within 4 cm keeps its id
            for i, d in enumerate(dets):
                ds = [(np.linalg.norm(t.centre[:2] - np.asarray(d["centre"])[:2]), t.id) for t in live if t.id not in assign.values()]
                if ds and min(ds)[0] < self.v1_same:
                    assign[i] = min(ds)[1]
        elif live and dets:
            C = np.array([[self.cost(t, d) for d in dets] for t in live])
            r, c = linear_sum_assignment(C)
            for a, b in zip(r, c):
                if C[a, b] <= self.gate:
                    assign[int(b)] = live[a].id
        if self.method in ("appearance", "both") and self.relink:
            # second pass: a track not where it should be (dropped, knocked far, moved while unseen) and a detection no
            # track explains: re-link by appearance alone, anywhere on the table
            left_t = [t for t in live if t.id not in assign.values() and t.status != "held" and t.emb is not None]
            left_d = [i for i in range(len(dets)) if i not in assign and dets[i].get("emb") is not None]
            if left_t and left_d:
                C = np.array([[(1 - float(t.emb @ (dets[i]["emb"] / np.linalg.norm(dets[i]["emb"])))) / self.app_scale
                               for i in left_d] for t in left_t])
                r, c = linear_sum_assignment(C)
                for a, b in zip(r, c):
                    if C[a, b] <= self.relink_gate:
                        assign[left_d[b]] = left_t[a].id
                        events.append(("relinked", left_t[a].id))
        for i, d in enumerate(dets):
            if i in assign:
                t = self.tracks[assign[i]]
                if t.expect and t.expect[0] == "placed" and np.linalg.norm(t.expect[1][:2] - np.asarray(d["centre"])[:2]) > DROP_M:
                    events.append(("drop?", t.id))
                    t.history.append((self.look, "drop?"))
                t.centre, t.dims, t.where = np.asarray(d["centre"], float), d.get("dims", t.dims), d.get("where", t.where)
                if d.get("emb") is not None:
                    e = d["emb"] / np.linalg.norm(d["emb"])
                    t.emb = e if t.emb is None else (1 - self.ema) * t.emb + self.ema * e
                    t.emb /= np.linalg.norm(t.emb)
                t.status = "held" if t.status == "held" and t.where == "gripper" else "seen"
                t.last_seen, t.misses, t.expect, t.n = self.look, 0, None, t.n + 1
            else:
                tid = f"p{self.next_id}"
                self.next_id += 1
                self.tracks[tid] = Track(tid, d, self.look)
                assign[i] = tid
                events.append(("new", tid))
        for t in live:
            if t.id in assign.values() or t.status == "held":
                continue
            visible = True if in_view is None else bool(in_view(self.predicted(t)[:2]))
            t.misses += visible
            if t.misses >= MISS_LOOKS:
                t.status = "missing"
                t.history.append((self.look, "missing"))
                events.append(("missing", t.id))
        return assign, events

    # --- reading it ----------------------------------------------------------------------------------------------------
    def goal_left(self, goal):
        """goal {track id: place}: the ids not yet at their place."""
        return [i for i, p in goal.items() if i not in self.tracks or self.tracks[i].where != p]

    def text(self):
        """The planner's and the executive's view: one line per track."""
        out = []
        for t in self.tracks.values():
            at = f"({t.centre[0] * 100:.0f}, {t.centre[1] * 100:.0f} cm)" if t.status != "held" else ""
            last = next((e for _, e in reversed(t.history) if e != "new"), "")
            out.append(f"{t.id} {t.name or 'unnamed'}: {t.where if t.status != 'missing' else 'missing'} {at}"
                       + (f"; {last} (look {t.history[-1][0]})" if last else ""))
        return "\n".join(out)
