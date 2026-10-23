"""Planner layer (docs/plan.md layer 3): a language model turns an open request ("clear the shelf", "put the toys in
the red tray") into pick-and-place moves, SayCan / Inner Monologue style. This file holds the symbolic side, shared
by the plan-only benchmark (plan_bench.py) and, later, the closed loop on the simulated arm:

  scene     objects (id, name, size, position, where it is) and the three places, as the planner sees them
  requests  open-ended request families over a scene, each with the exact goal (object id -> place; everything
            else must stay where it is) or None when the request cannot be done
  prompt    the instructions and the scene as text
  parse     the model's answer -> moves; check(moves) against the goal

Frame: x = cm forward from the robot base, y = cm to the robot's left (scene_general.xml). The planner names objects
by id (o1, o2, ...), never by free text, so a closed loop can only ask perception for objects it has seen.
"""
import json
import re

import numpy as np

PLACES = {"red_tray": ("the red tray", (14, 19)), "blue_tray": ("the blue tray", (14, -19)), "shelf": ("the shelf", (32, 0))}
WHERE = {"red_tray": "in the red tray", "blue_tray": "in the blue tray", "shelf": "on the shelf"}
TO = {"red_tray": "to the red tray", "blue_tray": "to the blue tray", "shelf": "to the shelf"}
# Request wording (2026-10-05): "Put everything on the table except the X in the red tray" read as "put ... on the
# table" (models answered "cannot"); group and superlative requests now say "Move ... to <place>".

# What each scanned-object phrase is, judged from the phrase alone (the planner sees only the phrase). Objects whose
# phrase leaves a category unclear are listed under "unclear" for it and never appear in that category's requests.
CATEGORIES = {
    "toys": {"members": {"wooden letter block", "baby rattle toy", "toy fire truck", "toy rhino", "toy cow", "toy lion",
                         "toy school bus", "green toy train", "blue toy train", "wind-up toy dog"},
             "unclear": {"pink dog toy", "grey game cartridge", "yellow game cartridge", "crayon box",  # figures: toy
                         "boy figurine in yellow hoodie", "silver android figure", "orange android figure",  # or ornament
                         "panda android figure", "green android figure", "lime green android figure",
                         "raccoon action figure", "ninja turtle figure", "mario figure"}},
    "toy vehicles": {"members": {"toy fire truck", "toy school bus", "green toy train", "blue toy train"}, "unclear": set()},
    "toy animals": {"members": {"toy rhino", "toy cow", "toy lion", "wind-up toy dog"},
                    "unclear": {"pink dog toy", "panda android figure", "raccoon action figure", "ninja turtle figure",
                                "green android figure"}},
    "pill bottles": {"members": {"white pill bottle", "pink label pill bottle", "red label pill bottle",
                                 "yellow label pill bottle", "blue label pill bottle", "dark blue pill bottle"},
                     "unclear": set()},
    "video games": {"members": {"grey game cartridge", "yellow game cartridge"}, "unclear": set()},
    "printer supplies": {"members": {"ink cartridge package", "ink cartridge box"}, "unclear": {"small cardboard box", "white box"}},
    "boxes": {"members": {"small cardboard box", "purple box", "pink box", "blue and white box", "crayon box",
                          "ink cartridge box", "white box", "beige box", "orange box"},
              "unclear": {"ink cartridge package", "fish tin"}},
}


# --- scene ----------------------------------------------------------------------------------------------
def dims_cm(spec, index=None):
    """Bounding size (length, width, height) in cm, longest side first for the footprint."""
    if spec["shape"] == "scanned":
        d = next(o["dims_m"] for o in index if o["name"] == spec["scanned"])
        l, w, h = d
    else:
        s, t = spec["size"], spec["type"]
        if t == "box":
            l, w, h = 2 * s[0], 2 * s[1], 2 * s[2]
        elif t == "cylinder" and spec["quat_axis"] is None:
            l = w = 2 * s[0]
            h = 2 * s[1]
        elif t == "cylinder":  # lying
            l, w, h = 2 * s[1], 2 * s[0], 2 * s[0]
        elif t == "sphere":
            l = w = h = 2 * s[0]
        elif t == "capsule":  # lying
            l, w, h = 2 * (s[1] + s[0]), 2 * s[0], 2 * s[0]
        else:  # ellipsoid lying on its side
            l, w, h = 2 * s[0], 2 * s[1], 2 * s[2]
    l, w = max(l, w), min(l, w)
    return round(l * 100, 1), round(w * 100, 1), round(h * 100, 1)


def make_scene(specs, poses, index=None, start=None):
    """Objects as the planner sees them. start: {object index: place} for objects that begin on a place."""
    objs = []
    for i, (s, (x, y, _)) in enumerate(zip(specs, poses)):
        where = (start or {}).get(i, "table")
        if where != "table":
            x, y = PLACES[where][1][0] / 100, PLACES[where][1][1] / 100
        objs.append(dict(id=f"o{i + 1}", name=s["name"], colour=s.get("colour"), noun=s["noun"], dims=dims_cm(s, index),
                         x=round(x * 100, 1), y=round(y * 100, 1), where=where))
    return objs


def scene_text(objs):
    lines = ["Objects (x = cm forward from the robot base, y = cm to the robot's left; size = length x width x height cm):"]
    for o in objs:
        at = "on the table" if o["where"] == "table" else WHERE[o["where"]]
        l, w, h = o["dims"]
        lines.append(f"- {o['id']}: {o['name']}, {l} x {w} x {h} cm, at x={o['x']:.0f} y={o['y']:.0f}, {at}")
    lines.append("Places: red_tray (x=14 y=19, flat pad on the table), blue_tray (x=14 y=-19, flat pad on the table), "
                 "shelf (x=32 y=0, a block whose top is 6 cm above the table; nothing above it limits how tall things can be).")
    # (2026-10-05: "shelf ... 6 cm high" read as a 6 cm height limit: gpt-oss refused objects over 6 cm tall)
    return "\n".join(lines)


SYSTEM = """You plan for a one-arm tabletop robot. It can do one thing: pick up one object and put it in a place.
Places: red_tray, blue_tray, shelf. Each place holds several objects side by side: putting an object where others
already are is fine (that is not stacking). The robot cannot put things back on the table, stack one object on top of
another, push things or open anything.
Left and right are from the robot's point of view (larger y = further left). "Closest" means distance from the robot
base (x=0, y=0). "Biggest" and "smallest" mean volume (length x width x height).
Move only the objects the request needs; everything else must stay where it is. An object that is already where the
request wants it needs no move.
Answer with one JSON object and nothing after it:
{"moves": [{"object": "o2", "place": "red_tray"}, ...], "cannot": false}
Moves run in order. If the request is impossible here (an object or place that does not exist, or a skill the robot
does not have), answer {"moves": [], "cannot": true, "reason": "..."}."""


def schema(objs):
    """JSON schema of a valid answer for this scene, for constrained decoding: object ids from this scene only, the
    three places only. Same field order as the instructions ask for."""
    move = {"type": "object", "properties": {"object": {"type": "string", "enum": [o["id"] for o in objs]},
                                             "place": {"type": "string", "enum": list(PLACES)}},
            "required": ["object", "place"]}
    return {"type": "object", "properties": {"moves": {"type": "array", "items": move}, "cannot": {"type": "boolean"},
                                             "reason": {"type": "string"}},
            "required": ["moves", "cannot"]}


def prompt(objs, request):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{scene_text(objs)}\n\nRequest: {request}"}]


# --- requests -------------------------------------------------------------------------------------------
def _plural(noun):
    return noun if noun.endswith("s") else noun + "es" if noun.endswith(("x", "sh", "ch")) else noun + "s"


def _vol(o):
    return float(np.prod(o["dims"]))


def requests(objs, rng, scanned):
    """All request families that fit this scene: [(family, text, goal)]. goal: {object id: place} (others stay), or
    None when the request cannot be done."""
    out = []
    on_table = [o for o in objs if o["where"] == "table"]
    placed0 = [o for o in objs if o["where"] != "table"]  # group requests skip groups with a member already placed
    places = list(PLACES)

    def pick_place(exclude=()):
        return str(rng.choice([p for p in places if p not in exclude]))

    if on_table:  # single move (sanity: what bench_general asks)
        o, p = on_table[rng.integers(len(on_table))], pick_place()
        out.append(("single", f"Put the {o['name']} {WHERE[p]}.", {o["id"]: p}))
    if len(on_table) >= 2:  # two moves to different places
        a, b = rng.choice(len(on_table), 2, replace=False)
        p1, p2 = rng.choice(places, 2, replace=False)
        a, b = on_table[a], on_table[b]
        out.append(("two", f"Put the {a['name']} {WHERE[p1]} and the {b['name']} {WHERE[p2]}.", {a["id"]: p1, b["id"]: p2}))
    if len(on_table) >= 3:  # everything except one
        keep = on_table[rng.integers(len(on_table))]
        p = pick_place()
        out.append(("all-but", f"Move every object that is on the table, except the {keep['name']}, {TO[p]}.",
                    {o["id"]: p for o in on_table if o is not keep}))
    # position and size superlatives, only with a clear winner (2 cm / 25 % margin)
    sup = [("closest", "the object closest to the robot", lambda o: -np.hypot(o["x"], o["y"]), 2.0),
           ("leftmost", "the leftmost object", lambda o: o["y"], 2.0),
           ("rightmost", "the rightmost object", lambda o: -o["y"], 2.0),
           ("furthest", "the object furthest from the robot", lambda o: np.hypot(o["x"], o["y"]), 2.0),
           ("tallest", "the tallest object", lambda o: o["dims"][2], 1.0),
           ("biggest", "the biggest object", _vol, None), ("smallest", "the smallest object", lambda o: -_vol(o), None)]
    if len(on_table) >= 3:
        kind, phrase, key, margin = sup[rng.integers(len(sup))]
        ranked = sorted(on_table, key=key, reverse=True)
        a, b = key(ranked[0]), key(ranked[1])
        clear = (a - b > margin) if margin else (abs(a) > 1.25 * abs(b) if a > 0 else abs(b) > 1.25 * abs(a))
        if clear:
            p = pick_place()
            out.append(("superlative", f"Of the objects on the table, move {phrase} {TO[p]}.", {ranked[0]["id"]: p}))
    # groups by kind: same noun (primitives) or a category (scanned)
    if scanned:
        names = {o["name"] for o in on_table}
        cats = [c for c, d in CATEGORIES.items() if not names & d["unclear"]
                and 1 <= len(names & d["members"]) < len(on_table)]
        cats = [c for c in cats if not {o["name"] for o in placed0} & CATEGORIES[c]["members"]]  # none already placed
        if cats:
            c = cats[rng.integers(len(cats))]
            p = pick_place()
            out.append(("category", f"Put all the {c} {WHERE[p]}.",
                        {o["id"]: p for o in on_table if o["name"] in CATEGORIES[c]["members"]}))
            p = pick_place()
            out.append(("not-category", f"Move every object on the table that is not one of the {c} {TO[p]}.",
                        {o["id"]: p for o in on_table if o["name"] not in CATEGORIES[c]["members"]}))
    else:
        nouns = {}
        for o in on_table:
            nouns.setdefault(o["noun"], []).append(o)
        groups = [n for n, g in nouns.items() if len(g) >= 2 and not any(o["noun"] == n for o in placed0)]
        if groups:
            n = groups[rng.integers(len(groups))]
            p = pick_place()
            out.append(("category", f"Put all the {_plural(n)} {WHERE[p]}.", {o["id"]: p for o in nouns[n]}))
        cols = sorted({o["colour"] for o in on_table} - {o["colour"] for o in placed0})
        if len(cols) >= 3:  # colour sort: two named colours to the trays (named for the trays' own colours)
            a, b = rng.choice(cols, 2, replace=False)
            out.append(("sort", f"Put the {a} things in the red tray and the {b} things in the blue tray.",
                        {**{o["id"]: "red_tray" for o in on_table if o["colour"] == a},
                         **{o["id"]: "blue_tray" for o in on_table if o["colour"] == b}}))
    # state: objects that start on a place
    placed = [o for o in objs if o["where"] != "table"]
    for where in {o["where"] for o in placed}:
        dest = pick_place(exclude=(where,))
        out.append(("clear", f"Clear {PLACES[where][0]}: move everything on it {TO[dest]}.",
                    {o["id"]: dest for o in placed if o["where"] == where}))
    trays = {w: [o for o in placed if o["where"] == w] for w in ("red_tray", "blue_tray")}
    if trays["red_tray"] and trays["blue_tray"]:
        out.append(("swap", "Swap what is in the red tray with what is in the blue tray.",
                    {**{o["id"]: "blue_tray" for o in trays["red_tray"]}, **{o["id"]: "red_tray" for o in trays["blue_tray"]}}))
    if placed and on_table:  # already done: the right answer is no moves
        o = placed[rng.integers(len(placed))]
        out.append(("already", f"Make sure the {o['name']} is {WHERE[o['where']]}.", {}))
    # conditional: one branch true
    if len(on_table) >= 2:
        o, other = (on_table[i] for i in rng.choice(len(on_table), 2, replace=False))
        p1, p2 = rng.choice(places, 2, replace=False)
        absent = _absent_name(objs, rng, scanned)
        if rng.random() < 0.5:
            out.append(("conditional", f"If there is a {o['name']}, put it {WHERE[p1]}; otherwise put the {other['name']} {WHERE[p2]}.",
                        {o["id"]: p1}))
        else:
            out.append(("conditional", f"If there is a {absent}, put it {WHERE[p1]}; otherwise put the {other['name']} {WHERE[p2]}.",
                        {other["id"]: p2}))
    # impossible: missing object, missing place, missing skill
    p = pick_place()
    out.append(("impossible", f"Put the {_absent_name(objs, rng, scanned)} {WHERE[p]}.", None))
    if on_table:
        o = on_table[rng.integers(len(on_table))]
        out.append(("impossible", rng.choice([f"Put the {o['name']} in the green tray.",
                                              f"Stack the {o['name']} on top of the {objs[0]['name'] if objs[0] is not o else objs[-1]['name']}.",
                                              f"Push the {o['name']} to the edge of the table."]), None))
    return out


def _absent_name(objs, rng, scanned):
    names = {o["name"] for o in objs}
    if scanned:
        pool = sorted({n for d in CATEGORIES.values() for n in d["members"]} - names)
    else:
        from so101.objects import COLOURS, NOUN
        pool = sorted({f"{c} {n}" for c in COLOURS for n in set(NOUN.values())} - names)
    return str(pool[rng.integers(len(pool))])


# --- answers ------------------------------------------------------------------------------------------
def parse(text):
    """The last JSON object in the answer, after any thinking. Returns dict or None. (Call only on finished answers:
    thinking cut off by the token limit often quotes the example JSON from the instructions.)"""
    for end in ("</think>", "<channel|>", "assistantfinal"):  # Qwen / Gemma 4 / gpt-oss: answer follows the last one
        if end in text:
            text = text.rsplit(end, 1)[1]
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in reversed(starts):  # the last complete object
        depth = 0
        for j in range(s, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                try:
                    d = json.loads(text[s:j + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(d, dict) and "moves" in d:
                    return d
                break
    return None


def check(objs, answer, goal):
    """True if the answer does what the request asks: for a possible request the final places equal the goal and no
    other object moved; for an impossible one, no moves and cannot = true."""
    if answer is None:
        return False
    moves = answer.get("moves") or []
    if goal is None:
        return bool(answer.get("cannot")) and not moves
    if answer.get("cannot"):
        return False
    state = {o["id"]: o["where"] for o in objs}
    start = dict(state)
    for m in moves:
        if not isinstance(m, dict) or m.get("object") not in state or m.get("place") not in PLACES:
            return False
        state[m["object"]] = m["place"]
    want = dict(start)
    want.update(goal)
    return state == want


def oracle(objs, goal):
    if goal is None:
        return {"moves": [], "cannot": True}
    return {"moves": [{"object": i, "place": p} for i, p in goal.items()
                      if next(o for o in objs if o["id"] == i)["where"] != p], "cannot": False}
