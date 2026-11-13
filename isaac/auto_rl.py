"""Auto RL pipeline for the walking G1 (owner 2026-10-28): from the fixed-base grasp policy (dex-v6b) toward a robot
that walks to a table, grasps one of many objects, carries it to another table and sets it down, trained, chosen and
scored without a human in the loop, inside a fixed GPU-hour budget, and stopping on its own when it is not learning.
Runs on the cluster login node (standard library only); the GPU work is Slurm jobs of isaac/g1_dexsuite.py through
isaac/auto_cluster.sbatch. Resumable: every step is recorded in the state file, so `run` can be stopped and restarted,
and `status` shows where things are.

Stages (each waits for its jobs, parses their logs, decides, moves on):
  knobs    eval-only (3 jobs, minutes): the warm-start policy on the walking base with three action-side settings
           against the base kick found 2026-10-27 (waist frozen / strong action filter / both + rate cap); picks the
           setting with the smallest pelvis kick among those whose hand gets within 8 cm of the object (else the
           smallest kick). Also the smoke test of the new code.
  train    --gpus parallel PPO fine-tunes (g1_dexsuite.py train --base=walking --init --walk_dr=1): the chosen knobs,
           primitives or the object pool, two smoothness weights; hours per run cut to fit --budget_h; each run stops
           at its time limit or when its success reward stops improving (--patience) or diverges
  select   the last 2 checkpoints of every run on validation layouts (walking base; primitives and, when the pool
           exists, pool objects), the warm start evaluated the same way as the control; the checkpoint with the most
           objects kept through the shake is selected. If nothing lifted anything, the pipeline ends here.
  task     the end-to-end job (g1_dexsuite.py task: walk in, grasp, carry to table B, set down) with the selected
           checkpoint on validation, then once on test; pool objects too when available
  report   runs/auto_rl/report.md: every number measured, how each run ended, and why the pipeline stopped if it did

  python isaac/auto_rl.py plan                     print the jobs and the GPU-hour budget, submit nothing
  python isaac/auto_rl.py run [--poll=120]         advance until done (Ctrl-C and rerun to resume)
  python isaac/auto_rl.py status
Settings: --budget_h (sum of the jobs' time limits, default 10), --gpus (parallel training runs, <= 8), --train_hours,
--patience, --init, --unitree, --pool, --share, --code, --partition, --constraint, --state, --max_tries,
--dry (fake Slurm, for tests). Budget accounting uses the jobs' time limits, so it cannot be exceeded.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["plan", "run", "status", "step"])
p.add_argument("--share", default=os.environ.get("SHARE", os.path.expandvars("/nfs/hpc/share/$USER")))
p.add_argument("--code", default="rlRobots", help="the checkout under $SHARE/isaac the jobs run")
p.add_argument("--init", default="", help="warm-start checkpoint (default: $SHARE/isaac/rlg1g/runs/isaac/dex-v6b/model_4999.pt)")
p.add_argument("--unitree", default="", help="unitree_sim_isaaclab checkout (default $SHARE/isaac/unitree)")
p.add_argument("--pool", default="", help="pool_g1.py output folder (default $SHARE/datasets/pool_g1); skipped when missing")
p.add_argument("--gpus", type=int, default=2, help="parallel training runs (one GPU each), at most 8")
p.add_argument("--train_hours", type=float, default=3.0, help="per training run; cut to fit --budget_h")
p.add_argument("--iters", type=int, default=8000, help="PPO iterations asked for; --train_hours and the plateau stop end it earlier")
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--budget_h", type=float, default=10.0, help="GPU-hours the whole plan may ask for (sum of the jobs' time limits)")
p.add_argument("--patience", type=int, default=150, help="training stops after this many iterations without a better success window "
               "(about 35 min at the measured ~260 iterations/h), or when it diverges")
p.add_argument("--partition", default="", help="sbatch --partition override")
p.add_argument("--constraint", default="", help="sbatch --constraint override (the template: rtx8000|a40|l40s)")
p.add_argument("--n_val", type=int, default=12, help="validation layouts per shape for selection (72 robots: a 2-minute job)")
p.add_argument("--n_task", type=int, default=12, help="task layouts per shape (val and test)")
p.add_argument("--walk_in", type=float, default=1.0, help="task: walk-in distance (m)")
p.add_argument("--state", default=str(REPO / "runs" / "auto_rl" / "state.json"))
p.add_argument("--poll", type=int, default=120, help="run: seconds between checks")
p.add_argument("--max_tries", type=int, default=2, help="submissions per job before the pipeline stops on its failure")
p.add_argument("--dry", action="store_true", help="no Slurm: jobs are recorded as submitted; logs are read from --dry_logs")
p.add_argument("--dry_logs", default="", help="--dry: folder with auto-run-<fake id>.log files to parse")
args = p.parse_args()

SHARE = args.share
I = f"{SHARE}/isaac"
INIT = args.init or f"{I}/rlg1g/runs/isaac/dex-v6b/model_4999.pt"
UNITREE = args.unitree or f"{I}/unitree"
POOL = args.pool or f"{SHARE}/datasets/pool_g1"
OUT = f"{I}/auto"  # training runs and task records on the share
COMMON = f"--base=walking --hand=pinch --unitree={UNITREE} --pool={POOL}"
KNOBS = {"fw-ema0.4": "--ema=0.4 --freeze_waist=1", "ema0.15": "--ema=0.15", "fw-ema0.4-clip0.5": "--ema=0.4 --freeze_waist=1 --act_clip=0.5"}
VARIANTS = [  # training runs, in priority order (the first --gpus are used)
    dict(name="prim-a", objects="primitives", extra="--rate_w=0.01"),
    dict(name="pool-a", objects="pool", extra="--rate_w=0.01"),
    dict(name="prim-b", objects="primitives", extra="--rate_w=0.03 --acc_w=1e-5"),
    dict(name="pool-b", objects="pool", extra="--rate_w=0.03 --acc_w=1e-5"),
    dict(name="prim-c", objects="primitives", extra="--rate_w=0.01 --walk_dr_from=2.5 --prim_frac=1.0"),
    dict(name="pool-c", objects="pool", extra="--rate_w=0.01 --prim_frac=0.5"),
    dict(name="prim-d", objects="primitives", extra="--rate_w=0.005 --lcp=0.002"),
    dict(name="pool-d", objects="pool", extra="--rate_w=0.005 --lcp=0.002"),
]
STAGES = ["knobs", "train", "select", "task_val", "task_test", "report"]
T_EVAL, T_TASK, T_SLACK = 0.25, 0.4, 0.25  # job time limits, h: a 72-robot eval took 2 min on an A40 (2026-10-28); task ~10 min
IT_PER_H = 260  # PPO iterations per hour at 4096 envs x 32 steps, from the RTX 8000 throughput measured 2026-10 (~9.6k steps/s)


def train_hours():
    """Per-run training time that fits the budget: the budget minus every other job's limit, shared by the runs."""
    sets = 1 + int(pool_exists())
    fixed = len(KNOBS) * T_EVAL + (len(train_variants()) + 1) * sets * T_EVAL + 2 * sets * T_TASK
    per_run = (args.budget_h - fixed) / max(1, len(train_variants())) - T_SLACK
    return max(0.5, min(args.train_hours, per_run))


# --- state ---------------------------------------------------------------------------------------------------------------
def load_state():
    f = Path(args.state)
    if f.exists():
        return json.loads(f.read_text())
    return dict(stage="knobs", jobs={}, knobs=None, selection=None, results={}, started=time.strftime("%FT%T"))


def save_state(st):
    f = Path(args.state)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(st, indent=1))


def say(*a):
    print("[auto_rl]", *a, flush=True)


# --- Slurm ---------------------------------------------------------------------------------------------------------------
def pool_exists():
    return Path(POOL, "pool_g1_index.json").exists()


def submit(st, name, script_args, hours, script="isaac/g1_dexsuite.py"):
    """One job; idempotent by name (a name already in the state is not resubmitted unless it failed)."""
    j = st["jobs"].get(name)
    if j and j["status"] not in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"):
        return j
    if j and j.get("tries", 0) >= args.max_tries:  # the same failure again and again: stop and show where to look
        log = Path(args.dry_logs if args.dry else I, f"auto-run-{j['id']}.log")
        tail = "\n".join(run_log(j["id"]).splitlines()[-15:])
        raise SystemExit(f"[auto_rl] {name} failed {j['tries']} times (last job {j['id']}, {log}):\n{tail}\n"
                         f"fix the cause, then rerun `auto_rl.py run`: failed jobs are retried ({args.max_tries} tries each; "
                         f"--max_tries raises it; delete the job from {args.state} to reset its count)")
    hh, mm = int(hours), int(round((hours - int(hours)) * 60))
    cmd = ["sbatch", "--parsable", f"--time={hh:02d}:{mm:02d}:00"]
    if args.partition:
        cmd.append(f"--partition={args.partition}")
    if args.constraint:
        cmd.append(f"--constraint={args.constraint}")
    cmd.append(str(REPO / "isaac" / "auto_cluster.sbatch"))
    env = dict(os.environ, SHARE=SHARE, CODE=args.code, SCRIPT=script, ARGS=script_args)
    if args.dry:  # deterministic fake ids, one per try
        jid = str(zlib.crc32(f"{name}:{(j or {}).get('tries', 0)}".encode()) % 10 ** 6)
    else:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout.strip()
        jid = out.split(";")[0]
    j = dict(id=jid, status="PENDING", args=script_args, hours=hours, script=script, submitted=time.strftime("%FT%T"),
             tries=(j or {}).get("tries", 0) + 1)
    st["jobs"][name] = j
    save_state(st)
    say(f"submitted {name}: job {jid} ({hours:.1f} h): {script} {script_args[:90]}...")
    return j


def job_state(jid):
    if args.dry:
        return "COMPLETED" if Path(args.dry_logs, f"auto-run-{jid}.log").exists() else "PENDING"
    try:
        out = subprocess.run(["sacct", "-j", jid, "-n", "-X", "-o", "State"], capture_output=True, text=True, timeout=60).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return "UNKNOWN"
    s = out.strip().split()
    return s[0].split("+")[0] if s else "UNKNOWN"


def run_log(jid):
    f = Path(args.dry_logs if args.dry else I, f"auto-run-{jid}.log")
    return f.read_text(errors="replace") if f.exists() else ""


def refresh(st, names):
    """Update job states; returns True when every named job is finished (any terminal state)."""
    done = True
    for n in names:
        j = st["jobs"].get(n)
        if not j:
            done = False
            continue
        if j["status"] == "COMPLETED" or (j["status"] == "TIMEOUT" and n.startswith("train-")):
            continue  # finished for good (a failed job was resubmitted by its stage's submit, or stopped the pipeline)
        j["status"] = job_state(j["id"])
        if j["status"] == "COMPLETED" and "AUTO-DONE rc=0" not in (Path(args.dry_logs if args.dry else I, f"auto-{j['id']}.log").read_text(errors="replace")
                                                                 if Path(args.dry_logs if args.dry else I, f"auto-{j['id']}.log").exists() else "AUTO-DONE rc=0"):
            j["status"] = "FAILED"
        if j["status"] in ("FAILED", "CANCELLED", "NODE_FAIL") or (j["status"] == "TIMEOUT" and not n.startswith("train-")):
            say(f"{n}: job {j['id']} {j['status']}; resubmitting (try {j['tries'] + 1} of {args.max_tries})")
            submit(st, n, j["args"], j["hours"], j.get("script", "isaac/g1_dexsuite.py"))  # stops the pipeline past max_tries
            done = False
        elif j["status"] not in ("COMPLETED", "TIMEOUT"):  # a training run's TIMEOUT is its normal end (the time limit)
            done = False
    save_state(st)
    return done


# --- log parsing -----------------------------------------------------------------------------------------------------------
def parse_base(text):
    """BASE <tag>: pelvis pitch ... median K deg (p90 P); closest pinch-to-object ..., median D m; ..."""
    m = re.search(r"^BASE .*?median(?: over envs)? ([\d.]+) deg \(p90 ([\d.]+)\); closest pinch-to-object[^\d]*([\d.]+) m", text, re.M)
    return dict(kick=float(m.group(1)), kick_p90=float(m.group(2)), near=float(m.group(3))) if m else None


def parse_evals(text):
    """EVAL <tag> seeds S+: lifted a/N, held b/N; per shape {...}; kept c/N after the shake  -> {tag: (lifted, held, kept, N)}"""
    out = {}
    for m in re.finditer(r"^EVAL (\S+) seeds \d+\+: lifted (\d+)/(\d+), held (\d+)/\d+;.*?kept (\d+)/\d+", text, re.M):
        out[m.group(1)] = dict(lifted=int(m.group(2)), n=int(m.group(3)), held=int(m.group(4)), kept=int(m.group(5)))
    return out


def parse_task(text):
    m = re.search(r"^TASK (\S+) .*?SUCCESS (\d+)/(\d+)", text, re.M)
    rec = None
    for line in text.splitlines():
        if line.startswith("BENCH_RESULT "):
            try:
                rec = json.loads(line[len("BENCH_RESULT "):])
            except json.JSONDecodeError:
                pass
    return dict(tag=m.group(1), success=int(m.group(2)), n=int(m.group(3)), record=rec) if m else None


# --- stages --------------------------------------------------------------------------------------------------------------
def stage_knobs(st):
    names = []
    for k, flags in KNOBS.items():
        names.append(f"knobs-{k}")
        submit(st, f"knobs-{k}", f"eval {COMMON} --checkpoint={INIT} --n=12 --split=val {flags} --tag=auto", T_EVAL)
    if not refresh(st, names):
        return False
    best = None
    for k in KNOBS:
        j = st["jobs"][f"knobs-{k}"]
        b = parse_base(run_log(j["id"])) if j["status"] == "COMPLETED" else None
        j["result"] = b
        say(f"knobs {k}: {b}")
        if b is None:
            continue
        key = (0 if b["near"] <= 0.08 else 1, b["kick"])
        if best is None or key < best[0]:
            best = (key, k)
    if best is None:
        say("every knob eval failed: check the logs; fix, then delete the knobs-* jobs from the state file to retry")
        return False
    st["knobs"] = dict(name=best[1], flags=KNOBS[best[1]], result=st["jobs"][f"knobs-{best[1]}"]["result"])
    say(f"knobs chosen: {st['knobs']}")
    st["stage"] = "train"
    save_state(st)
    return True


def train_variants():
    vs = [v for v in VARIANTS if v["objects"] == "primitives" or pool_exists()]
    return vs[:max(1, min(args.gpus, 8))]


def stage_train(st):
    flags = st["knobs"]["flags"]
    names = []
    for v in train_variants():
        n = f"train-{v['name']}"
        names.append(n)
        h = train_hours()
        a = (f"train {COMMON} --init={INIT} --walk_dr=1 --objects={v['objects']} --num_envs={args.num_envs} --iters={args.iters} "
             f"--time_limit_h={h} --save_every=50 --stop_patience={args.patience} --out={OUT}/{v['name']} {flags} {v['extra']} --tag={v['name']}")
        submit(st, n, a, h + T_SLACK)
    if not refresh(st, names):
        return False
    ok = [n for n in names if st["jobs"][n]["status"] in ("COMPLETED", "TIMEOUT") and checkpoints(n)]
    for n in names:  # how each run ended (TRAIN-STOP reason=... best_window=...) and its last LEARN line
        text = run_log(st["jobs"][n]["id"])
        stop = re.search(r"^TRAIN-STOP (.*)$", text, re.M)
        learn = re.findall(r"^LEARN (.*)$", text, re.M)
        st["jobs"][n]["result"] = dict(stop=stop.group(1) if stop else "ran to the job limit or crashed", last=learn[-1] if learn else "")
        say(f"{n}: {st['jobs'][n]['result']}")
    say(f"training finished: {len(ok)} of {len(names)} runs have checkpoints")
    if not ok:
        say("no run produced a checkpoint: see the logs")
        return False
    st["stage"] = "select"
    save_state(st)
    return True


def checkpoints(train_name, last=2):
    d = Path(OUT) / train_name.replace("train-", "")
    files = sorted(glob.glob(str(d / "model_*.pt")), key=lambda f: int(Path(f).stem.split("_")[1]))
    return files[-last:]


def stage_select(st):
    names = []
    runs = {n: checkpoints(n) for n in st["jobs"] if n.startswith("train-")}
    runs = {n: c for n, c in runs.items() if c}
    for n, ck in runs.items():
        for objs in ["primitives"] + (["pool"] if pool_exists() else []):
            jn = f"select-{n}-{objs}"
            names.append(jn)
            submit(st, jn, f"eval {COMMON} --checkpoint={','.join(ck)} --n={args.n_val} --split=val --objects={objs} {st['knobs']['flags']} --tag=sel", T_EVAL)
    for objs in ["primitives"] + (["pool"] if pool_exists() else []):  # the warm start itself, the control
        jn = f"select-control-{objs}"
        names.append(jn)
        submit(st, jn, f"eval {COMMON} --checkpoint={INIT} --n={args.n_val} --split=val --objects={objs} {st['knobs']['flags']} --tag=sel", T_EVAL)
    if not refresh(st, names):
        return False
    scores = {}  # checkpoint path -> summed kept over object sets
    for n in names:
        j = st["jobs"][n]
        if j["status"] != "COMPLETED":
            continue
        ev = parse_evals(run_log(j["id"]))
        j["result"] = ev
        run = n.split("-", 1)[1].rsplit("-", 1)[0]
        cks = [INIT] if run == "control" else checkpoints(f"train-{run}")
        for ck in cks:
            tag = Path(ck).parent.name + "-" + Path(ck).stem
            hit = next((v for t, v in ev.items() if t.startswith(tag)), None)
            if hit:
                scores.setdefault(ck, dict(kept=0, lifted=0, n=0, sets=0))
                for k in ("kept", "lifted", "n"):
                    scores[ck][k] += hit[k]
                scores[ck]["sets"] += 1
    st["results"]["select"] = scores
    cands = {ck: v for ck, v in scores.items() if ck != INIT}
    if not cands:
        say("no evaluated checkpoint: see the select-* logs")
        return False
    best = max(cands, key=lambda ck: (cands[ck]["kept"], cands[ck]["lifted"]))
    ctrl = scores.get(INIT, {})
    st["selection"] = dict(checkpoint=best, score=cands[best], control=ctrl)
    say(f"selected {best}: {cands[best]} (warm start: {ctrl})")
    if cands[best]["lifted"] == 0:  # nothing learned: the task stage would only spend GPU time confirming it
        st["results"]["stop"] = "no checkpoint lifted any object on validation: the fine-tune did not learn; the task stage is skipped"
        say(st["results"]["stop"])
        st["stage"] = "report"
    else:
        st["stage"] = "task_val"
    save_state(st)
    return True


def stage_task(st, split):
    ck = st["selection"]["checkpoint"]
    names = []
    for objs in ["primitives"] + (["pool"] if pool_exists() else []):
        jn = f"task-{split}-{objs}"
        names.append(jn)
        submit(st, jn, f"task {COMMON} --checkpoint={ck} --n={args.n_task} --split={split} --objects={objs} --walk_in={args.walk_in} "
                       f"{st['knobs']['flags']} --tag=auto-{split}", T_TASK)
    if not refresh(st, names):
        return False
    for jn in names:
        j = st["jobs"][jn]
        t = parse_task(run_log(j["id"])) if j["status"] == "COMPLETED" else None
        j["result"] = {k: v for k, v in (t or {}).items() if k != "record"}
        st["results"][jn] = j["result"]
        if t and t.get("record"):
            d = REPO / "bench" / "results"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"auto-{jn}.json").write_text(json.dumps(t["record"], indent=1))
        say(f"{jn}: {j['result'] or j['status']}")
    st["stage"] = "task_test" if split == "val" else "report"
    save_state(st)
    return True


def stage_report(st):
    lines = ["# Auto RL pipeline report", "", f"started {st['started']}, report {time.strftime('%FT%T')}", "",
             f"warm start: `{INIT}`", f"knobs: {st['knobs']}", "", "## Selection (validation, walking base; kept through the shake, summed over object sets)", ""]
    for ck, v in sorted(st["results"].get("select", {}).items(), key=lambda kv: -kv[1]["kept"]):
        lines.append(f"- {'CONTROL ' if ck == INIT else ''}`{ck}`: kept {v['kept']}, lifted {v['lifted']} of {v['n']} ({v['sets']} sets)")
    lines += ["", f"selected: `{st['selection']['checkpoint']}`" if st.get("selection") else "selected: none", "",
              "## End-to-end task (walk in, grasp, carry to table B, set down)", ""]
    for k, v in st["results"].items():
        if k.startswith("task-"):
            lines.append(f"- {k}: {v}")
    if st["results"].get("stop"):
        lines += ["", f"**stopped early**: {st['results']['stop']}"]
    lines += ["", "## Training runs", ""] + [f"- {n}: {j.get('result')}" for n, j in st["jobs"].items() if n.startswith("train-")]
    lines += ["", "## Jobs", ""] + [f"- {n}: job {j['id']} {j['status']} ({j['hours']:.2f} h limit)" for n, j in st["jobs"].items()]
    f = Path(args.state).parent / "report.md"
    f.write_text("\n".join(lines) + "\n")
    say(f"report: {f}")
    st["stage"] = "done"
    save_state(st)
    return True


def plan():
    hours, th = 0.0, train_hours()
    nv, sets = len(train_variants()), 1 + int(pool_exists())
    rows = [("knobs", len(KNOBS), T_EVAL), ("train", nv, th + T_SLACK), ("select", nv * sets, T_EVAL), ("select control", sets, T_EVAL),
            ("task val + test", 2 * sets, T_TASK)]
    for name, n, h in rows:
        hours += n * h
        print(f"  {name:18s} {n:3d} jobs x {h:5.2f} h = {n * h:6.2f} GPU-h (job time limits)")
    print(f"  total {hours:.1f} GPU-h of the {args.budget_h:.0f} budget; wall time about {T_EVAL + th + T_SLACK + T_EVAL + 2 * T_TASK:.1f} h at most; "
          f"pool objects: {'yes' if pool_exists() else 'no (' + POOL + ' missing)'}; warm start {'found' if Path(INIT).exists() else 'MISSING: ' + INIT}")
    print(f"  training: {th:.2f} h per run = about {int(th * IT_PER_H)} PPO iterations (estimate from the RTX 8000 throughput measured "
          f"2026-10; A40 / L40S differ), checkpoints every 50, plateau stop after {args.patience} iterations without a better success window")
    print("  training runs: " + ", ".join(f"{v['name']} ({v['objects']}; {v['extra']})" for v in train_variants()))
    return hours


def step(st):
    s = st["stage"]
    return {"knobs": stage_knobs, "train": stage_train, "select": stage_select, "task_val": lambda st: stage_task(st, "val"),
            "task_test": lambda st: stage_task(st, "test"), "report": stage_report}.get(s, lambda st: True)(st)


def main():
    st = load_state()
    if args.cmd == "plan":
        plan()
        return 0
    if args.cmd == "status":
        print(f"stage: {st['stage']}; knobs: {st.get('knobs')}; selection: {st.get('selection')}")
        for n, j in st["jobs"].items():
            print(f"  {n:32s} job {j['id']:>8s} {j['status']:10s} {j.get('result') or ''}")
        return 0
    if st["stage"] == "knobs" and not st["jobs"]:
        if plan() > args.budget_h:
            say("the plan exceeds the budget: lower --gpus or --train_hours")
            return 2
        if not args.dry and not Path(INIT).exists():
            say(f"warm start checkpoint missing: {INIT} (--init)")
            return 2
    while True:
        while step(st) and st["stage"] != "done":  # a finished stage submits the next one's jobs at once
            pass
        if st["stage"] == "done":
            say("done")
            return 0
        if args.cmd == "step":
            return 0
        if args.dry:
            return 1
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
