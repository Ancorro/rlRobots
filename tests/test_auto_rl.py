"""isaac/auto_rl.py in --dry mode: the whole state machine on fake job logs, no Slurm."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(tmp, *cmd):
    r = subprocess.run([PY, str(REPO / "isaac" / "auto_rl.py"), *cmd, "--dry", f"--share={tmp}/share", f"--dry_logs={tmp}/logs",
                        f"--state={tmp}/state.json", "--gpus=2", "--train_hours=1", f"--init={tmp}/share/isaac/v6b.pt"],
                       capture_output=True, text=True, cwd=str(REPO))
    return r.returncode, r.stdout + r.stderr


def jobs(tmp):
    return json.loads(Path(tmp, "state.json").read_text())["jobs"]


def log(tmp, name, text):
    jid = jobs(tmp)[name]["id"]
    Path(tmp, "logs").mkdir(exist_ok=True)
    Path(tmp, "logs", f"auto-run-{jid}.log").write_text(text)


def bench_line(n=96, k=40):
    sys.path.insert(0, str(REPO / "isaac"))
    import bench_spec as B
    rows = [dict(seed=s, shape=sh, success=i < k, fell=False, upright=True, released=True) for i, (sh, s) in enumerate(B.seeds("walk_carry", "val", 16, B.SHAPES))]
    return "BENCH_RESULT " + json.dumps(B.record("walk_carry", "val", "dex:x", "a" * 64, "fixed", "primitives", dict(n=16, a=1, b=2), rows))


def test_pipeline_dry(tmp_path):
    tmp = str(tmp_path)
    share = Path(tmp, "share"); (share / "isaac").mkdir(parents=True)
    (share / "isaac" / "v6b.pt").write_bytes(b"x")
    (share / "datasets" / "pool_g1").mkdir(parents=True)
    (share / "datasets" / "pool_g1" / "pool_g1_index.json").write_text("{}")
    rc, out = run(tmp, "plan")
    assert rc == 0 and "GPU-h" in out and "pool objects: yes" in out, out
    # knobs: three evals submitted, then wait
    rc, out = run(tmp, "step")
    assert {"knobs-fw-ema0.4", "knobs-ema0.15", "knobs-fw-ema0.4-clip0.5"} <= set(jobs(tmp)), out
    log(tmp, "knobs-fw-ema0.4", "BASE t: pelvis pitch in the first 0.5 s, median over envs 4.0 deg (p90 7.0); closest pinch-to-object over the episode, median 0.030 m; x\n")
    log(tmp, "knobs-ema0.15", "BASE t: pelvis pitch in the first 0.5 s, median over envs 2.0 deg (p90 3.0); closest pinch-to-object over the episode, median 0.150 m; x\n")
    log(tmp, "knobs-fw-ema0.4-clip0.5", "BASE t: pelvis pitch in the first 0.5 s, median over envs 6.0 deg (p90 9.0); closest pinch-to-object over the episode, median 0.040 m; x\n")
    rc, out = run(tmp, "step")
    st = json.loads(Path(tmp, "state.json").read_text())
    assert st["knobs"]["name"] == "fw-ema0.4" and st["stage"] == "train", out  # reaches (<= 8 cm) with the smaller kick
    # train: 2 runs (gpus=2): prim-a, pool-a
    assert {"train-prim-a", "train-pool-a"} <= set(jobs(tmp))
    assert "--init=" in jobs(tmp)["train-prim-a"]["args"] and "--walk_dr=1" in jobs(tmp)["train-prim-a"]["args"] and "--freeze_waist=1" in jobs(tmp)["train-prim-a"]["args"]
    for v in ("prim-a", "pool-a"):
        d = share / "isaac" / "auto" / v; d.mkdir(parents=True)
        for it in (250, 500, 750, 1000):
            (d / f"model_{it}.pt").write_bytes(b"w")
        log(tmp, f"train-{v}", "LEARN it=100 success=0.1 reward=1.0 best_window=0.1 elapsed_h=0.5\nTRAIN-STOP reason=time it=120 saved=model_120.pt best_window=0.12 elapsed_h=1.0\n")
    rc, out = run(tmp, "step")
    assert json.loads(Path(tmp, "state.json").read_text())["stage"] == "select", out
    # select: evals of the last 3 checkpoints per run on primitives and pool, plus the control
    rc, out = run(tmp, "step")
    names = set(jobs(tmp))
    assert {"select-train-prim-a-primitives", "select-train-pool-a-pool", "select-control-primitives"} <= names, names
    a = jobs(tmp)["select-train-prim-a-primitives"]["args"]
    assert "model_500" not in a and all(f"model_{it}.pt" in a for it in (750, 1000))  # only the last 2 checkpoints
    def ev(tag, kept):
        return f"EVAL {tag} seeds 700000+: lifted {kept + 5}/120, held {kept + 2}/120; per shape {{}}; kept {kept}/120 after the shake\n"
    for n in names:
        if not n.startswith("select-"):
            continue
        if "control" in n:
            log(tmp, n, ev("isaac-v6b-walkbase-sel", 10))  # tag = <parent dir>-<stem> of the init checkpoint
        else:
            v = n.split("-")[2] + "-" + n.split("-")[3]
            log(tmp, n, "".join(ev(f"{v}-model_{it}-walkbase-sel", kept) for it, kept in ((750, 50 if v == "pool-a" else 45), (1000, 40))))
    rc, out = run(tmp, "step")
    st = json.loads(Path(tmp, "state.json").read_text())
    assert st["stage"] == "task_val" and st["selection"]["checkpoint"].endswith("pool-a/model_750.pt"), out
    assert st["selection"]["score"]["kept"] == 100 and st["selection"]["control"]["kept"] == 20  # summed over the two object sets
    # task val, then test, then report
    for split in ("val", "test"):
        rc, out = run(tmp, "step")
        for objs in ("primitives", "pool"):
            log(tmp, f"task-{split}-{objs}", f"TASK pool-a-model_750-auto ({split}, 96 robots, objects {objs}): lifted 60, SUCCESS 41/96; x\n" + bench_line() + "\n")
        rc, out = run(tmp, "step")
    st = json.loads(Path(tmp, "state.json").read_text())
    assert st["stage"] == "done" and st["results"]["task-test-pool"]["success"] == 41, out  # the report follows at once
    rep = Path(tmp, "report.md").read_text()
    assert "selected: `" in rep and "task-test-primitives" in rep
    assert (REPO / "bench" / "results" / "auto-task-val-primitives.json").exists()
    for f in (REPO / "bench" / "results").glob("auto-task-*.json"):
        f.unlink()
    rc, out = run(tmp, "status")
    assert rc == 0 and "stage: done" in out


def test_failed_job_stops_after_max_tries(tmp_path):
    tmp = str(tmp_path)
    share = Path(tmp, "share"); (share / "isaac").mkdir(parents=True)
    (share / "isaac" / "v6b.pt").write_bytes(b"x")
    rc, out = run(tmp, "step")

    def fail_all():  # every knobs job fails (the sbatch log says rc=1)
        st = json.loads(Path(tmp, "state.json").read_text())
        Path(tmp, "logs").mkdir(exist_ok=True)
        for n in st["jobs"]:
            jid = st["jobs"][n]["id"]
            Path(tmp, "logs", f"auto-run-{jid}.log").write_text("Traceback (most recent call last):\nRuntimeError: size mismatch\n")
            Path(tmp, "logs", f"auto-{jid}.log").write_text("AUTO-DONE rc=1\n")
    fail_all()
    rc, out = run(tmp, "step")  # sees the failures and resubmits each once (new ids)
    assert "submitted knobs-ema0.15" in out and rc == 0, out
    fail_all()
    rc, out = run(tmp, "step")  # failed again: stop, with the log tail
    assert rc != 0 and "failed 2 times" in out and "size mismatch" in out, out


def test_no_learning_stops_before_task(tmp_path):
    tmp = str(tmp_path)
    share = Path(tmp, "share"); (share / "isaac").mkdir(parents=True)
    (share / "isaac" / "v6b.pt").write_bytes(b"x")
    run(tmp, "step")
    for n in ("knobs-fw-ema0.4", "knobs-ema0.15", "knobs-fw-ema0.4-clip0.5"):
        log(tmp, n, "BASE t: pelvis pitch in the first 0.5 s, median over envs 11.0 deg (p90 13.7); closest pinch-to-object over the episode, median 0.139 m; x\n")
    run(tmp, "step")  # knobs chosen (none reaches: the smallest kick), training submitted
    st = json.loads(Path(tmp, "state.json").read_text())
    assert st["stage"] == "train" and "--stop_patience=150" in st["jobs"]["train-prim-a"]["args"] and "--save_every=50" in st["jobs"]["train-prim-a"]["args"]
    for v in ("prim-a", "prim-b"):
        d = share / "isaac" / "auto" / v; d.mkdir(parents=True)
        (d / "model_50.pt").write_bytes(b"w")
        log(tmp, f"train-{v}", "TRAIN-STOP reason=plateau it=300 saved=model_300.pt best_window=0.0000 elapsed_h=1.2\n")
    run(tmp, "step")
    st = json.loads(Path(tmp, "state.json").read_text())
    assert st["stage"] == "select" and "plateau" in st["jobs"]["train-prim-a"]["result"]["stop"]
    for n in list(st["jobs"]):
        if n.startswith("select-"):
            tag = "isaac-v6b-walkbase-sel" if "control" in n else f"{n.split('-')[2]}-{n.split('-')[3]}-model_50-walkbase-sel"
            log(tmp, n, f"EVAL {tag} seeds 700000+: lifted 0/72, held 0/72; per shape {{}}; kept 0/72 after the shake\n")
    rc, out = run(tmp, "step")
    st = json.loads(Path(tmp, "state.json").read_text())
    assert st["stage"] == "done" and "did not learn" in st["results"]["stop"] and not any(n.startswith("task-") for n in st["jobs"]), out
    assert "stopped early" in Path(tmp, "report.md").read_text()
