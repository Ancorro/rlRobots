"""The OpenHLM-data benchmark contract (owner 2026-10-31): which episodes are train / validation / test, what one
episode's "success" means, and the per-episode record that scripts/check_openhlm.py reads. Standard library only, so
the CPU checks and tests import it without torch.

The data. `OpenHLM/OpenHLM-data` (Hugging Face) holds five LeRobot-format recordings of a real Unitree G1 humanoid in
three recording modes ("full_teleop", "stationary_teleop", "humi"); by their names, an "HLM-12" task set
("g1_HLM-12_*") and a long five-fruit task ("long_g1_5_fruits_*"); what the modes and tasks are exactly is read from
the data (openhlm/fetch.py prints the tasks), not assumed here. Each set is a folder with meta/info.json (fps, the
features and their names), meta/episodes.jsonl, data/chunk-*/episode_*.parquet (observation.state, action, timestamps)
and videos/ (one mp4 per camera per episode). Nothing here is simulated, and no simulator of these scenes exists, so a
policy cannot be run closed loop on them: the benchmark is an open-loop, held-out-episode test of action prediction,
the offline proxy used for real-robot data (LeRobot's validation loss, OpenVLA's action L1), turned into a per-episode
pass/fail that the check script can count.

The test. A policy sees the robot's proprioceptive history (HISTORY_S seconds of observation.state, at the set's frame
rate) at the start of a chunk and must write the next HORIZON_S seconds of actions. Chunks tile the episode without
overlap; at every chunk start the policy is given the recorded history (teacher forcing), so the chunks are independent
open-loop predictions, exactly what an action-chunking policy (ACT, pi0, SmolVLA) does between re-plans.
Action dimensions that no state dimension tracks (a base velocity command, say) are "held" at the last recorded
command by the reference and by the policies' residual base alike: a controller knows what it last commanded; for
every other dimension neither sees past actions.
Error = mean over chunks, steps and action dimensions of |predicted - recorded| / sigma_d, with sigma_d the training
split's standard deviation of action dimension d (dimensions that do not move, std under 1 % of the largest, are left
out). The reference is the hold
baseline, the same chunks filled with the robot's current position (a_{t+k} = s_t for every action dimension that is a
state dimension, else the last recorded action): what a controller that does nothing new would command.
    skill = 1 - error(policy) / error(hold)
An episode is a SUCCESS when skill >= SKILL_PASS: the policy removes at least half of the error of standing still
over the whole episode. 0.5 was fixed before any run (2026-10-31) and is not to be tuned on validation; the evaluation
also reports the rate at 0.25 and 0.75 and the paired comparison with a ridge-regression baseline for context.

The splits. Episodes are assigned by a hash of "<set>/<episode_index>" with SALT, so the assignment is fixed before
anything is downloaded and independent of episode order. Per set, in hash order: the first TEST_PER_SET episodes are
test, the next ~VAL_FRACTION are validation, the rest training. With all five sets that is 5 x 4 = 20 test episodes,
the sample size the check wants (SAMPLE); with fewer sets the per-set count grows so the sample stays 20. Validation
chooses between candidates; every test run is counted (runs/openhlm/test-history.jsonl) so "scored once" can be checked.
"""
import hashlib
import json
import math

REPO_ID = "OpenHLM/OpenHLM-data"
SETS = ("g1_HLM-12_full_teleop", "g1_HLM-12_humi", "g1_HLM-12_stationary_teleop",
        "long_g1_5_fruits_full_teleop", "long_g1_5_fruits_humi")
# The tables' column names differ between LeRobot datasets: OpenHLM-data (snapshot fe7f131, 2026-11-02) uses "state" and
# "actions", Unitree's and LeRobot's own use "observation.state" and "action". The first present is taken (data.resolve_keys).
STATE_KEYS = ("observation.state", "state", "observation.states", "states", "robot_state", "qpos")
ACTION_KEYS = ("action", "actions", "actions.joint", "action.joint")
STATE_KEY, ACTION_KEY = STATE_KEYS[0], ACTION_KEYS[0]  # the names used in records and messages
# Without joint names the hold map (which action dimension is which state dimension) is inferred from the training
# episodes: an action dimension is held from the state dimension that tracks it, at the best lag of up to 0.5 s,
# within HOLD_MATCH of its std (median |a_d(t) - s_j(t + lag)| over frames), one state dimension per action dimension
# (data.infer_hold_map). The map is stored in the checkpoint and printed in the report.
HOLD_MATCH = 0.25

SAMPLE = 20  # test episodes in all (scripts/check_openhlm.py --episodes)
VAL_FRACTION = 0.15  # of the non-test episodes of a set (at least MIN_VAL)
MIN_VAL = 2
SALT = "openhlm-2026-10-31"

HORIZON_S = 1.0  # seconds of actions a policy writes per chunk
HISTORY_S = 1.0 / 3  # seconds of state history it sees (10 frames at 30 fps)
MAX_HORIZON = 60  # frames, whatever the frame rate
SKILL_PASS = 0.5  # success: error at most half of the hold baseline's
SKILL_LEVELS = (0.25, 0.5, 0.75)  # reported beside the pass level
STD_FLOOR = 1e-6  # numerical floor for the normalisation
STD_MIN_FRACTION = 0.01  # action dimensions whose training std is below this fraction of the largest are constant: left out
PASS_RATE = 0.5  # what the check asks of the sample


def episode_rank(set_name, episode):
    """Deterministic position of an episode inside its set's ordering (a hash, 0 <= rank < 2**32)."""
    h = hashlib.sha256(f"{SALT}:{set_name}/{int(episode)}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def test_per_set(n_sets):
    """Test episodes taken from each set so the sample is SAMPLE: the first sets get the remainder."""
    if n_sets < 1:
        raise ValueError("no sets")
    base, rem = divmod(SAMPLE, n_sets)
    return [base + (1 if i < rem else 0) for i in range(n_sets)]


def split_episodes(episodes_by_set, canonical=False):
    """{set: [episode ids]} -> {set: {episode: 'train' | 'val' | 'test'}}. Sets in the order given (it decides which
    sets take the remainder of the 20); a set with fewer episodes than its test share raises. canonical=True uses the
    five-set shares (4 per set) whatever sets are given, for runs on one set that must agree with the full split."""
    sets = list(episodes_by_set)
    out = {}
    shares = [SAMPLE // len(SETS)] * len(sets) if canonical else test_per_set(len(sets))
    for name, n_test in zip(sets, shares):
        eps = sorted(set(int(e) for e in episodes_by_set[name]), key=lambda e: (episode_rank(name, e), e))
        if len(eps) < n_test + MIN_VAL + 1:
            raise ValueError(f"{name}: {len(eps)} episodes, needs at least {n_test + MIN_VAL + 1} (test {n_test}, val {MIN_VAL}, train 1)")
        n_val = max(MIN_VAL, int(round(VAL_FRACTION * (len(eps) - n_test))))
        roles = {}
        for i, e in enumerate(eps):
            roles[e] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        out[name] = roles
    return out


def horizon(fps):
    return max(1, min(MAX_HORIZON, int(round(HORIZON_S * fps))))


def history(fps):
    return max(1, int(round(HISTORY_S * fps)))


def scored_dims(std):
    """Which action dimensions count: std at least STD_MIN_FRACTION of the largest (a list of bools)."""
    top = max(float(v) for v in std) if len(std) else 0.0
    return [float(v) >= max(STD_FLOOR, STD_MIN_FRACTION * top) for v in std]


def skill(err_policy, err_hold):
    """1 - err_policy / err_hold; 0 when the hold baseline is already exact (nothing to remove)."""
    if err_hold <= 0:
        return 0.0
    return 1.0 - float(err_policy) / float(err_hold)


def success(skill_value):
    return bool(skill_value >= SKILL_PASS)


def record(set_name, episode, split, policy, err_policy, err_hold, chunks, length, **extra):
    """One per-episode record, in the shape scripts/check_openhlm.py reads (set, episode, success) plus the numbers
    behind it."""
    if set_name not in SETS:
        raise ValueError(f"{set_name!r} is not an OpenHLM set")
    sk = skill(err_policy, err_hold)
    return dict(set=set_name, episode=int(episode), split=split, policy=policy, success=success(sk), skill=round(sk, 4),
                error=round(float(err_policy), 5), error_hold=round(float(err_hold), 5), chunks=int(chunks),
                length=int(length), **extra)


def wilson(k, n, z=1.96):
    """95 % Wilson interval of a rate (as in docs/results.md)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(b, c):
    """Exact two-sided McNemar p for b (only the first succeeded) and c (only the second), as scripts/goal_common.py."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def summary(records, levels=SKILL_LEVELS):
    """Counts over per-episode records: successes with a Wilson interval, the rate at every skill level, per set."""
    n = len(records)
    wins = sum(bool(r["success"]) for r in records)
    lo, hi = wilson(wins, n)
    out = dict(episodes=n, successes=wins, rate=(wins / n if n else 0.0), wilson95=[round(lo, 3), round(hi, 3)],
               mean_skill=(sum(r["skill"] for r in records) / n if n else 0.0),
               median_skill=(sorted(r["skill"] for r in records)[n // 2] if n else 0.0),
               at_level={str(l): sum(r["skill"] >= l for r in records) for l in levels}, by_set={})
    for s in sorted({r["set"] for r in records}):
        sub = [r for r in records if r["set"] == s]
        out["by_set"][s] = dict(episodes=len(sub), successes=sum(bool(r["success"]) for r in sub),
                                mean_skill=round(sum(r["skill"] for r in sub) / len(sub), 4))
    return out


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
