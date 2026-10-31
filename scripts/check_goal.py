"""The whole G1 goal in one check: day 1 (benchmarks fixed, baselines), day 2 (learned grab chosen on val), day 3
(held-out test beats the GPU best). Runs all three and prints where the goal stands. CPU only, no Isaac.

  python scripts/check_goal.py [RESULTS ...]          exit 0 only when all three days pass
  python scripts/check_goal.py --day=2 [RESULTS ...]  one day
"""
import sys

import check_day1
import check_day2
import check_day3

DAYS = {1: check_day1.main, 2: check_day2.main, 3: check_day3.main}


def main(argv):
    day = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--day=")), None)
    paths = [a for a in argv if not a.startswith("--")]
    done = {d: f(paths) for d, f in DAYS.items() if day in (None, d)}
    print("\n== goal: " + ", ".join(f"day {d} {'passed' if ok else 'not yet'}" for d, ok in done.items()))
    return all(done.values())


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
