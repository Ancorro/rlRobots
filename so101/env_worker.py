"""Run skill-level episodes side by side: one env per worker process, stepped in lockstep.

`Envs(task, n, local)` gives n envs: `local` (the main process's env, the one in the viewer) plus n - 1
worker processes. Workers talk pickles over stdin/stdout (plain subprocesses, so Windows' spawn
does not re-run the training script). `run(seeds, choose)` plays one episode per env;
choose(states) -> actions is called once per round with every unfinished env's state text,
so the decision model sees one batch. choose=None uses each env's rule baseline.
"""
import os
import pickle
import subprocess
import sys


def make_env(task):
    if task == "sort":
        from so101.skill_env import SO101Sort
        return SO101Sort(drop_prob=0.3)
    from so101.tabletop import SO101Tabletop
    return SO101Tabletop()


class Local:
    def __init__(self, env):
        self.env = env

    def send(self, cmd, arg=None):
        self.reply = handle(self.env, cmd, arg)

    def recv(self):
        return self.reply


class Worker:
    def __init__(self, task):
        self.p = subprocess.Popen([sys.executable, "-m", "so101.env_worker", task],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def send(self, cmd, arg=None):
        pickle.dump((cmd, arg), self.p.stdin)
        self.p.stdin.flush()

    def recv(self):
        return pickle.load(self.p.stdout)

    def close(self):
        self.send("quit")
        self.p.wait(timeout=30)


def handle(env, cmd, arg):
    if cmd == "reset":
        env.reset(seed=arg)
        return env.state_text()
    if cmd == "oracle":
        return env.oracle_action()
    if cmd == "step":
        _, r, term, trunc, info = env.step(arg)
        return env.state_text(), r, term or trunc, info["is_success"]


class Envs:
    def __init__(self, task, n, local):
        self.envs = [Local(local)] + [Worker(task) for _ in range(n - 1)]

    def __len__(self):
        return len(self.envs)

    def _all(self, envs, cmd, args):
        for e, a in zip(envs, args):  # workers first, so the local env overlaps with them
            if not isinstance(e, Local):
                e.send(cmd, a)
        for e, a in zip(envs, args):
            if isinstance(e, Local):
                e.send(cmd, a)
        return [e.recv() for e in envs]

    def run(self, seeds, choose=None):
        """One episode per seed (len(seeds) <= n). Returns [(states, actions, return, success)]."""
        envs = self.envs[:len(seeds)]
        cur = self._all(envs, "reset", seeds)
        out = [([], [], 0.0, False) for _ in seeds]
        live = list(range(len(seeds)))
        while live:
            acts = (self._all([envs[i] for i in live], "oracle", [None] * len(live)) if choose is None
                    else choose([cur[i] for i in live]))
            replies = self._all([envs[i] for i in live], "step", acts)
            still = []
            for i, a, (s, r, done, ok) in zip(live, acts, replies):
                ss, aa, ret, _ = out[i]
                ss.append(cur[i]), aa.append(a)
                out[i] = (ss, aa, ret + r, ok)
                cur[i] = s
                if not done:
                    still.append(i)
            live = still
        return out

    def close(self):
        for e in self.envs[1:]:
            e.close()


if __name__ == "__main__":
    proto_in, proto_out = sys.stdin.buffer, os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # library prints go to stderr, not into the pickle stream
    sys.stdout = sys.stderr
    env = make_env(sys.argv[1])
    while True:
        cmd, arg = pickle.load(proto_in)
        if cmd == "quit":
            break
        pickle.dump(handle(env, cmd, arg), proto_out)
        proto_out.flush()
