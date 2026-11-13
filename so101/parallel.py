"""Run several SO-101 sort envs at once, one per CPU core, so episodes collect in parallel while the
decision model decides for all of them in one batched GPU call.

Each worker process owns one env and answers commands over a pipe:
    ("reset", seed)    -> state text
    ("step", action)   -> (state text, reward, done, info); action None = the rule baseline's choice
    ("snapshot", None) -> qpos and tray (mocap) positions, to mirror the env in the main process's viewer
"""
import multiprocessing as mp

import numpy as np


def _worker(pipe, drop_prob):
    from so101.skill_env import SO101Sort
    env = SO101Sort(drop_prob=drop_prob)
    while True:
        cmd, arg = pipe.recv()
        if cmd == "reset":
            env.reset(seed=arg)
            pipe.send(env.state_text())
        elif cmd == "step":
            action = env.oracle_action() if arg is None else arg
            _, r, term, trunc, info = env.step(action)
            info["action"] = [int(a) for a in action]
            pipe.send((env.state_text(), r, term or trunc, info))
        elif cmd == "snapshot":
            pipe.send((env.data.qpos.copy(), env.data.mocap_pos.copy()))
        elif cmd == "close":
            return


class EnvPool:
    def __init__(self, n, drop_prob=0.0):
        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        for _ in range(n):
            here, there = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(there, drop_prob), daemon=True)
            p.start()
            self.pipes.append(here), self.procs.append(p)

    def episodes(self, seeds, choose, on_round=None):
        """Run one episode per seed, len(self.pipes) at a time.
        choose(list of state texts) -> list of actions (None entries = rule baseline).
        Returns [(states, actions, return, success)] in seed order."""
        results, queue = [None] * len(seeds), list(enumerate(seeds))
        live = {}  # worker index -> [episode index, states, actions, return]
        while queue or live:
            for w, pipe in enumerate(self.pipes):  # give idle workers a new episode
                if w not in live and queue:
                    i, seed = queue.pop(0)
                    pipe.send(("reset", seed))
                    live[w] = [i, [pipe.recv()], [], 0.0]
            ws = list(live)
            actions = choose([live[w][1][-1] for w in ws])
            for w, a in zip(ws, actions):
                self.pipes[w].send(("step", a))
            for w in ws:
                state, r, done, info = self.pipes[w].recv()
                ep = live[w]
                ep[2].append(info["action"])
                ep[3] += r
                if done:
                    results[ep[0]] = (ep[1], ep[2], ep[3], bool(info["is_success"]))
                    del live[w]
                else:
                    ep[1].append(state)
            if on_round:
                on_round(sum(r is not None for r in results), len(seeds))
        return results

    def snapshot(self, w=0):
        self.pipes[w].send(("snapshot", None))
        return self.pipes[w].recv()

    def close(self):
        for pipe in self.pipes:
            pipe.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)


def mirror(display_env, snap):
    """Copy a worker's state into the main process's env, for the viewer."""
    import mujoco
    qpos, mocap_pos = snap
    display_env.data.qpos[:] = qpos
    display_env.data.mocap_pos[:] = mocap_pos
    mujoco.mj_forward(display_env.model, display_env.data)
    return np.asarray(qpos)
