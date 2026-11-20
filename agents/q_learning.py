"""Tabular Q-learning on the discretised (dx, dy, vy) features of the next pipe."""
import math

import numpy as np

from envs.flappy import make_env, pipe_features


def epsilon(cfg, step):
    """Linear, or exponential (99% decayed by eps_decay_steps), decay from eps_start to eps_end."""
    frac = min(step / cfg["eps_decay_steps"], 1.0)
    if cfg["eps_schedule"] == "exp":
        frac = (1 - math.exp(-5 * step / cfg["eps_decay_steps"])) / (1 - math.exp(-5))
        frac = min(frac, 1.0)
    return cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])


class Discretizer:
    """Uniform bins over clipped (dx, dy, vy) ranges -> index tuple into the Q-table."""

    def __init__(self, bins, ranges):
        self.spec = [(lo, (hi - lo) / n, n) for n, (lo, hi) in zip(bins, ranges)]

    def __call__(self, obs):
        return tuple(min(max(int((f - lo) / w), 0), n - 1) for f, (lo, w, n) in zip(pipe_features(obs), self.spec))


def train(cfg, log):
    rng = np.random.default_rng(cfg["seed"])
    disc, q = Discretizer(cfg["bins"], cfg["ranges"]), np.zeros((*cfg["bins"], 2))
    alpha, gamma = cfg["alpha"], cfg["gamma"]
    policy = lambda o: int(np.argmax(q[disc(o)]))  # ties (unseen states) -> 0 = don't flap

    env = make_env(cfg["max_steps"])
    obs, _ = env.reset(seed=cfg["seed"])
    s, ret, length = disc(obs), 0.0, 0
    for step in range(1, cfg["total_steps"] + 1):
        eps = epsilon(cfg, step)
        a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(q[s]))
        obs, r, term, trunc, info = env.step(a)
        s2 = disc(obs)
        target = r if term else r + gamma * q[s2].max()  # truncation still bootstraps
        q[s + (a,)] += alpha * (target - q[s + (a,)])
        s, ret, length = s2, ret + r, length + 1
        if term or trunc:
            log.episode(step, ret, info["score"], length, epsilon=round(eps, 4))
            obs, _ = env.reset()
            s, ret, length = disc(obs), 0.0, 0
        log.tick(step, policy)

    np.save(log.dir / "q_table.npy", q)
    return {"states": int(np.prod(cfg["bins"])), "states_visited": int((q != 0).any(-1).sum()), "q_table_bytes": q.nbytes}


def load_policy(run_dir, cfg):
    q, disc = np.load(run_dir / "q_table.npy"), Discretizer(cfg["bins"], cfg["ranges"])
    return lambda o: int(np.argmax(q[disc(o)]))
