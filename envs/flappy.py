"""FlappyBird-v0 factory and feature helpers.

Reward structure (flappy-bird-gymnasium 0.4.0, use_lidar=False), per step:
  +0.1  bird alive
  +1.0  a pipe was passed this step (replaces the +0.1)
  -0.5  bird is above the top of the screen (overrides both of the above)
  -1.0  crash into ground or pipe -> terminated=True
The game never ends on its own, so episodes are truncated by a TimeLimit (max_steps)
and optionally by score_limit. One pipe takes ~37 steps (pipes move 4 px/step).

Observation (12 floats, normalised): for the 3 visible pipes sorted by x
[x, top-pipe bottom y, bottom-pipe top y] * 3, then bird y, bird vy / 10, rotation / 90.
"""
import sys

import gymnasium as gym
import numpy as np

import flappy_bird_gymnasium  # noqa: F401  (registers FlappyBird-v0)

if sys.platform == "win32":
    # Windows 11 EcoQoS throttles background processes to ~1/10 speed after a few seconds; opt this process out.
    import ctypes
    from ctypes import wintypes

    class _Throttle(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    _k = ctypes.WinDLL("kernel32")
    _k.GetCurrentProcess.restype = wintypes.HANDLE
    _k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _s = _Throttle(1, 0x1 | 0x4, 0)  # control EXECUTION_SPEED + timer resolution, state 0 = never throttle
    _k.SetProcessInformation(_k.GetCurrentProcess(), 4, ctypes.byref(_s), ctypes.sizeof(_s))  # 4 = ProcessPowerThrottling

W, H = 288, 512
PLAYER_X, PLAYER_H, PIPE_W, VEL_MAX = int(W * 0.2), 24, 52, 10

# Per-feature mean/std of observations over 200k steps of the scripted policy below (5% random actions).
# The decisive features (gap and bird y) have std ~0.05 vs ~0.5 for velocity, so networks standardise inputs
# with these frozen statistics (like VecNormalize with fixed stats).
OBS_MEAN = np.array([0.409, 0.262, 0.457, 0.812, 0.158, 0.677, 0.991, 0.03, 0.938, 0.312, -0.148, 0.248], np.float32)
OBS_STD = np.array([0.33, 0.044, 0.044, 0.213, 0.134, 0.267, 0.031, 0.085, 0.173, 0.068, 0.537, 0.181], np.float32)


def make_env(max_steps=10_000, score_limit=None, render_mode=None):
    return gym.make("FlappyBird-v0", use_lidar=False, score_limit=score_limit,
                    render_mode=render_mode, max_episode_steps=max_steps)


def pipe_features(obs):
    """(dx, dy, vy) in pixels for the next pipe the bird has not cleared yet.

    dx = pipe left edge - bird x, dy = bird centre - gap centre (+ means bird is below), vy = velocity (+ is down).
    """
    xs = obs[[0, 3, 6]] * W
    i = int(np.argmax(xs + PIPE_W > PLAYER_X))  # pipes are sorted by x
    gap_centre = (obs[3 * i + 1] + obs[3 * i + 2]) * H / 2
    return xs[i] - PLAYER_X, obs[9] * H + PLAYER_H / 2 - gap_centre, round(obs[10] * VEL_MAX)  # vy is an int


def run_policy(policy, episodes, seed, **env_kw):
    """Roll out `policy(obs) -> action`; returns list of (return, pipes, length)."""
    env, out = make_env(**env_kw), []
    obs, _ = env.reset(seed=seed)
    for _ in range(episodes):
        ret, done, n = 0.0, False, 0
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            ret, n, done = ret + r, n + 1, term or trunc
        out.append((ret, info["score"], n))
        obs, _ = env.reset()
    env.close()
    return out


if __name__ == "__main__":
    import time

    # 1. reward values are exactly the documented set
    env, rng, seen = make_env(max_steps=2000), np.random.default_rng(0), set()
    obs, _ = env.reset(seed=0)
    t, steps = time.perf_counter(), 50_000
    for _ in range(steps):
        obs, r, term, trunc, _ = env.step(int(rng.random() < 0.1))
        seen.add(round(float(r), 2))
        if term or trunc:
            obs, _ = env.reset()
    print(f"env speed: {steps / (time.perf_counter() - t):,.0f} steps/s; rewards seen: {sorted(seen)}")
    assert seen <= {0.1, 1.0, -0.5, -1.0}, seen

    # 2. same seed + same actions -> identical trajectory
    def traj(seed):
        e = make_env(); o, _ = e.reset(seed=seed); xs = [o]
        for k in range(300):
            o, _, term, trunc, _ = e.step(k % 7 == 0); xs.append(o)
            if term or trunc:
                o, _ = e.reset(); xs.append(o)
        return np.array(xs)
    assert np.array_equal(traj(1), traj(1)) and not np.array_equal(traj(1), traj(2))
    print("seeding: reproducible")

    # 3. features at reset: first pipe at x=W, bird at screen centre
    dx, dy, vy = pipe_features(make_env().reset(seed=0)[0])
    assert dx == W - PLAYER_X and vy == -9, (dx, dy, vy)

    # 4. baselines; the heuristic only works if pipe_features finds the right pipe
    heuristic = lambda o: int(pipe_features(o)[1] > 10 and pipe_features(o)[2] >= 0)
    for name, pol in [("never flap", lambda o: 0), ("random p=0.5", lambda o: int(rng.random() < 0.5)),
                      ("random p=0.1", lambda o: int(rng.random() < 0.1)), ("heuristic", heuristic)]:
        res = np.array(run_policy(pol, 20, seed=0, max_steps=5000))
        print(f"{name:13s} return {res[:, 0].mean():7.2f}  pipes {res[:, 1].mean():6.2f} (max {res[:, 1].max():.0f})  len {res[:, 2].mean():6.0f}")
    assert res[:, 1].mean() > 10, "heuristic should pass pipes -> pipe_features is broken"
    print("env OK")
