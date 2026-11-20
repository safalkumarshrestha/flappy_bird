"""Final evaluation of trained runs: greedy/deterministic policy, no exploration.

  python evaluate.py runs/ppo/baseline runs/dqn/baseline     # every finished seed dir under these paths
  python evaluate.py runs --episodes 100 --cap 10000 --gif

Writes final_eval.json (mean/std/median/max pipes over `episodes`, capped at `cap` pipes) and optionally
agent.gif next to each model. Runs that already have final_eval.json are skipped unless --force.
"""
import argparse
import importlib
import json
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from envs.flappy import make_env, run_policy

FINAL_SEED = 20_000  # differs from the periodic-eval seed in train.py


def above_screen_steps(ret, pipes, length, crashed):
    """Steps spent above the screen, recovered from the reward identity
    ret = 0.1 * (length - pipes - crashed - k) + pipes - crashed - 0.5 * k.
    A pipe scored on the crash step (its +1 is replaced by -1) shifts the estimate by exactly +1.5."""
    raw = (0.1 * (length - pipes - crashed) + pipes - crashed - ret) / 0.6
    return np.round(np.where(np.isclose(raw % 1, 0.5, atol=0.05), raw - 1.5, raw))


def record_gif(policy, path, max_pipes=30):
    env = make_env(max_steps=None, score_limit=max_pipes, render_mode="rgb_array")
    obs, _ = env.reset(seed=FINAL_SEED)
    frames, done = [env.render()[::2, ::2]], False
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs))
        done = term or trunc
        frames.append(env.render()[::2, ::2])  # half resolution keeps the GIF small
    env.close()
    imageio.mimsave(path, frames[::2], duration=1000 / 15, loop=0)  # every 2nd frame at 15 fps = real time
    return info["score"]


def evaluate(job):
    run_dir, episodes, cap, gif = job
    cfg = json.loads((run_dir / "config.json").read_text())
    policy = importlib.import_module(f"agents.{cfg['algo']}").load_policy(run_dir, cfg)
    ret, pipes, length = np.array(run_policy(policy, episodes, FINAL_SEED, score_limit=cap, max_steps=50 * cap)).T
    res = dict(episodes=episodes, cap=cap, mean=pipes.mean(), std=pipes.std(), median=float(np.median(pipes)),
               max=pipes.max(), min=pipes.min(), crash_rate=(pipes < cap).mean(),
               above_screen_per_1k=1000 * above_screen_steps(ret, pipes, length, pipes < cap).sum() / length.sum(),
               pipes=pipes.astype(int).tolist())
    if gif:
        res["gif_pipes"] = record_gif(policy, run_dir / "agent.gif")
    (run_dir / "final_eval.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{run_dir}: {res['mean']:.1f} ± {res['std']:.1f} pipes (median {res['median']:.0f}, max {res['max']:.0f})",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--cap", type=int, default=1000)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    runs = sorted({s.parent for path in args.paths for s in path.rglob("summary.json")})
    jobs = [(r, args.episodes, args.cap, args.gif) for r in runs if args.force or not (r / "final_eval.json").exists()]
    print(f"{len(jobs)} runs to evaluate")
    with Pool(args.workers) as pool:
        list(pool.imap_unordered(evaluate, jobs))
