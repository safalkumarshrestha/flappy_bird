"""Train an agent.

  python train.py --algo q_learning --seeds 0 1 2              # baseline config, seeds run in parallel
  python train.py --algo dqn --set double=true --tag double    # override config values (JSON or plain strings)
  python train.py --algo ppo --sweep --workers 12              # baseline + every one-at-a-time variant in the config's "sweep"
                                                               # (variants run for "sweep_steps" if the config sets it)

Each run writes runs/<algo>/<tag>/seed<k>/: config.json, train.csv (per episode), eval.csv (greedy policy,
every eval_every env steps), summary.json (written last, marks the run done; done runs are skipped) and the model.
"""
import argparse
import csv
import importlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from envs.flappy import run_policy

COMMON = dict(max_steps=10_000,      # training episode TimeLimit (~265 pipes)
              eval_every=25_000, eval_episodes=10, eval_cap=100)  # periodic greedy eval, capped at 100 pipes
EVAL_SEED = 10_000                   # every periodic eval sees the same pipe layouts


class Logger:
    def __init__(self, run_dir, cfg):
        self.dir, self.cfg, self.t0, self.next_eval, self.files = run_dir, cfg, time.time(), 0, {}

    def _write(self, name, row):
        if name not in self.files:
            f = open(self.dir / f"{name}.csv", "w", newline="")
            self.files[name] = (f, csv.DictWriter(f, fieldnames=list(row)))
            self.files[name][1].writeheader()
        self.files[name][1].writerow(row)

    def episode(self, step, ret, pipes, length, **extra):
        self._write("train", dict(step=step, ret=round(float(ret), 2), pipes=int(pipes), length=int(length),
                                  time=round(time.time() - self.t0, 1), **extra))

    def tick(self, step, policy):
        """Evaluate `policy(obs) -> action` greedily once every eval_every env steps."""
        if step < self.next_eval:
            return
        self.next_eval += self.cfg["eval_every"]
        cap = self.cfg["eval_cap"]
        res = np.array(run_policy(policy, self.cfg["eval_episodes"], EVAL_SEED, score_limit=cap, max_steps=50 * cap))
        pipes = res[:, 1]
        self._write("eval", dict(step=step, time=round(time.time() - self.t0, 1), ret=round(res[:, 0].mean(), 2),
                                 pipes=pipes.mean(), pipes_std=round(pipes.std(), 2), pipes_max=pipes.max(),
                                 crash_rate=(pipes < cap).mean()))
        for f, _ in self.files.values():
            f.flush()
        print(f"{self.dir}  step {step:>9,}  eval pipes {pipes.mean():6.1f} ± {pipes.std():5.1f}  "
              f"({time.time() - self.t0:5.0f}s)", flush=True)


def run(job):
    algo, cfg, run_dir = job
    if (run_dir / "summary.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    log = Logger(run_dir, cfg)
    extra = importlib.import_module(f"agents.{algo}").train(cfg, log) or {}
    for f, _ in log.files.values():
        f.close()
    (run_dir / "summary.json").write_text(json.dumps({"wall_time": round(time.time() - log.t0, 1), **extra}, indent=1))


def tag_of(k, v):
    return f"{k}={'x'.join(map(str, v)) if isinstance(v, list) else v}"


def parse(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def make_jobs(args):
    base = {**COMMON, **json.loads(Path(f"configs/{args.algo}.json").read_text())}
    sweep, sweep_steps = base.pop("sweep", {}), base.pop("sweep_steps", base["total_steps"])  # variants may be shorter
    for kv in args.set:
        k, v = kv.split("=", 1)
        base[k] = parse(v)
    variants = [(args.tag, {})]
    if args.sweep:  # shorter variants get a same-length baseline as reference (lr annealing depends on total_steps)
        variants += [("sweep_baseline", {"total_steps": sweep_steps})] if sweep_steps != base["total_steps"] else []
        variants += [(tag_of(k, v), {k: v, "total_steps": sweep_steps}) for k, vs in sweep.items() for v in vs]
    return [(args.algo, {**base, **over, "seed": s, "algo": args.algo}, Path(args.out) / args.algo / tag / f"seed{s}")
            for tag, over in variants for s in args.seeds]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["q_learning", "dqn", "ppo", "sb3_ppo"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default="runs")
    args = p.parse_args()
    jobs = make_jobs(args)
    print(f"{len(jobs)} runs")
    if args.workers == 1:
        for j in jobs:
            run(j)
    else:
        with Pool(args.workers, maxtasksperchild=1) as pool:
            list(pool.imap_unordered(run, jobs, chunksize=1))
