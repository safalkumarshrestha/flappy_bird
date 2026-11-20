# Flappy Bird RL: tabular Q-learning vs DQN vs PPO

Three agents trained on `FlappyBird-v0` from [`flappy-bird-gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium)
(`use_lidar=False`, 12-dim feature observation), compared on a common environment-step axis.
The write-up with all results is in **[REPORT.md](REPORT.md)**; figures and the full results table are in `figures/`.

| final evaluation (100 greedy episodes × 3 seeds, cap 1000) | mean pipes | steps to 50 pipes |
|---|---|---|
| PPO | 997.6 | 326k |
| Double DQN | 170.9 | 525k |
| Q-learning | 68.7 | 1.10M |
| DQN | 6.6 (unstable; best periodic eval ≈96) | 700k |

## Setup

```bash
python -m venv .venv            # Python 3.11 used here
.venv/Scripts/activate          # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m envs.flappy           # environment self-check: rewards, seeding, speed, baselines
```

## Watch or play

```bash
python play.py
```

A menu in the game window lists every evaluated agent with its final test score, plus "You (keyboard)".
Pick with the mouse or Up/Down, start with a click or Enter, and change the speed (1x to 8x) with Left/Right.
When you play yourself, space, up arrow or a click flaps. After a crash, Enter plays again; Esc returns to the
menu, and Esc in the menu quits.

## Layout

| path | what |
|---|---|
| `envs/flappy.py` | env factory (`make_env`), next-pipe features for Q-learning, frozen obs statistics, rollout helper, self-check |
| `agents/q_learning.py` | tabular Q-learning on discretised (dx, dy, vy); ε schedules (shared with DQN) |
| `agents/dqn.py` | DQN / Double DQN: replay, target net (hard or Polyak), Huber, Adam, grad clipping |
| `agents/ppo.py` | PPO from scratch: clipped surrogate, value loss, entropy, GAE(λ), adv. normalisation, SyncVectorEnv |
| `agents/sb3_ppo.py` | sanity check only: Stable-Baselines3 PPO with the same settings (`pip install stable-baselines3==2.9.0`) |
| `configs/*.json` | baseline hyperparameters + the one-at-a-time `sweep` for each algorithm |
| `train.py` | runs configs × seeds in parallel; logs `train.csv` (per episode) and `eval.csv` (greedy policy) |
| `evaluate.py` | final 100-episode evaluation (`final_eval.json`) and `agent.gif` |
| `plot.py` | every figure in `figures/` and the results table `figures/results.md` |
| `play.py` | game window with a menu: pick a trained agent to watch, or play yourself |

Each run lives in `runs/<algo>/<config tag>/seed<k>/` with `config.json`, `train.csv`, `eval.csv`, `summary.json`
(wall time, Q-table size) and the model (`q_table.npy` or `model.pt`). Finished runs are skipped on re-launch.
`evaluate.py` adds `final_eval.json` and `agent.gif` (one greedy episode, stopped at 30 pipes).

## Environment and reward

| event (per step) | reward |
|---|---|
| bird alive | +0.1 |
| passed a pipe (replaces the +0.1) | +1.0 |
| bird above the top of the screen (overrides the above) | −0.5 |
| crash into pipe or ground (episode terminates) | −1.0 |

The game never ends by itself, so training episodes are truncated at 10,000 steps (≈265 pipes); truncation is
bootstrapped, termination is not. Periodic evaluation: 10 greedy episodes every 25k steps, capped at 100 pipes.
Final evaluation: 100 greedy episodes, capped at 1000 pipes. Seeds fix NumPy, PyTorch and the env RNG.

## Reproduce

```bash
# everything in the report (≈2 h on a 14-core laptop CPU, run the lines concurrently)
python train.py --algo q_learning --sweep --workers 3     # baseline + α, γ, ε schedule/start/length, bins; 3 seeds, 3M steps
python train.py --algo dqn --sweep --workers 6            # baseline (2M) + 1M sweep_baseline + lr, target update, buffer, ε decay (1M)
python train.py --algo dqn --set double=true --tag double --workers 3                                   # Double DQN, 2M
python train.py --algo ppo --sweep --workers 4            # baseline (2M) + 1M sweep_baseline + clip, entropy, GAE λ, lr (1M)
python train.py --algo dqn --set obs_norm=false total_steps=1000000 --tag obs_norm=false               # ablation
python train.py --algo ppo --set obs_norm=false total_steps=1000000 --tag obs_norm=false               # ablation
python train.py --algo sb3_ppo                            # optional SB3 sanity check, 1M, 3 seeds

python evaluate.py runs/q_learning/baseline runs/dqn/baseline runs/dqn/double runs/ppo/baseline --gif
python plot.py
```

Single runs: `python train.py --algo ppo --seeds 0 --workers 1 --set lr=0.0003 --tag my_test`.

## Notes

- **Windows:** Windows 11 throttles background processes to ~1/10 speed after a few seconds (EcoQoS).
  `envs/flappy.py` opts every process out; without it DQN ran at ~100 steps/s instead of ~1,000.
- **Input standardisation** (`obs_norm`): DQN and PPO standardise the 12 observations with frozen per-feature
  mean/std (`envs/flappy.py`). The deciding features (gap and bird height) have std ≈0.05 against ≈0.5 for
  velocity; without standardisation both networks learn several times slower (see the ablation in the report).
