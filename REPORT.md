# Flappy Bird RL: tabular Q-learning vs DQN vs PPO

Generated from `runs/` (via `figures/results.md`). Environment: `FlappyBird-v0`, `use_lidar=False`, 12-dim observation. Every configuration was trained with 3 seeds. Final evaluation = 100 greedy episodes per seed. The four baseline configs (Q-learning, DQN, Double DQN, PPO) were evaluated with a cap of 1000 pipes (section 1). The sweep configs and SB3 were evaluated with a cap of 10,000 pipes, so their numbers are not comparable with the baselines; they are reported separately at the end of section 2, next to the periodic evals (10 episodes, cap 100). Budgets differ by algorithm (Q-learning 3M steps, DQN and PPO 2M), so the headline comparison is on the shared step axis only up to 2M.

## Setup and implementations

**Environment and protocol.** Rewards per step: +0.1 alive, +1.0 on passing a pipe, −0.5 above the top of the screen, −1.0 on a crash (terminates). The game never ends by itself, so training episodes are truncated at 10,000 steps (about 265 pipes); truncation bootstraps, termination does not. Periodic eval: 10 greedy episodes every 25k steps, capped at 100 pipes. Final eval (`evaluate.py`): 100 greedy episodes from the same fixed evaluation seed for every run, cap 1000 for the four baselines and 10,000 for the sweep configs and SB3. All agents are compared on the same environment-step axis.

| agent | file | what it is | baseline settings |
|---|---|---|---|
| Q-learning | `agents/q_learning.py` | Tabular Q-learning on three discretised features of the next pipe (horizontal distance, vertical offset, bird velocity). Uniform bins over clipped ranges; unseen states default to "don't flap". | 12×24×20 bins, α=0.5, γ=0.99, ε 0.05→0.001 linear, 3M steps |
| DQN / Double DQN | `agents/dqn.py` | From scratch in PyTorch. MLP 12→128→128→2 (ReLU), replay buffer, target network, Huber loss, Adam, grad-norm clipping. `double=true` selects a′ with the online net and evaluates it with the target net. | lr 2.5e-4, γ=0.99, buffer 100k, batch 64, train every 4 steps, hard target copy every 1000, ε 1.0→0.01 over 100k, 2M steps |
| PPO | `agents/ppo.py` | From scratch. Separate actor and critic MLPs (64×64, tanh, orthogonal init), clipped surrogate, value loss, entropy bonus, GAE(λ), per-minibatch advantage normalisation, linear LR annealing, 8 envs in a `SyncVectorEnv`. Time-limit truncation is bootstrapped from the true final observation. | lr 1e-3, clip 0.2, λ=0.95, γ=0.99, ent_coef 0.05, 256 steps × 8 envs, 10 epochs, 4 minibatches, 2M steps |
| SB3 PPO | `agents/sb3_ppo.py` | Sanity check only: Stable-Baselines3 PPO (2.9.0) with the same hyperparameters, network shape, activation and input standardisation as the from-scratch PPO, logged with the same protocol. | as PPO, 1M steps |

DQN and PPO standardise the 12 observations with frozen per-feature statistics (`obs_norm`); this is the switch ablated in section 2. Sweep runs use 1M steps (`sweep_baseline`) for DQN and PPO, and 3M for Q-learning. Each run is seeded for NumPy, PyTorch and the environment, and every configuration uses seeds 0, 1 and 2.

## 1. Headline result

| algo | final eval (mean ± std across seeds) | median | crash % | steps to 10 pipes | steps to 50 pipes | wall time / run |
|---|---|---|---|---|---|---|
| PPO | **997.6 ± 3.4** | 1000 | 0.3 | 176k | 326k | 32.7 min |
| Double DQN | 170.9 ± 160.8 | 56.5 | 96 | 375k | 525k | 56.1 min |
| Q-learning | 68.7 ± 20.6 | 39 | 100 | 275k | 1100k | 3.1 min |
| DQN (vanilla) | 6.6 ± 5.5 | 2 | 100 | 500k | 700k | 53.0 min |
| SB3 PPO (sanity check, 1M steps) | 9645.1 ± 385.4 at cap 10,000 (not comparable) | 10000 | 4 | 225k | 350k | 20.4 min |

"Crash %" is the share of final-eval episodes that ended below the cap (`evaluate.py`: `pipes < cap`).

Per-seed final eval (mean pipes over 100 greedy episodes, from `runs/*/seed*/final_eval.json`):

| algo | seed 0 | seed 1 | seed 2 | crash rate per seed |
|---|---|---|---|---|
| PPO | 1000.0 | 992.9 | 1000.0 | 0%, 1%, 0% |
| Double DQN | 24.2 | 93.7 | 394.8 | 100%, 100%, 88% |
| Q-learning | 61.6 | 96.8 | 47.8 | 100% each |
| DQN | 2.6 | 14.4 | 2.8 | 100% each |

- PPO is the only agent that reaches the cap reliably. 299 of its 300 final episodes hit 1000 pipes; the one crash (seed 1) came after 287 pipes.
- Vanilla DQN is unstable. Its best periodic eval reached about 96 pipes, but the final policies average 2.6, 14.4 and 2.8.
- Double DQN is better on average (170.9) but the spread is large (std 160.8): one seed averages 395 and is the only agent besides PPO to reach the cap in any episode (12 of its 100 episodes), while the other two seeds average 24 and 94.
- Q-learning is cheap (about 3 min per run) and reaches 50 pipes in all seeds, but slowly (1.1M steps) and every final episode crashes. Its table has 5,760 cells and about 3,000 of them were visited per seed (2,964, 3,132, 3,067).
- The from-scratch PPO performs in the same range as Stable-Baselines3 PPO run with the same settings for 1M steps, which is the check that the implementation is sound. Both reach 100.0 on the last periodic eval and reach 50 pipes in 3/3 seeds (our `sweep_baseline` at 301k steps, SB3 at 350k), but that eval is capped at 100 pipes and cannot separate them. On the cap-10,000 final eval (section 2) SB3 averages 9645 (seeds 9917 / 9100 / 9918, 4% crashes) and ours 7425 (9853 / 3062 / 9360, 36%). Two of our seeds match SB3's, so the gap comes from one seed; with three seeds it is a hint that ours is less reliable, not a result.

## 2. Sweeps (one hyperparameter at a time)

Metric: last periodic eval (mean pipes ± across seeds) and how many seeds reached 50 pipes.

### PPO
| setting | last eval | best eval | reached 50 pipes |
|---|---|---|---|
| sweep_baseline | 100.0 ± 0.0 | 100.0 | 3/3 |
| clip=0.1 | 100.0 ± 0.0 | 100.0 | 3/3 |
| clip=0.3 | 100.0 ± 0.0 | 100.0 | 3/3 (fastest, 225k) |
| gae_lambda=0.9 | 100.0 ± 0.0 | 100.0 | 3/3 |
| gae_lambda=0.99 | 88.5 ± 7.5 | 100.0 | 3/3 |
| ent_coef=0.1 | 98.3 ± 2.3 | 100.0 | 3/3 |
| ent_coef=0.01 | 31.8 ± 43.5 | 34.0 | 1/3 |
| ent_coef=0.0 | 1.0 ± 0.0 | 1.0 | 0/3 (never learns) |
| lr=0.003 | 96.8 ± 4.4 | 99.4 | 3/3 |
| lr=0.00025 | 15.3 ± 20.2 | 20.8 | 1/3 |
| obs_norm=false | 8.2 ± 1.5 | 17.7 | 0/3 |

On the periodic eval PPO looks insensitive to clip range and GAE λ, but that eval saturates at 100 pipes. The cap-10,000 final eval (end of this section) shows large seed-to-seed differences inside these settings, so they cannot be ranked from three seeds. Only `gae_lambda=0.9` is high on all three seeds. PPO is clearly sensitive to the entropy coefficient (`0.0` never learns on any seed), a too-small learning rate (`lr=0.00025`: 1 / 38 / 1 on the final eval) and observation normalisation (7 / 7 / 6).

### DQN
| setting | last eval | best eval | reached 50 pipes |
|---|---|---|---|
| target_update=250 | 50.4 ± 24.9 | 84.8 | 3/3 (500k) |
| buffer_size=10000 | 50.0 ± 32.6 | 86.9 | 3/3 (500k) |
| eps_decay_steps=20000 | 40.2 ± 33.2 | 91.2 | 3/3 (300k) |
| sweep_baseline | 33.2 ± 10.6 | 75.7 | 3/3 (700k) |
| lr=0.0001 | 17.7 ± 10.3 | 75.5 | 3/3 (775k) |
| target_update=5000 | 17.5 ± 13.5 | 73.6 | 3/3 (675k) |
| eps_decay_steps=500000 | 17.7 ± 16.8 | 27.6 | 0/3 |
| buffer_size=500000 | 8.3 ± 2.3 | 53.7 | 1/3 |
| obs_norm=false | 6.2 ± 1.8 | 26.8 | 0/3 |
| lr=0.001 | 3.6 ± 2.5 | 46.4 | 0/3 |

On the periodic eval, faster target updates (`target_update=250`) and a fast ε decay (`eps_decay_steps=20000`) look best. The final eval does not confirm it: each has one standout seed, while the other two sit around or below the `sweep_baseline` range (217 / 12 / 41 and 5 / 472 / 25, against 39 / 21 / 27). The harmful settings hold on every seed: a large buffer (5 / 12 / 6), a high learning rate (5 / 4 / 2) and no observation normalisation (5 / 5 / 4) are all below every `sweep_baseline` seed. Last eval is noisy (e.g. `lr=0.0001` ends at 17.7 but peaked at 75.5), so read best eval as the ceiling a setting reached, not its stable performance.

### Q-learning
| setting | last eval | best eval | reached 50 pipes |
|---|---|---|---|
| gamma=0.95 | 60.8 ± 14.5 | 85.8 | 3/3 (1300k) |
| gamma=0.9 | 53.4 ± 27.7 | 98.2 | 3/3 (1375k) |
| baseline | 44.0 ± 12.7 | 81.7 | 3/3 (1100k) |
| eps_schedule=exp | 34.7 ± 14.0 | 64.0 | 3/3 (1050k) |
| bins=6x12x10 | 24.2 ± 29.7 | 91.1 | 3/3 (525k, fastest) |
| eps_decay_steps=300000 | 18.1 ± 9.1 | 22.8 | 0/3 |
| bins=24x48x20 | 10.3 ± 2.0 | 20.3 | 0/3 |
| alpha=0.1 | 9.7 ± 4.6 | 16.6 | 0/3 |
| alpha=0.05 | 4.7 ± 2.4 | 5.1 | 0/3 |
| eps_start=1.0 | 1.8 ± 0.4 | 2.0 | 0/3 |

Coarser bins reach 50 pipes sooner; the finest bins do not get there in this budget. Smaller learning rates are too slow. `eps_start=1.0` fails completely, with about 30 above-screen steps per 1k late in training (the bird keeps flying off the top).

### Final eval of the sweep configs (cap 10,000 pipes)

Each sweep run also has a 100-episode greedy final eval, but with a cap of 10,000 pipes instead of 1000, so it separates settings that all saturate the 100-pipe periodic eval. Sweep runs are 1M steps (3M for Q-learning). The four baselines (cap 1000, section 1) are not repeated. Computed by calling `plot.tables` on `runs/`; `figures/results.md` was written on 2026-10-18 21:31, before 93 of these 105 `final_eval.json` files (all 31 sweep configs × 3 seeds), so it still shows dashes for these rows.

| algo | config | seed means (0, 1, 2) | mean ± std across seeds | median | max | crash % |
|---|---|---|---|---|---|---|
| dqn | buffer_size=10000 | 81 / 333 / 9 | 140.9 ± 138.9 | 32 | 2228 | 100 |
| dqn | buffer_size=500000 | 5 / 12 / 6 | 7.6 ± 2.8 | 3 | 57 | 100 |
| dqn | eps_decay_steps=20000 | 5 / 472 / 25 | 167.3 ± 215.5 | 16 | 2459 | 100 |
| dqn | eps_decay_steps=500000 | 4 / 58 / 15 | 25.5 ± 23.4 | 9 | 371 | 100 |
| dqn | lr=0.0001 | 8 / 11 / 49 | 22.7 ± 18.7 | 10 | 280 | 100 |
| dqn | lr=0.001 | 5 / 4 / 2 | 3.6 ± 1.2 | 2 | 34 | 100 |
| dqn | obs_norm=false | 5 / 5 / 4 | 4.7 ± 0.4 | 3 | 24 | 100 |
| dqn | sweep_baseline | 39 / 21 / 27 | 28.9 ± 7.7 | 18 | 269 | 100 |
| dqn | target_update=250 | 217 / 12 / 41 | 90.0 ± 90.5 | 20 | 1057 | 100 |
| dqn | target_update=5000 | 20 / 16 / 19 | 18.4 ± 2.0 | 10 | 233 | 100 |
| ppo | clip=0.1 | 3102 / 8086 / 9917 | 7035.2 ± 2879.7 | 10000 | 10000 | 44 |
| ppo | clip=0.3 | 9700 / 9867 / 1574 | 7047.0 ± 3870.4 | 10000 | 10000 | 35 |
| ppo | ent_coef=0.0 | 1 / 1 / 1 | 1.0 ± 0.0 | 1 | 1 | 100 |
| ppo | ent_coef=0.01 | 1 / 2016 / 1 | 672.6 ± 949.8 | 1 | 10000 | 100 |
| ppo | ent_coef=0.1 | 1547 / 8435 / 432 | 3471.4 ± 3539.1 | 1170 | 10000 | 77 |
| ppo | gae_lambda=0.9 | 8486 / 9900 / 10000 | 9462.0 ± 691.4 | 10000 | 10000 | 8 |
| ppo | gae_lambda=0.99 | 507 / 6879 / 7022 | 4802.7 ± 3037.8 | 3388 | 10000 | 66 |
| ppo | lr=0.00025 | 1 / 38 / 1 | 13.4 ± 17.5 | 1 | 218 | 100 |
| ppo | lr=0.003 | 272 / 8244 / 6152 | 4889.3 ± 3374.9 | 3808 | 10000 | 66 |
| ppo | obs_norm=false | 7 / 7 / 6 | 6.8 ± 0.4 | 4 | 52 | 100 |
| ppo | sweep_baseline | 9853 / 3062 / 9360 | 7425.0 ± 3091.9 | 10000 | 10000 | 36 |
| q_learning | alpha=0.05 | 4 / 6 / 2 | 4.1 ± 1.6 | 3 | 42 | 100 |
| q_learning | alpha=0.1 | 15 / 5 / 5 | 8.6 ± 4.8 | 5 | 51 | 100 |
| q_learning | bins=24x48x20 | 13 / 9 / 8 | 10.1 ± 2.4 | 7 | 63 | 100 |
| q_learning | bins=6x12x10 | 70 / 3 / 7 | 26.9 ± 30.9 | 6 | 352 | 100 |
| q_learning | eps_decay_steps=300000 | 17 / 12 / 17 | 15.4 ± 2.6 | 11 | 104 | 100 |
| q_learning | eps_schedule=exp | 39 / 62 / 27 | 42.5 ± 14.3 | 28 | 224 | 100 |
| q_learning | eps_start=1.0 | 1 / 2 / 2 | 2.0 ± 0.4 | 2 | 12 | 100 |
| q_learning | gamma=0.9 | 58 / 1199 / 40 | 432.3 ± 542.2 | 67 | 7601 | 100 |
| q_learning | gamma=0.95 | 30 / 295 / 45 | 123.4 ± 121.5 | 39 | 1783 | 100 |
| sb3_ppo | baseline | 9917 / 9100 / 9918 | 9645.1 ± 385.4 | 10000 | 10000 | 4 |

Reading it, with the variance in mind. The "± std" is the spread of three seed means, and the seed means themselves are wildly different, so one seed often decides the headline number:

- **Dominated by one seed.** Q-learning `gamma=0.9` (mean 432) is one seed at 1199 and two at 58 and 40, which is about the baseline's range, so it is not clearly better than the baseline (68.7). The same holds for DQN `eps_decay_steps=20000` (5 / 472 / 25), `buffer_size=10000` (81 / 333 / 9), `target_update=250` (217 / 12 / 41), Q-learning `bins=6x12x10` (70 / 3 / 7) and PPO `ent_coef=0.01` (1 / 2016 / 1). For these the between-seed std is 1.0 to 1.4 times the mean.
- **PPO is bimodal, not noisy.** PPO seeds often split into near-cap and far-below-cap runs: `sweep_baseline` is 9853 / 3062 / 9360, `clip=0.3` is 9700 / 9867 / 1574, `lr=0.003` is 272 / 8244 / 6152. These configs read 96.8 to 100.0 on the periodic eval because it caps at 100 pipes.
- **Consistent settings.** `gae_lambda=0.9` (8486 / 9900 / 10000, 92% of episodes at the cap) and SB3 PPO (9917 / 9100 / 9918, 96%) are the only sweep configs where all three seeds are high. Within a seed the episode std is also large (about 300 to 3,800 for the PPO configs that learn), but with 100 episodes the standard error of a seed mean is at most about 380, far below the thousands separating seeds. So the 100 episodes per seed are not the limiting factor; the number of seeds is.
- **SB3 vs ours.** At 1M steps SB3 averages 9645 and ours 7425, but two of our three seeds (9853, 9360) are close to SB3's. The gap comes from one seed (3062). With three seeds this is a hint, not a result.
- **Failures are stable.** `ent_coef=0.0`, `obs_norm=false`, `lr=0.001` (DQN) and the small Q-learning learning rates are low on every seed (best seed at most about 3x the worst; `alpha=0.1` is 15 / 5 / 5), so those effects are real even with three seeds.

Conclusion: rankings between settings that differ by less than the seed spread (most of this table) are not supported by three seeds. Only the all-low and all-high rows are.

## 3. Caveats

- Turning observation normalisation off drops PPO from about 100 to 8 and DQN from about 33 to 6 on the last periodic eval. On the 1M-step final eval it drops PPO from 7425 to 6.8 and DQN from 28.9 to 4.7, on every seed.
- Training cost per run: Q-learning ≈ 3 min, PPO ≈ 33 min, DQN ≈ 53–56 min. Wall times were measured on one Windows laptop with several runs in parallel, so they are rough.
- Final greedy eval is stricter than the periodic eval; the DQN baseline shows the gap (best ≈ 96, final 6.6). Compare using both.
- Three seeds only, so the standard deviations are rough and close settings (e.g. PPO clip 0.1 vs 0.3) cannot be ranked.
- Double DQN beats DQN in direction but not in a measurable size. Every Double DQN seed (24, 94, 395) is above every DQN seed (2.6, 14.4, 2.8), so the direction is consistent across seeds. The size is not: the mean of 170.9 is driven by one seed, and the pooled median is 56.5 against DQN's 2.
- The sweep tables use 1M-step runs, so they are not directly comparable with the 2M baselines. The same DQN configuration ends at 33.2 in the 1M `sweep_baseline` but 11.6 in the 2M baseline (last periodic eval), which shows how much a late DQN eval can move.
- These runs do not test *why* a setting helps or hurts (replay forgetting, reward scale, etc.); treat such explanations as hypotheses.

## 4. What the figures show

All in `figures/`. The full table is `figures/results.md`, which is stale for the sweep final-eval columns (see the appendix note).

- **`comparison.png`** (greedy eval every 25k steps, capped at 100 pipes). PPO rises fastest, to about 80 pipes by 0.5M steps, and ends near 100. DQN, Double DQN and Q-learning hover around 20 to 55 pipes with wide seed bands and no steady upward trend after about 1M steps. The DQN curves swing up and down rather than converge.
- **`final_boxplot.png`.** Pooled over seeds, PPO's box sits at the 1000 cap. Double DQN has the longest tail (33 of 300 episodes above 500, 12 at the cap, all from seed 2) but a median of only 56.5. DQN's median is near zero. The per-seed means (black dots) show how far apart the seeds are for Double DQN.
- **`crash_rates.png`.** During training, PPO's crashes per 10k steps fall to about 1, while DQN and Double DQN plateau near 40 and Q-learning falls slowly from about 150 to 10. In greedy eval, PPO's crash rate (episodes crashing before 100 pipes) goes to about 3%, and the other three stay at roughly 75 to 95%.
- **`exploration.png`.** Left: the ε schedules used. DQN reaches ε=0.01 at 100k steps and stays there; the Q-learning baseline decays slowly from 0.05 over all 3M steps; `eps_start=1.0` stays high for most of training, which matches that sweep failing. Right: PPO policy entropy falls quickly for `ent_coef=0.0` and `0.01`, and stays highest for `0.1`. The `0.0` run collapses to a near-deterministic policy early and never learns (section 2).
- **`dqn_q_values.png`.** Mean Q on replay batches for DQN and Double DQN both rise to about 12 and flatten, against a reference line "value of never dying ≈ 12.4". DQN ends slightly higher (about 12.6) than Double DQN (about 12.1). The gap is small, and this figure does not by itself explain the difference in final scores.
- **`q_heatmap.png`.** Policy and value of the Q-learning baseline (seed 0) over (dx, dy) for rising, apex and falling bird. Flap actions cluster when the bird is below the gap centre. Large parts of the grid are never visited, and the visited values drop to red (death expected) near the pipe edge with the bird far from the gap.
- **`learning_curves.png`** (per algorithm, greedy eval vs training episodes). For DQN and Double DQN the training episodes stay at about 5 pipes for the whole run while the greedy policy reaches 30 to 45, so exploration noise, not the learned policy, is what ends most training episodes. The two DQN greedy curves swing between about 15 and 45 and do not settle. PPO's training episodes climb to about 175 pipes in the last 0.5M steps, above the 100-pipe eval cap.
- **`obs_norm_ablation.png`** (1M steps). With raw observations PPO stalls at about 8 to 10 pipes where the standardised run reaches 100. DQN with raw observations peaks around 15 against about 32 standardised, and both end low and noisy.
- **`ppo_vs_sb3.png`.** Same hyperparameters, 1M steps. The from-scratch PPO rises earlier (about 70 pipes by 0.4M against about 45 for SB3), SB3 overtakes it around 0.5M to 0.65M, and both finish at 100. The curves differ in shape but not in outcome, which fits the "sanity check passes" reading rather than proving the two are identical.
- **`sweep_*.png`** (one per swept hyperparameter) show the sweep curves behind the tables in section 2.

## Appendix: full results table

Copied verbatim from `figures/results.md` (generated by `plot.py` from `runs/`). This copy is stale for the sweep rows: it shows dashes where the runs now have final evals (see the sweep final-eval table in section 2). Running `python plot.py` regenerates `results.md`; the four baseline rows would not change. The crash % column is rounded, so PPO baseline shows 0 where the exact rate is 0.3% (see section 1).

Final eval mean ± std: mean over all episodes, std across seed means. Steps to N pipes: median over the seeds that got there (first greedy eval with mean ≥ N).

| algo | config | seeds | final eval mean ± std | median | max | crash % | steps to 10 pipes | steps to 50 pipes | last eval | best eval | above-screen steps /1k (train last 20% / final eval) | wall time / run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dqn | baseline | 3 | 6.6 ± 5.5 | 2 | 62 | 100 | 500k (3/3) | 700k (3/3) | 11.6 ± 10.4 | 95.9 | 0.38 / 0.00 | 53.0 min |
| dqn | buffer_size=10000 | 3 | – | – | – | – | 250k (3/3) | 500k (3/3) | 50.0 ± 32.6 | 86.9 | 0.13 / – | 12.0 min |
| dqn | buffer_size=500000 | 3 | – | – | – | – | 350k (3/3) | 625k (1/3) | 8.3 ± 2.3 | 53.7 | 2.17 / – | 10.8 min |
| dqn | double | 3 | 170.9 ± 160.8 | 56 | 1000 | 96 | 375k (3/3) | 525k (3/3) | 58.7 ± 26.7 | 92.8 | 0.43 / 0.00 | 56.1 min |
| dqn | eps_decay_steps=20000 | 3 | – | – | – | – | 300k (3/3) | 300k (3/3) | 40.2 ± 33.2 | 91.2 | 2.01 / – | 27.9 min |
| dqn | eps_decay_steps=500000 | 3 | – | – | – | – | 838k (2/3) | never (0/3) | 17.7 ± 16.8 | 27.6 | 0.52 / – | 28.4 min |
| dqn | lr=0.0001 | 3 | – | – | – | – | 475k (3/3) | 775k (3/3) | 17.7 ± 10.3 | 75.5 | 0.35 / – | 22.6 min |
| dqn | lr=0.001 | 3 | – | – | – | – | 475k (3/3) | never (0/3) | 3.6 ± 2.5 | 46.4 | 0.83 / – | 29.1 min |
| dqn | obs_norm=false | 3 | – | – | – | – | 612k (2/3) | never (0/3) | 6.2 ± 1.8 | 26.8 | 0.43 / – | 16.7 min |
| dqn | sweep_baseline | 3 | – | – | – | – | 500k (3/3) | 700k (3/3) | 33.2 ± 10.6 | 75.7 | 0.75 / – | 24.1 min |
| dqn | target_update=250 | 3 | – | – | – | – | 425k (3/3) | 500k (3/3) | 50.4 ± 24.9 | 84.8 | 0.43 / – | 21.0 min |
| dqn | target_update=5000 | 3 | – | – | – | – | 400k (3/3) | 675k (3/3) | 17.5 ± 13.5 | 73.6 | 0.06 / – | 19.8 min |
| ppo | baseline | 3 | 997.6 ± 3.4 | 1000 | 1000 | 0 | 176k (3/3) | 326k (3/3) | 98.3 ± 2.4 | 100.0 | 0.00 / 0.00 | 32.7 min |
| ppo | clip=0.1 | 3 | – | – | – | – | 201k (3/3) | 350k (3/3) | 100.0 ± 0.0 | 100.0 | 0.00 / – | 15.2 min |
| ppo | clip=0.3 | 3 | – | – | – | – | 127k (3/3) | 225k (3/3) | 100.0 ± 0.0 | 100.0 | 0.00 / – | 18.8 min |
| ppo | ent_coef=0.0 | 3 | – | – | – | – | never (0/3) | never (0/3) | 1.0 ± 0.0 | 1.0 | 0.01 / – | 11.5 min |
| ppo | ent_coef=0.01 | 3 | – | – | – | – | 252k (1/3) | 301k (1/3) | 31.8 ± 43.5 | 34.0 | 0.01 / – | 9.6 min |
| ppo | ent_coef=0.1 | 3 | – | – | – | – | 152k (3/3) | 225k (3/3) | 98.3 ± 2.3 | 100.0 | 0.03 / – | 8.8 min |
| ppo | gae_lambda=0.9 | 3 | – | – | – | – | 225k (3/3) | 276k (3/3) | 100.0 ± 0.0 | 100.0 | 0.00 / – | 20.1 min |
| ppo | gae_lambda=0.99 | 3 | – | – | – | – | 201k (3/3) | 326k (3/3) | 88.5 ± 7.5 | 100.0 | 0.09 / – | 17.9 min |
| ppo | lr=0.00025 | 3 | – | – | – | – | 475k (1/3) | 801k (1/3) | 15.3 ± 20.2 | 20.8 | 0.03 / – | 12.5 min |
| ppo | lr=0.003 | 3 | – | – | – | – | 152k (3/3) | 350k (3/3) | 96.8 ± 4.4 | 99.4 | 0.00 / – | 17.2 min |
| ppo | obs_norm=false | 3 | – | – | – | – | 526k (3/3) | never (0/3) | 8.2 ± 1.5 | 17.7 | 0.04 / – | 5.2 min |
| ppo | sweep_baseline | 3 | – | – | – | – | 152k (3/3) | 301k (3/3) | 100.0 ± 0.0 | 100.0 | 0.00 / – | 16.2 min |
| q_learning | alpha=0.05 | 3 | – | – | – | – | never (0/3) | never (0/3) | 4.7 ± 2.4 | 5.1 | 0.01 / – | 2.2 min |
| q_learning | alpha=0.1 | 3 | – | – | – | – | 750k (2/3) | never (0/3) | 9.7 ± 4.6 | 16.6 | 0.00 / – | 2.2 min |
| q_learning | baseline | 3 | 68.7 ± 20.6 | 39 | 552 | 100 | 275k (3/3) | 1100k (3/3) | 44.0 ± 12.7 | 81.7 | 0.09 / 0.00 | 3.1 min |
| q_learning | bins=24x48x20 | 3 | – | – | – | – | 1800k (3/3) | never (0/3) | 10.3 ± 2.0 | 20.3 | 0.31 / – | 2.7 min |
| q_learning | bins=6x12x10 | 3 | – | – | – | – | 75k (3/3) | 525k (3/3) | 24.2 ± 29.7 | 91.1 | 0.32 / – | 3.4 min |
| q_learning | eps_decay_steps=300000 | 3 | – | – | – | – | 250k (3/3) | never (0/3) | 18.1 ± 9.1 | 22.8 | 0.01 / – | 3.3 min |
| q_learning | eps_schedule=exp | 3 | – | – | – | – | 300k (3/3) | 1050k (3/3) | 34.7 ± 14.0 | 64.0 | 0.00 / – | 2.8 min |
| q_learning | eps_start=1.0 | 3 | – | – | – | – | never (0/3) | never (0/3) | 1.8 ± 0.4 | 2.0 | 29.98 / – | 2.1 min |
| q_learning | gamma=0.9 | 3 | – | – | – | – | 600k (3/3) | 1375k (3/3) | 53.4 ± 27.7 | 98.2 | 0.09 / – | 3.5 min |
| q_learning | gamma=0.95 | 3 | – | – | – | – | 300k (3/3) | 1300k (3/3) | 60.8 ± 14.5 | 85.8 | 0.02 / – | 3.3 min |
| sb3_ppo | baseline | 3 | – | – | – | – | 225k (3/3) | 350k (3/3) | 100.0 ± 0.0 | 100.0 | 0.00 / – | 20.4 min |
