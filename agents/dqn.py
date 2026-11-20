"""DQN / Double DQN: replay buffer, target network (hard or Polyak), Huber loss, Adam, grad-norm clipping."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.q_learning import epsilon
from envs.flappy import OBS_MEAN, OBS_STD, make_env

OBS = 12


class Standardize(nn.Module):
    """(obs - mean) / std with the frozen statistics from envs.flappy; identity when off."""

    def __init__(self, on):
        super().__init__()
        self.on, self.mean, self.std = on, torch.as_tensor(OBS_MEAN), torch.as_tensor(OBS_STD)

    def forward(self, x):
        return (x - self.mean) / self.std if self.on else x


def mlp(hidden, obs_norm):
    return nn.Sequential(Standardize(obs_norm), nn.Linear(OBS, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))


def train(cfg, log):
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    q, q_targ = mlp(cfg["hidden"], cfg["obs_norm"]), mlp(cfg["hidden"], cfg["obs_norm"])
    q_targ.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg["lr"])

    n = cfg["buffer_size"]
    b_obs, b_next = np.zeros((n, OBS), np.float32), np.zeros((n, OBS), np.float32)
    b_act, b_rew, b_term = np.zeros(n, np.int64), np.zeros(n, np.float32), np.zeros(n, np.float32)
    ptr = size = 0

    @torch.no_grad()
    def greedy(o):
        return int(q(torch.as_tensor(o, dtype=torch.float32)).argmax())

    env = make_env(cfg["max_steps"])
    obs, _ = env.reset(seed=cfg["seed"])
    ret, length, losses, qs = 0.0, 0, [], []
    for step in range(1, cfg["total_steps"] + 1):
        eps = epsilon(cfg, step)
        a = int(rng.integers(2)) if rng.random() < eps else greedy(obs)
        nxt, r, term, trunc, info = env.step(a)
        b_obs[ptr], b_act[ptr], b_rew[ptr], b_next[ptr], b_term[ptr] = obs, a, r, nxt, term  # truncation bootstraps
        ptr, size = (ptr + 1) % n, min(size + 1, n)
        obs, ret, length = nxt, ret + r, length + 1

        if step >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
            i = rng.integers(size, size=cfg["batch_size"])
            o, a_, r_, o2, d = (torch.as_tensor(x[i]) for x in (b_obs, b_act, b_rew, b_next, b_term))
            with torch.no_grad():
                if cfg["double"]:  # online net picks a', target net evaluates it
                    q2 = q_targ(o2).gather(1, q(o2).argmax(1, keepdim=True)).squeeze(1)
                else:
                    q2 = q_targ(o2).max(1).values
                target = r_ + cfg["gamma"] * (1 - d) * q2
            pred = q(o).gather(1, a_.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), cfg["max_grad_norm"])
            opt.step()
            losses.append(loss.item())
            qs.append(pred.mean().item())
        if step % cfg["target_update"] == 0:  # tau=1 -> hard copy, tau<1 -> Polyak
            with torch.no_grad():
                for p, pt in zip(q.parameters(), q_targ.parameters()):
                    pt.lerp_(p, cfg["tau"])

        if term or trunc:
            log.episode(step, ret, info["score"], length, epsilon=round(eps, 4),
                        loss=round(np.mean(losses), 5) if losses else "", q_mean=round(np.mean(qs), 3) if qs else "")
            obs, _ = env.reset()
            ret, length, losses, qs = 0.0, 0, [], []
        log.tick(step, greedy)

    torch.save(q.state_dict(), log.dir / "model.pt")


def load_policy(run_dir, cfg):
    q = mlp(cfg["hidden"], cfg["obs_norm"])
    q.load_state_dict(torch.load(run_dir / "model.pt"))

    @torch.no_grad()
    def greedy(o):
        return int(q(torch.as_tensor(o, dtype=torch.float32)).argmax())
    return greedy
