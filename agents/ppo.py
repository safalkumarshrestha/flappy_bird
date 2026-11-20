"""PPO from scratch: clipped surrogate, value loss, entropy bonus, GAE(lambda), advantage normalisation,
several epochs of minibatch updates per rollout, data from a Gymnasium SyncVectorEnv."""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.dqn import Standardize
from envs.flappy import make_env

OBS = 12
STATS = ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac")


def layer(n_in, n_out, std=np.sqrt(2)):
    lin = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.zeros_(lin.bias)
    return lin


class ActorCritic(nn.Module):
    def __init__(self, hidden, obs_norm):
        super().__init__()
        self.norm = Standardize(obs_norm)
        self.actor = nn.Sequential(layer(OBS, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(), layer(hidden, 2, 0.01))
        self.critic = nn.Sequential(layer(OBS, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(), layer(hidden, 1, 1.0))

    def dist(self, x):
        return Categorical(logits=self.actor(self.norm(x)))

    def value(self, x):
        return self.critic(self.norm(x)).squeeze(-1)


def train(cfg, log):
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    n_envs, n_steps, gamma, lam = cfg["n_envs"], cfg["n_steps"], cfg["gamma"], cfg["gae_lambda"]
    envs = gym.vector.SyncVectorEnv([lambda: make_env(cfg["max_steps"])] * n_envs,
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    net = ActorCritic(cfg["hidden"], cfg["obs_norm"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)

    @torch.no_grad()
    def greedy(o):
        return int(net.dist(torch.as_tensor(o, dtype=torch.float32)).logits.argmax())

    buf_obs = torch.zeros(n_steps, n_envs, OBS)
    buf_act, buf_logp, buf_rew, buf_done, buf_val = (torch.zeros(n_steps, n_envs) for _ in range(5))
    obs, _ = envs.reset(seed=[cfg["seed"] * 1000 + i for i in range(n_envs)])
    ep_ret, ep_len = np.zeros(n_envs), np.zeros(n_envs, int)
    n_updates, step, stats = cfg["total_steps"] // (n_envs * n_steps), 0, dict.fromkeys(STATS, "")

    for update in range(n_updates):
        if cfg["anneal_lr"]:
            opt.param_groups[0]["lr"] = cfg["lr"] * (1 - update / n_updates)
        for t in range(n_steps):
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                d, v = net.dist(o), net.value(o)
                a = d.sample()
            obs, r, term, trunc, info = envs.step(a.numpy())
            step += n_envs
            ep_ret, ep_len = ep_ret + r, ep_len + 1
            r_boot = r.copy()
            if trunc.any():  # time-limit cut: bootstrap from the true final observation
                fo = torch.as_tensor(np.stack(info["final_obs"][trunc]), dtype=torch.float32)
                with torch.no_grad():
                    r_boot[trunc] += gamma * net.value(fo).numpy()
            buf_obs[t], buf_act[t], buf_logp[t], buf_val[t] = o, a, d.log_prob(a), v
            buf_rew[t], buf_done[t] = torch.as_tensor(r_boot), torch.as_tensor(term | trunc, dtype=torch.float32)
            for i in np.flatnonzero(term | trunc):
                log.episode(step, ep_ret[i], info["final_info"]["score"][i], ep_len[i], **stats)
                ep_ret[i], ep_len[i] = 0, 0

        with torch.no_grad():  # GAE(lambda)
            next_v = net.value(torch.as_tensor(obs, dtype=torch.float32))
            adv, last = torch.zeros(n_steps, n_envs), 0
            for t in reversed(range(n_steps)):
                nonterm = 1 - buf_done[t]
                delta = buf_rew[t] + gamma * next_v * nonterm - buf_val[t]
                adv[t] = last = delta + gamma * lam * nonterm * last
                next_v = buf_val[t]
            ret = adv + buf_val

        b_obs, b_act, b_logp, b_adv, b_ret = (x.reshape(n_steps * n_envs, *x.shape[2:])
                                              for x in (buf_obs, buf_act, buf_logp, adv, ret))
        batch, mb = n_steps * n_envs, n_steps * n_envs // cfg["n_minibatches"]
        acc = {k: [] for k in STATS}
        for _ in range(cfg["n_epochs"]):
            perm = torch.randperm(batch)
            for s in range(0, batch, mb):
                i = perm[s:s + mb]
                d = net.dist(b_obs[i])
                log_ratio = d.log_prob(b_act[i]) - b_logp[i]
                ratio = log_ratio.exp()
                A = (b_adv[i] - b_adv[i].mean()) / (b_adv[i].std() + 1e-8)
                pg_loss = -torch.min(ratio * A, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * A).mean()
                v_loss = 0.5 * (net.value(b_obs[i]) - b_ret[i]).pow(2).mean()
                entropy = d.entropy().mean()
                loss = pg_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                opt.step()
                with torch.no_grad():
                    for k, val in (("pg_loss", pg_loss), ("v_loss", v_loss), ("entropy", entropy),
                                   ("approx_kl", ((ratio - 1) - log_ratio).mean()),
                                   ("clipfrac", ((ratio - 1).abs() > cfg["clip"]).float().mean())):
                        acc[k].append(val.item())
        stats = {k: round(float(np.mean(v)), 5) for k, v in acc.items()}
        log.tick(step, greedy)

    torch.save(net.state_dict(), log.dir / "model.pt")


def load_policy(run_dir, cfg):
    net = ActorCritic(cfg["hidden"], cfg["obs_norm"])
    net.load_state_dict(torch.load(run_dir / "model.pt"))

    @torch.no_grad()
    def greedy(o):
        return int(net.dist(torch.as_tensor(o, dtype=torch.float32)).logits.argmax())
    return greedy
