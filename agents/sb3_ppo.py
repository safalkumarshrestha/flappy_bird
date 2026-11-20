"""Sanity check only: Stable-Baselines3 PPO with the same hyperparameters, network and input standardisation as
agents/ppo.py, logged with the same protocol.  pip install stable-baselines3==2.9.0"""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from envs.flappy import OBS_MEAN, OBS_STD, make_env


def standardise(o):
    return ((o - OBS_MEAN) / OBS_STD).astype(np.float32)


def train(cfg, log):
    torch.set_num_threads(1)
    make = lambda: gym.wrappers.TransformObservation(make_env(cfg["max_steps"]), standardise,
                                                     gym.spaces.Box(-np.inf, np.inf, (12,), np.float32))
    env = make_vec_env(make, n_envs=cfg["n_envs"], seed=cfg["seed"] * 1000)
    h = cfg["hidden"]
    model = PPO("MlpPolicy", env, learning_rate=lambda frac: cfg["lr"] * frac, n_steps=cfg["n_steps"],
                batch_size=cfg["n_steps"] * cfg["n_envs"] // cfg["n_minibatches"], n_epochs=cfg["n_epochs"],
                gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip"], ent_coef=cfg["ent_coef"],
                vf_coef=cfg["vf_coef"], max_grad_norm=cfg["max_grad_norm"], seed=cfg["seed"], device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[h, h], vf=[h, h]), activation_fn=torch.nn.Tanh))
    policy = lambda o: int(model.predict(standardise(o), deterministic=True)[0])

    class Log(BaseCallback):
        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    log.episode(self.num_timesteps, info["episode"]["r"], info["score"], info["episode"]["l"])
            log.tick(self.num_timesteps, policy)
            return True

    model.learn(cfg["total_steps"], callback=Log())
    model.save(log.dir / "model.zip")


def load_policy(run_dir, cfg):
    model = PPO.load(run_dir / "model.zip", device="cpu")
    return lambda o: int(model.predict(standardise(o), deterministic=True)[0])
