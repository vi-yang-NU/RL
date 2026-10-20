"""Gym environments for Policy Gradient Parts 1 and 2 (the versions the handout requires)."""
from __future__ import annotations

from typing import Any

import numpy as np

ENV_IDS = {"cartpole": "CartPole-v1", "pong": "ALE/Pong-v5"}
GAMMA = {"cartpole": 0.95, "pong": 0.99}  # discount factors given in the handout
PONG_ACTIONS = (2, 3)  # RIGHT, LEFT: the only Pong actions the policy may choose between


def make_env(name: str, **kwargs: Any):
    """Create CartPole-v1 or ALE/Pong-v5. Extra kwargs go to gymnasium.make."""
    import gymnasium as gym

    if name == "pong":
        import ale_py

        if hasattr(gym, "register_envs"):  # gymnasium >= 1.0 needs ALE envs registered explicitly
            gym.register_envs(ale_py)
    return gym.make(ENV_IDS[name], **kwargs)


def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float32).ravel(), [80, 80])
