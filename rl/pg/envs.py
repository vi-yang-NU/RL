"""Gym environments for Policy Gradient Parts 1 and 2 (the versions the handout requires), and the adapters
that turn their observations into policy inputs and policy outputs into environment actions.

* CartPole-v1 (and any environment with a 1-D Box observation space, such as the toy environments in
  rl/pg/toy.py): the observation vector is the policy input; action indices are the environment's actions.
* ALE/Pong-v5: frames go through the handout's `preprocess` (80x80 binary image); by default the policy sees the
  difference between the current and the previous preprocessed frame. The policy chooses between the two
  actions RIGHT (2) and LEFT (3) only.
"""
from __future__ import annotations

from typing import Any

import numpy as np

ENV_IDS = {"cartpole": "CartPole-v1", "pong": "ALE/Pong-v5"}
GAMMA = {"cartpole": 0.95, "pong": 0.99}  # discount factors given in the handout
PONG_ACTIONS = (2, 3)  # RIGHT, LEFT: the only Pong actions the policy may choose between
# Game time per agent step: CartPole-v1 integrates its physics with tau = 0.02 s; Pong-v5 repeats each action for
# 4 of the Atari's 60 frames per second. (How long an episode "lasts" = steps × this.)
SECONDS_PER_STEP = {"cartpole": 0.02, "pong": 4 / 60}
PONG_FRAME = (80, 80)


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


class Adapter:
    """Vector observations (CartPole-v1, toy environments): identity encoding, identity action mapping."""

    name = "vector"

    def __init__(self, env: Any, options: Any = None):
        shape = tuple(getattr(env.observation_space, "shape", ()) or ())
        if len(shape) != 1:
            raise ValueError(f"the vector adapter needs a 1-D observation space, got shape {shape}")
        self.obs_shape = (int(shape[0]),)
        self.n_actions = int(env.action_space.n)

    def reset(self, env: Any, seed: int | None = None) -> np.ndarray:
        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
        return self.encode(obs)

    def step(self, env: Any, action_index: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = env.step(self.action(action_index))
        return self.encode(obs), float(reward), bool(terminated or truncated), {"terminated": terminated, "truncated": truncated, **info}

    def encode(self, obs: Any) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(self.obs_shape)

    def action(self, action_index: int) -> Any:
        if not 0 <= action_index < self.n_actions:
            raise ValueError(f"action index {action_index} out of range for {self.n_actions} actions")
        return int(action_index)


class PongAdapter(Adapter):
    """Pong frames -> preprocessed 80x80 (difference) image; policy action {0, 1} -> {RIGHT, LEFT}."""

    name = "pong"

    def __init__(self, env: Any, options: Any = None):
        self.cnn = getattr(options, "arch", "mlp") == "cnn"
        self.frame_diff = bool(getattr(options, "frame_diff", True))
        self.kernels = bool(getattr(options, "kernels", False))  # rl/cuda/pg.py preprocess kernel
        self.obs_shape = (1, *PONG_FRAME) if self.cnn else (PONG_FRAME[0] * PONG_FRAME[1],)
        self.n_actions = len(PONG_ACTIONS)
        self.previous: np.ndarray | None = None

    def reset(self, env: Any, seed: int | None = None) -> np.ndarray:
        self.previous = None
        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
        return self.encode(obs)

    def encode(self, obs: Any) -> np.ndarray:
        if self.kernels:
            from rl.cuda.pg import preprocess_frames_gpu

            binarized, diff = preprocess_frames_gpu(np.asarray(obs), None if self.previous is None else self.previous[None])
            frame, x = binarized[0], diff[0]
            if self.frame_diff:
                self.previous = frame
                return x.reshape(self.obs_shape).astype(np.float32)
            return frame.reshape(self.obs_shape).astype(np.float32)
        frame = preprocess(np.array(obs, copy=True))  # copy: preprocess modifies its input in place
        if self.frame_diff:
            x = frame - self.previous if self.previous is not None else np.zeros_like(frame)
            self.previous = frame
        else:
            x = frame
        return x.reshape(self.obs_shape).astype(np.float32)

    def action(self, action_index: int) -> int:
        if not 0 <= action_index < self.n_actions:
            raise ValueError(f"action index {action_index} out of range for {self.n_actions} actions")
        return PONG_ACTIONS[action_index]


def make_adapter(cfg: Any, env: Any, options: Any = None) -> Adapter:
    """The adapter for cfg.env: PongAdapter for "pong", the vector adapter for everything else.

    `options` defaults to the Options derived from cfg (cfg.extra), so adapters built from the same cfg agree.
    """
    if options is None:
        from rl.pg.options import options_for

        options = options_for(cfg)
    if getattr(cfg, "env", "") == "pong":
        return PongAdapter(env, options)
    return Adapter(env, options)
