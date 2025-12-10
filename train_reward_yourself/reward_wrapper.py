"""
Reward wrapper that blends environment rewards with learned value-based rewards.

Uses the pretrained critic to compute learned rewards from the Bellman equation:
    r_learned = V(s) - γ*V(s')
and blends with environment reward using a configurable ratio:
    r_blended = (1-α)*r_env + α*r_learned
"""

import numpy as np
import torch
import gymnasium as gym
from typing import Optional
from stable_baselines3.common.vec_env import VecEnvWrapper


class LearnedRewardWrapper(gym.Wrapper):
    """
    Wrapper that computes rewards as R = (1-α) * r_env + α * r_learned.

    Where r_learned = V(s) - γ*V(s') from the Bellman equation.
    This allows blending environment rewards with learned value-based rewards
    from a pretrained critic network.
    """

    def __init__(
        self,
        env: gym.Env,
        critic_model,
        learned_reward_ratio: float = 0.5,
        device: str = "cpu",
        gamma: float = 0.99,
    ):
        """
        Args:
            env: Base environment
            critic_model: PPO/SAC model with trained value network (critic)
            learned_reward_ratio: Ratio in [0, 1].
                                  0 = use only env reward
                                  1 = use only learned reward
            device: Device for critic inference
            gamma: Discount factor for computing learned reward
        """
        super().__init__(env)

        assert 0.0 <= learned_reward_ratio <= 1.0, "Ratio must be in [0, 1]"

        self.critic = critic_model
        self.alpha = learned_reward_ratio
        self.device = device
        self.gamma = gamma
        self.last_obs = None
        self.last_value = None

        # Stats tracking
        self.env_reward_sum = 0.0
        self.learned_reward_sum = 0.0
        self.step_count = 0

    def reset(self, **kwargs):
        """Reset environment and store initial observation."""
        obs, info = self.env.reset(**kwargs)

        # Compute and cache initial value
        with torch.no_grad():
            obs_tensor = self._to_tensor(obs)
            self.last_value = self._get_value(obs_tensor)
            self.last_obs = obs

        return obs, info

    def step(self, action):
        """Step environment and blend rewards."""
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Compute value of new state
        with torch.no_grad():
            obs_tensor = self._to_tensor(obs)
            curr_value = self._get_value(obs_tensor)

            # Learned reward from Bellman equation: r = V(s) - γ*V(s')
            learned_reward = (self.last_value - self.gamma * curr_value).item()

            # Blend rewards
            blended_reward = (1 - self.alpha) * env_reward + self.alpha * learned_reward

            # Update tracking
            self.env_reward_sum += env_reward
            self.learned_reward_sum += learned_reward
            self.step_count += 1

            # Store for next step
            self.last_value = curr_value
            self.last_obs = obs

            # Add debug info
            info['env_reward'] = env_reward
            info['learned_reward'] = learned_reward
            info['blended_reward'] = blended_reward

        return obs, blended_reward, terminated, truncated, info

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert observation to tensor."""
        tensor = torch.from_numpy(obs).float()
        # Add batch dimension
        if len(tensor.shape) == len(self.observation_space.shape):
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _get_value(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Get value estimate from critic."""
        # For PPO/SAC, extract features then pass to value_net
        features = self.critic.policy.extract_features(obs_tensor)
        value = self.critic.policy.value_net(features)
        return value.squeeze()

    def get_stats(self):
        """Get reward statistics."""
        if self.step_count == 0:
            return {
                'avg_env_reward': 0.0,
                'avg_learned_reward': 0.0,
                'steps': 0
            }

        return {
            'avg_env_reward': self.env_reward_sum / self.step_count,
            'avg_learned_reward': self.learned_reward_sum / self.step_count,
            'steps': self.step_count
        }

    def reset_stats(self):
        """Reset tracking stats."""
        self.env_reward_sum = 0.0
        self.learned_reward_sum = 0.0
        self.step_count = 0


class VecLearnedRewardWrapper(VecEnvWrapper):
    """
    Vectorized version of LearnedRewardWrapper for VecEnv.

    Wraps around VecEnv to blend rewards efficiently across parallel environments.
    """

    def __init__(
        self,
        venv,
        critic_model,
        learned_reward_ratio: float = 0.5,
        device: str = "cpu",
        return_mean: Optional[float] = None,
        return_std: Optional[float] = None,
        gamma: float = 0.99,
    ):
        """
        Args:
            venv: Vectorized environment
            critic_model: PPO/SAC model with trained value network
            learned_reward_ratio: Blend ratio [0, 1]
            device: Device for critic inference
            return_mean: Mean used to normalize returns during critic training
            return_std: Std used to normalize returns during critic training
            gamma: Discount factor for computing learned reward
        """
        super().__init__(venv)
        self.critic = critic_model
        self.alpha = learned_reward_ratio
        self.device = device
        self.gamma = gamma
        self.return_mean = return_mean
        self.return_std = return_std

        # Track last values for each env
        self.last_values = None

        # Stats
        self.env_reward_sum = np.zeros(self.num_envs)
        self.learned_reward_sum = np.zeros(self.num_envs)
        self.step_counts = np.zeros(self.num_envs)

        # Initialize by resetting
        self._first_reset = True

    def reset(self):
        """Reset all environments."""
        obs = self.venv.reset()

        # Compute initial values
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            self.last_values = self._get_values(obs_tensor)

        self._first_reset = False
        return obs

    def step_wait(self):
        """Wait for step to complete and blend rewards."""
        # Initialize on first step if reset wasn't called
        if self.last_values is None:
            self.reset()

        # Get results from wrapped environment
        obs, env_rewards, dones, infos = self.venv.step_wait()

        # Compute values for new states
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            curr_values = self._get_values(obs_tensor)

            # Learned rewards from Bellman equation: r = V(s) - γ*V(s')
            learned_rewards = (self.last_values - curr_values).cpu().numpy()

            # Blend rewards
            blended_rewards = (1 - self.alpha) * env_rewards + self.alpha * learned_rewards
            # print(f"Env Rewards: {env_rewards}")
            # print(f"Learned Rewards: {learned_rewards}")

            # Update stats
            self.env_reward_sum += env_rewards
            self.learned_reward_sum += learned_rewards
            self.step_counts += 1

            # Update last values for all environments
            self.last_values = curr_values

            # Add debug info to each env's info dict
            for i, info in enumerate(infos):
                info['env_reward'] = env_rewards[i]
                info['learned_reward'] = learned_rewards[i]
                info['blended_reward'] = blended_rewards[i]

        return obs, blended_rewards, dones, infos

    def _get_values(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Get value estimates from critic for batch of observations."""
        features = self.critic.policy.extract_features(obs_tensor)
        values = self.critic.policy.value_net(features)
        values = values.squeeze()

        # Denormalize if normalization stats were provided
        if self.return_mean is not None and self.return_std is not None:
            values = values * self.return_std + self.return_mean

        return values

    def get_stats(self):
        """Get average reward statistics across all envs."""
        total_steps = self.step_counts.sum()
        if total_steps == 0:
            return {
                'avg_env_reward': 0.0,
                'avg_learned_reward': 0.0,
                'total_steps': 0
            }

        return {
            'avg_env_reward': self.env_reward_sum.sum() / total_steps,
            'avg_learned_reward': self.learned_reward_sum.sum() / total_steps,
            'total_steps': int(total_steps)
        }

    def reset_stats(self):
        """Reset tracking stats."""
        self.env_reward_sum = np.zeros(self.num_envs)
        self.learned_reward_sum = np.zeros(self.num_envs)
        self.step_counts = np.zeros(self.num_envs)
