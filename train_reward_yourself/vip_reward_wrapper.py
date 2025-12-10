"""
VIP Reward Wrapper for using goal-conditioned value function as rewards.

Uses VIP critic V(s, g) to provide dense rewards for RL training.
"""

import numpy as np
import torch
import h5py
from typing import Optional, List
from pathlib import Path
from stable_baselines3.common.vec_env import VecEnvWrapper


def extract_expert_goals(
    hdf5_path: str,
    n_demos: int = 100,
    obs_type: str = "state",
    sample_mode: str = "final",
    samples_per_demo: int = 1,
    use_single_goal: bool = False,
) -> torch.Tensor:
    """Extract goal states from expert demonstrations.

    Args:
        hdf5_path: Path to HDF5 demonstrations
        n_demos: Number of demonstrations to use
        obs_type: "state" or "image"
        sample_mode: "final" (last state), "random" (random states), "all" (all states), "canonical" (single best goal)
        samples_per_demo: Number of samples per demo (for "random" mode)
        use_single_goal: If True, return only one canonical goal (best final state with highest reward)

    Returns:
        Tensor of goal states (N, obs_dim) or (1, obs_dim) if use_single_goal=True
    """
    goals = []
    episode_returns = []  # Track returns to find best episode

    with h5py.File(hdf5_path, "r") as f:
        # First pass: collect goals and returns if using single goal mode
        if use_single_goal:
            print("Collecting episode returns to find best goal...")
            for i in range(n_demos):
                ep_key = f"demo_{i}"
                if ep_key not in f["data"]:
                    continue
                ep_grp = f[f"data/{ep_key}"]

                # Get episode return
                rewards = np.array(ep_grp["rewards"])
                episode_return = rewards.sum()
                episode_returns.append((i, episode_return))

            # Find best episode
            best_demo_idx = max(episode_returns, key=lambda x: x[1])[0]
            best_return = max(episode_returns, key=lambda x: x[1])[1]
            print(f"Best demo: {best_demo_idx} with return {best_return:.2f}")

            # Extract only the final state from best episode
            ep_key = f"demo_{best_demo_idx}"
            ep_grp = f[f"data/{ep_key}"]
            if obs_type == "state":
                obs = np.array(ep_grp["obs/state"])
            elif obs_type == "image":
                obs = np.array(ep_grp["obs/images"])
            else:
                raise ValueError(f"Unknown obs_type: {obs_type}")

            goal = obs[-1]  # Final state
            goals_array = np.array([goal])
            print(f"Using single canonical goal from best episode")
            print(f"Goal shape: {goals_array.shape}")
            return torch.from_numpy(goals_array).float()

        # Original multi-goal extraction
        for i in range(n_demos):
            ep_key = f"demo_{i}"
            if ep_key not in f["data"]:
                continue

            ep_grp = f[f"data/{ep_key}"]

            if obs_type == "state":
                obs = np.array(ep_grp["obs/state"])
            elif obs_type == "image":
                obs = np.array(ep_grp["obs/images"])
            else:
                raise ValueError(f"Unknown obs_type: {obs_type}")

            if sample_mode == "final":
                # Use last state as goal
                goals.append(obs[-1])
            elif sample_mode == "random":
                # Sample random states from episode
                indices = np.random.choice(len(obs), size=samples_per_demo, replace=False)
                goals.extend([obs[idx] for idx in indices])
            elif sample_mode == "all":
                # Use all states as potential goals
                goals.extend(obs)
            else:
                raise ValueError(f"Unknown sample_mode: {sample_mode}")

    goals_array = np.array(goals)
    print(f"Extracted {len(goals_array)} expert goals from {n_demos} demos")
    print(f"Goal shape: {goals_array.shape}")

    return torch.from_numpy(goals_array).float()


class VIPRewardWrapper(VecEnvWrapper):
    """
    Wrapper that uses VIP critic V(s, g) to provide rewards.

    Can use different reward modes:
    - "value": r = V(s, g) (value as reward)
    - "td": r = V(s', g) - V(s, g) (temporal difference)
    - "blend": r = (1-α) * r_env + α * V(s, g) (blend with env reward)
    """

    def __init__(
        self,
        venv,
        vip_critic,
        expert_goals: Optional[torch.Tensor] = None,
        device: str = "cpu",
        reward_mode: str = "value",
        blend_ratio: float = 0.5,
        goal_sample_mode: str = "episode",
        normalize_rewards: bool = True,
    ):
        """
        Args:
            venv: Vectorized environment
            vip_critic: Trained VIP critic (goal-conditioned value function)
            expert_goals: Tensor of expert goal states (N, obs_dim).
                         None if critic uses learned goal embedding.
            device: Device for critic inference
            reward_mode: "value", "td", or "blend"
            blend_ratio: Ratio for blend mode [0, 1]
            goal_sample_mode: "episode" (sample per episode) or "step" (sample per step)
            normalize_rewards: Whether to normalize VIP rewards
        """
        super().__init__(venv)
        self.critic = vip_critic
        self.use_learned_goal = hasattr(vip_critic, 'use_learned_goal') and vip_critic.use_learned_goal

        if self.use_learned_goal:
            # Critic has learned goal embedding - no need for expert goals
            self.expert_goals = None
            self.use_single_goal = False
            print("VIPRewardWrapper: Using critic's learned goal embedding")
        else:
            # Need expert goals for goal-conditioned rewards
            if expert_goals is None:
                raise ValueError("expert_goals required when critic doesn't use learned goal embedding")
            self.expert_goals = expert_goals.to(device)
            self.use_single_goal = len(expert_goals) == 1

        self.device = device
        self.reward_mode = reward_mode
        self.blend_ratio = blend_ratio
        self.goal_sample_mode = goal_sample_mode
        self.normalize_rewards = normalize_rewards

        # Track current goals for each environment (not used with learned goals)
        self.current_goals = None
        self.last_values = None

        # Running statistics for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # Stats tracking
        self.env_reward_sum = np.zeros(self.num_envs)
        self.vip_reward_sum = np.zeros(self.num_envs)
        self.step_counts = np.zeros(self.num_envs)

        print(f"VIPRewardWrapper initialized:")
        print(f"  Reward mode: {reward_mode}")
        if self.use_learned_goal:
            print(f"  Using learned goal embedding (no expert goals needed)")
        else:
            if self.use_single_goal:
                print(f"  Using single canonical goal (no sampling)")
            else:
                print(f"  Expert goals: {len(expert_goals)}")
                print(f"  Goal sampling: {goal_sample_mode}")
        print(f"  Normalize: {normalize_rewards}")

    def _sample_goals(self, n: int) -> torch.Tensor:
        """Sample n random goals from expert goals (or repeat single goal)."""
        if self.use_single_goal:
            # Repeat single goal n times
            return self.expert_goals.expand(n, -1).clone()
        else:
            # Sample randomly from multiple goals
            indices = torch.randint(0, len(self.expert_goals), (n,))
            return self.expert_goals[indices]

    def reset(self):
        """Reset all environments and sample new goals."""
        obs = self.venv.reset()

        if not self.use_learned_goal:
            # Sample goals for each environment (only if not using learned goal)
            self.current_goals = self._sample_goals(self.num_envs)

        # Compute initial values for TD mode
        if self.reward_mode == "td":
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
                if self.use_learned_goal:
                    self.last_values = self.critic(obs_tensor)
                else:
                    self.last_values = self.critic(obs_tensor, self.current_goals)

        return obs

    def step_wait(self):
        """Wait for step to complete and compute VIP rewards."""
        # Initialize on first step if reset wasn't called
        if not self.use_learned_goal and self.current_goals is None:
            self.reset()

        # Get results from wrapped environment
        obs, env_rewards, dones, infos = self.venv.step_wait()

        # Resample goals if using per-step sampling (only for explicit goals)
        if not self.use_learned_goal and self.goal_sample_mode == "step":
            self.current_goals = self._sample_goals(self.num_envs)

        # Compute VIP rewards
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(self.device)

            if self.reward_mode == "value":
                # Reward = V(s, g)
                if self.use_learned_goal:
                    vip_values = self.critic(obs_tensor)
                else:
                    vip_values = self.critic(obs_tensor, self.current_goals)
                vip_rewards = vip_values.squeeze(-1).cpu().numpy()

            elif self.reward_mode == "td":
                # Reward = V(s', g) - V(s, g)
                if self.use_learned_goal:
                    curr_values = self.critic(obs_tensor)
                else:
                    curr_values = self.critic(obs_tensor, self.current_goals)
                vip_rewards = (curr_values - self.last_values).squeeze(-1).cpu().numpy()
                self.last_values = curr_values
                

            # Normalize VIP rewards
            if self.normalize_rewards:
                # Update running statistics
                self.reward_count += len(vip_rewards)
                delta = vip_rewards - self.reward_mean
                self.reward_mean += delta.sum() / self.reward_count
                self.reward_std = np.sqrt(
                    ((self.reward_std ** 2) * (self.reward_count - len(vip_rewards)) +
                     (delta ** 2).sum()) / self.reward_count
                ) + 1e-8

                # Normalize
                vip_rewards = (vip_rewards - self.reward_mean) / self.reward_std

            # Blend with environment rewards if requested
            final_rewards = (1 - self.blend_ratio) * env_rewards + self.blend_ratio * vip_rewards
            # print(f"Env Rewards: {env_rewards}")
            # print(f"VIP Rewards: {vip_rewards}")
            # print(f"Final Rewards: {final_rewards}")

            # Update stats
            self.env_reward_sum += env_rewards
            self.vip_reward_sum += vip_rewards
            self.step_counts += 1

            # Add debug info
            for i, info in enumerate(infos):
                info['env_reward'] = env_rewards[i]
                info['vip_reward'] = vip_rewards[i]
                info['final_reward'] = final_rewards[i]

            # Resample goals for done episodes (only for explicit goals)
            if not self.use_learned_goal:
                for i, done in enumerate(dones):
                    if done and self.goal_sample_mode == "episode":
                        self.current_goals[i] = self._sample_goals(1).squeeze(0)

        return obs, final_rewards, dones, infos

    def get_stats(self):
        """Get average reward statistics."""
        total_steps = self.step_counts.sum()
        if total_steps == 0:
            return {
                'avg_env_reward': 0.0,
                'avg_vip_reward': 0.0,
                'total_steps': 0
            }

        return {
            'avg_env_reward': self.env_reward_sum.sum() / total_steps,
            'avg_vip_reward': self.vip_reward_sum.sum() / total_steps,
            'total_steps': int(total_steps)
        }

    def reset_stats(self):
        """Reset tracking stats."""
        self.env_reward_sum = np.zeros(self.num_envs)
        self.vip_reward_sum = np.zeros(self.num_envs)
        self.step_counts = np.zeros(self.num_envs)


class VIPRewardStatsCallback:
    """Callback to log VIP reward statistics."""

    def __init__(self, vip_wrapper: VIPRewardWrapper, log_freq: int = 1000, verbose: int = 0):
        self.vip_wrapper = vip_wrapper
        self.log_freq = log_freq
        self.verbose = verbose
        self.n_calls = 0

    def __call__(self, locals_dict, globals_dict):
        """Called at each step."""
        self.n_calls += 1

        if self.n_calls % self.log_freq == 0:
            stats = self.vip_wrapper.get_stats()
            if stats['total_steps'] > 0:
                if self.verbose:
                    print(f"VIP Stats @ step {self.n_calls}:")
                    print(f"  Env reward: {stats['avg_env_reward']:.3f}")
                    print(f"  VIP reward: {stats['avg_vip_reward']:.3f}")

                # Log to tensorboard if available
                if 'self' in locals_dict:
                    model = locals_dict['self']
                    if hasattr(model, 'logger'):
                        model.logger.record("vip/env_reward", stats['avg_env_reward'])
                        model.logger.record("vip/vip_reward", stats['avg_vip_reward'])

                # Reset stats after logging
                self.vip_wrapper.reset_stats()

        return True
