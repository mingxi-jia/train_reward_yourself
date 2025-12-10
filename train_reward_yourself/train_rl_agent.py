"""
RL Agent Training Script

Supports training with PPO or SAC on:
- Classic control environments (LunarLander, etc.)
- Robosuite robotic manipulation environments (future support)

Usage:
    # Train with PPO on LunarLander
    python train_rl_agent.py --env-type gym --env-name LunarLander-v3 --algo ppo --timesteps 500000

    # Train with SAC on LunarLander
    python train_rl_agent.py --env-type gym --env-name LunarLander-v3 --algo sac --timesteps 500000

    # Future: Train on robosuite environments
    python train_rl_agent.py --env-type robosuite --env-name Lift --algo ppo --control-type OSC_POSE
"""

import os
import warnings
import argparse
import random
import numpy as np
import torch
import gymnasium as gym
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback

# Import shared utilities
from train_reward_yourself.env_utils import EnvConfig
from train_reward_yourself.reward_wrapper import VecLearnedRewardWrapper
from train_reward_yourself.vip_reward_wrapper import VIPRewardWrapper, extract_expert_goals
from train_reward_yourself.train_vip_critic import VIPCritic

# Import panda_gym to register Panda environments (if available)
try:
    import panda_gym
except ImportError:
    pass  # panda_gym not installed, skip

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def set_random_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_normalization_stats(config_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Parse return normalization stats from BC config file.

    Returns:
        (mean, std) tuple, or (None, None) if not found
    """
    if not config_path.exists():
        return None, None

    mean, std = None, None
    with open(config_path, 'r') as f:
        in_norm_section = False
        for line in f:
            line = line.strip()
            if line == "Return Normalization:":
                in_norm_section = True
            elif in_norm_section:
                if line.startswith("mean:"):
                    mean = float(line.split(":")[1].strip())
                elif line.startswith("std:"):
                    std = float(line.split(":")[1].strip())
                    break  # Found both, done

    return mean, std


class RewardStatsCallback(BaseCallback):
    """Callback to log reward statistics from LearnedRewardWrapper."""

    def __init__(self, reward_wrapper, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.reward_wrapper = reward_wrapper
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            stats = self.reward_wrapper.get_stats()
            if stats['total_steps'] > 0:
                self.logger.record("reward/env_reward_avg", stats['avg_env_reward'])
                self.logger.record("reward/learned_reward_avg", stats['avg_learned_reward'])
                self.logger.record("reward/total_steps", stats['total_steps'])
                # Reset stats after logging
                self.reward_wrapper.reset_stats()
        return True


class VIPRewardStatsCallback(BaseCallback):
    """Callback to log reward statistics from VIPRewardWrapper."""

    def __init__(self, vip_wrapper, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.vip_wrapper = vip_wrapper
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            stats = self.vip_wrapper.get_stats()
            if stats['total_steps'] > 0:
                self.logger.record("vip/env_reward_avg", stats['avg_env_reward'])
                self.logger.record("vip/vip_reward_avg", stats['avg_vip_reward'])
                self.logger.record("vip/total_steps", stats['total_steps'])
                # Reset stats after logging
                self.vip_wrapper.reset_stats()
        return True


def train_ppo(
    env,
    total_timesteps: int,
    n_envs: int,
    tensorboard_log: str,
    save_path: str,
    device: str,
    policy_type: str = "MlpPolicy",
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    pretrained_model_path: Optional[str] = None,
    eval_env = None,
    eval_freq: int = 0,
    eval_episodes: int = 10,
    critic_model = None,
    learned_reward_ratio: float = 0.0,
    return_mean: Optional[float] = None,
    return_std: Optional[float] = None,
    seed: int = 0,
    vip_critic = None,
    vip_expert_goals = None,
    vip_reward_mode: str = "value",
    vip_blend_ratio: float = 0.5,
    vip_goal_mode: str = "episode",
    vip_normalize: bool = True,
) -> PPO:
    """
    Train a PPO agent.

    Args:
        env: Vectorized environment
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        tensorboard_log: Path for tensorboard logs
        save_path: Path to save the trained model
        device: Device to use for training ('cuda' or 'cpu')
        policy_type: Policy type ('MlpPolicy' or 'CnnPolicy')
        learning_rate: Learning rate
        batch_size: Batch size for training
        n_epochs: Number of epochs per update
        pretrained_model_path: Optional path to pretrained model to load
        eval_env: Optional evaluation environment
        eval_freq: Evaluation frequency in timesteps (0 to disable)
        eval_episodes: Number of episodes for evaluation
        critic_model: Optional pretrained critic for learned rewards (Bellman-based)
        learned_reward_ratio: Blend ratio for learned rewards [0, 1]
        return_mean: Mean for denormalizing critic predictions
        return_std: Std for denormalizing critic predictions
        seed: Random seed for reproducibility
        vip_critic: Optional VIP critic for goal-conditioned rewards
        vip_expert_goals: Expert goal states for VIP
        vip_reward_mode: VIP reward mode ("value", "td", or "blend")
        vip_blend_ratio: Blend ratio for VIP blend mode [0, 1]
        vip_goal_mode: Goal sampling mode ("episode" or "step")
        vip_normalize: Whether to normalize VIP rewards

    Returns:
        Trained PPO model
    """
    # Wrap environment with reward shaping (VIP or Bellman-based)
    reward_wrapper = None
    vip_wrapper = None

    if vip_critic is not None and vip_expert_goals is not None:
        # Use VIP rewards (goal-conditioned value function)
        print(f"\nWrapping environment with VIP rewards")
        print(f"  Reward mode: {vip_reward_mode}")
        print(f"  Goal sampling: {vip_goal_mode}")
        print(f"  Normalize: {vip_normalize}")
        if vip_reward_mode == "blend":
            print(f"  Blend ratio: {vip_blend_ratio:.2f}")
        vip_wrapper = VIPRewardWrapper(
            env,
            vip_critic=vip_critic,
            expert_goals=vip_expert_goals,
            device=device,
            reward_mode=vip_reward_mode,
            blend_ratio=vip_blend_ratio,
            goal_sample_mode=vip_goal_mode,
            normalize_rewards=vip_normalize,
        )
        env = vip_wrapper
    elif critic_model is not None and learned_reward_ratio > 0:
        # Use Bellman-based learned rewards (backward compatibility)
        print(f"\nWrapping environment with Bellman-based learned rewards (ratio={learned_reward_ratio:.2f})")
        if return_mean is not None and return_std is not None:
            print(f"Using return denormalization: mean={return_mean:.2f}, std={return_std:.2f}")
        reward_wrapper = VecLearnedRewardWrapper(
            env, critic_model, learned_reward_ratio, device,
            return_mean=return_mean, return_std=return_std, gamma=0.99
        )
        env = reward_wrapper

    print("\n" + "="*60)
    if pretrained_model_path:
        print("Loading pretrained PPO agent...")
        print(f"Pretrained model: {pretrained_model_path}")
        model = PPO.load(pretrained_model_path, env=env, device=device)
        # Override tensorboard log path to create new logs
        model.tensorboard_log = tensorboard_log
        print(f"Loaded model. Continuing training on device: {model.device}")
        print(f"Tensorboard logs will be written to: {tensorboard_log}")
    else:
        print("Creating PPO agent...")
        print(f"Policy type: {policy_type}")
        print("="*60)

        # Calculate rollout steps per environment to maintain total buffer size of 2048
        # With vectorized envs, total buffer = rollout_steps_per_env * n_envs
        rollout_steps_per_env = 2048 

        model = PPO(
            policy_type,
            env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=rollout_steps_per_env,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            seed=seed,
        )

        print(f"Model created. Training on device: {model.device}")
        print(f"Rollout buffer size: {rollout_steps_per_env} (steps/env) * {n_envs} (envs) = {rollout_steps_per_env * n_envs} total steps")

    print("\n" + "="*60)
    print(f"Starting PPO training ({total_timesteps:,} timesteps)...")
    if eval_freq > 0 and eval_env is not None:
        print(f"Evaluation enabled every {eval_freq:,} timesteps with {eval_episodes} episodes")
    print("="*60)

    # Setup evaluation callback if enabled
    callbacks = []
    if eval_freq > 0 and eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./best_model_{save_path}",
            log_path=f"./eval_logs_{save_path}",
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
        print(f"Best model will be saved to: ./best_model_{save_path}/")

    # Add reward stats callback if using learned rewards
    if reward_wrapper is not None:
        reward_stats_callback = RewardStatsCallback(reward_wrapper, log_freq=1000)
        callbacks.append(reward_stats_callback)
    elif vip_wrapper is not None:
        vip_stats_callback = VIPRewardStatsCallback(vip_wrapper, log_freq=1000)
        callbacks.append(vip_stats_callback)

    callback = CallbackList(callbacks) if callbacks else None
    # Reset num_timesteps when loading pretrained model to start fresh tensorboard logs
    reset_timesteps = pretrained_model_path is not None
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=reset_timesteps,
    )

    model.save(save_path)
    print(f"\nModel saved as '{save_path}.zip'")

    return model


def train_sac(
    env,
    total_timesteps: int,
    tensorboard_log: str,
    save_path: str,
    device: str,
    policy_type: str = "MlpPolicy",
    learning_rate: float = 3e-4,
    buffer_size: int = 1_000_000,
    learning_starts: int = 1000,
    batch_size: int = 256,
    pretrained_model_path: Optional[str] = None,
    eval_env = None,
    eval_freq: int = 0,
    eval_episodes: int = 10,
    critic_model = None,
    learned_reward_ratio: float = 0.0,
    return_mean: Optional[float] = None,
    return_std: Optional[float] = None,
    seed: int = 0,
    vip_critic = None,
    vip_expert_goals = None,
    vip_reward_mode: str = "value",
    vip_blend_ratio: float = 0.5,
    vip_goal_mode: str = "episode",
    vip_normalize: bool = True,
) -> SAC:
    """
    Train a SAC agent.

    Args:
        env: Vectorized environment
        total_timesteps: Total training timesteps
        tensorboard_log: Path for tensorboard logs
        save_path: Path to save the trained model
        device: Device to use for training ('cuda' or 'cpu')
        policy_type: Policy type ('MlpPolicy' or 'CnnPolicy')
        learning_rate: Learning rate
        buffer_size: Replay buffer size
        learning_starts: Number of steps before training starts
        batch_size: Batch size for training
        pretrained_model_path: Optional path to pretrained model to load
        eval_env: Optional evaluation environment
        eval_freq: Evaluation frequency in timesteps (0 to disable)
        eval_episodes: Number of episodes for evaluation
        critic_model: Optional pretrained critic for learned rewards (Bellman-based)
        learned_reward_ratio: Blend ratio for learned rewards [0, 1]
        return_mean: Mean for denormalizing critic predictions
        return_std: Std for denormalizing critic predictions
        seed: Random seed for reproducibility
        vip_critic: Optional VIP critic for goal-conditioned rewards
        vip_expert_goals: Expert goal states for VIP
        vip_reward_mode: VIP reward mode ("value", "td", or "blend")
        vip_blend_ratio: Blend ratio for VIP blend mode [0, 1]
        vip_goal_mode: Goal sampling mode ("episode" or "step")
        vip_normalize: Whether to normalize VIP rewards

    Returns:
        Trained SAC model
    """
    # Wrap environment with reward shaping (VIP or Bellman-based)
    reward_wrapper = None
    vip_wrapper = None

    if vip_critic is not None and vip_expert_goals is not None:
        # Use VIP rewards (goal-conditioned value function)
        print(f"\nWrapping environment with VIP rewards")
        print(f"  Reward mode: {vip_reward_mode}")
        print(f"  Goal sampling: {vip_goal_mode}")
        print(f"  Normalize: {vip_normalize}")
        if vip_reward_mode == "blend":
            print(f"  Blend ratio: {vip_blend_ratio:.2f}")
        vip_wrapper = VIPRewardWrapper(
            env,
            vip_critic=vip_critic,
            expert_goals=vip_expert_goals,
            device=device,
            reward_mode=vip_reward_mode,
            blend_ratio=vip_blend_ratio,
            goal_sample_mode=vip_goal_mode,
            normalize_rewards=vip_normalize,
        )
        env = vip_wrapper
    elif critic_model is not None and learned_reward_ratio > 0:
        # Use Bellman-based learned rewards (backward compatibility)
        print(f"\nWrapping environment with Bellman-based learned rewards (ratio={learned_reward_ratio:.2f})")
        if return_mean is not None and return_std is not None:
            print(f"Using return denormalization: mean={return_mean:.2f}, std={return_std:.2f}")
        reward_wrapper = VecLearnedRewardWrapper(
            env, critic_model, learned_reward_ratio, device,
            return_mean=return_mean, return_std=return_std, gamma=0.99
        )
        env = reward_wrapper

    print("\n" + "="*60)
    if pretrained_model_path:
        print("Loading pretrained SAC agent...")
        print(f"Pretrained model: {pretrained_model_path}")
        model = SAC.load(pretrained_model_path, env=env, device=device)
        # Override tensorboard log path to create new logs
        model.tensorboard_log = tensorboard_log
        print(f"Loaded model. Continuing training on device: {model.device}")
        print(f"Tensorboard logs will be written to: {tensorboard_log}")
    else:
        print("Creating SAC agent...")
        print(f"Policy type: {policy_type}")
        print("="*60)

        model = SAC(
            policy_type,
            env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            seed=seed,
        )

        print(f"Model created. Training on device: {model.device}")

    print("\n" + "="*60)
    print(f"Starting SAC training ({total_timesteps:,} timesteps)...")
    if eval_freq > 0 and eval_env is not None:
        print(f"Evaluation enabled every {eval_freq:,} timesteps with {eval_episodes} episodes")
    print("="*60)

    # Setup evaluation callback if enabled
    callbacks = []
    if eval_freq > 0 and eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./best_model_{save_path}",
            log_path=f"./eval_logs_{save_path}",
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
        print(f"Best model will be saved to: ./best_model_{save_path}/")

    # Add reward stats callback if using learned rewards
    if reward_wrapper is not None:
        reward_stats_callback = RewardStatsCallback(reward_wrapper, log_freq=1000)
        callbacks.append(reward_stats_callback)
    elif vip_wrapper is not None:
        vip_stats_callback = VIPRewardStatsCallback(vip_wrapper, log_freq=1000)
        callbacks.append(vip_stats_callback)

    callback = CallbackList(callbacks) if callbacks else None
    # Reset num_timesteps when loading pretrained model to start fresh tensorboard logs
    reset_timesteps = pretrained_model_path is not None
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=reset_timesteps,
    )

    model.save(save_path)
    print(f"\nModel saved as '{save_path}.zip'")

    return model


def evaluate_agent(
    model,
    env_config: EnvConfig,
    n_eval_episodes: int = 10,
    render: bool = True,
):
    """
    Evaluate a trained agent.

    Args:
        model: Trained RL model (PPO or SAC)
        env_config: Environment configuration
        n_eval_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    """
    print("\n" + "="*60)
    print("Evaluating trained agent...")
    print("="*60)

    # Create evaluation environment
    render_mode = "human" if render else None
    eval_env = env_config.create_single_env(render_mode=render_mode)

    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        render=render
    )

    print(f"\n--- Evaluation Complete ---")
    print(f"Mean reward over {n_eval_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("-----------------------------")

    eval_env.close()

    return mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(
        description='Train RL agents (PPO/SAC) on various environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PPO on LunarLander
  python train_rl_agent.py --env-type gym --env-name LunarLander-v3 --algo ppo --timesteps 500000

  # Train with periodic evaluation every 10k timesteps
  python train_rl_agent.py --env-type gym --env-name LunarLander-v3 --algo ppo --timesteps 500000 --eval-freq 10000

  # Train from BC-pretrained model
  python train_rl_agent.py --env-type gym --env-name LunarLander-v3 --algo ppo --pretrain-model experiments/run_001 --timesteps 100000

  # Train with Bellman-based learned rewards from pretrained critic (50% blend)
  python train_rl_agent.py --env-type gym --env-name LunarLander-v3 --algo ppo --pretrain-model experiments/run_001 --critic-model experiments/run_001/ppo_init.zip --learned-reward-ratio 0.5 --timesteps 500000

  # Train with VIP (goal-conditioned) rewards
  python train_rl_agent.py --env-type gym --env-name LunarLander-v3 --algo ppo --use-vip-rewards --vip-critic vip_critic_output/vip_critic.pt --vip-demos demos_lunarlander_v3_state.hdf5 --vip-reward-mode value --timesteps 500000

  # Train SAC on LunarLander with fewer parallel environments
  python train_rl_agent.py --env-type gym --env-name LunarLander-v3 --algo sac --timesteps 500000 --n-envs 1

  # Future: Train on robosuite
  python train_rl_agent.py --env-type robosuite --env-name Lift --algo ppo --control-type OSC_POSE
        """
    )

    # Environment arguments
    parser.add_argument('--env-type', type=str, required=True, choices=['gym', 'robosuite'],
                        help='Type of environment (gym or robosuite)')
    parser.add_argument('--env-name', type=str, required=True,
                        help='Name of the environment (e.g., LunarLander-v3, Lift)')
    parser.add_argument('--control-type', type=str, default='OSC_POSE',
                        help='Controller type for robosuite (default: OSC_POSE)')

    # Observation arguments
    parser.add_argument('--use-image-obs', action='store_true',
                        help='Use image observations instead of state (enables CnnPolicy)')
    parser.add_argument('--image-size', type=int, default=84,
                        help='Size to resize images to (default: 84x84)')
    parser.add_argument('--frame-stack', type=int, default=4,
                        help='Number of frames to stack for image observations (default: 4)')

    # Algorithm arguments
    parser.add_argument('--algo', type=str, required=True, choices=['ppo', 'sac'],
                        help='RL algorithm to use (ppo or sac)')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Total training timesteps (default: 500,000)')

    # Training arguments
    parser.add_argument('--n-envs', type=int, default=None,
                        help='Number of parallel environments (default: auto based on CPUs)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: 64 for PPO, 256 for SAC)')
    parser.add_argument('--pretrain-model', type=str, default=None,
                        help='Path to BC pretrain folder (e.g., experiments/run_001)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')

    # SAC-specific arguments
    parser.add_argument('--buffer-size', type=int, default=1_000_000,
                        help='Replay buffer size for SAC (default: 1,000,000)')
    parser.add_argument('--learning-starts', type=int, default=1000,
                        help='Number of steps before SAC training starts (default: 1000)')

    # Evaluation arguments
    parser.add_argument('--eval-freq', type=int, default=0,
                        help='Evaluate every n timesteps during training (0 to disable, default: 0)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation (default: 10)')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip final evaluation after training')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering during final evaluation')

    # Learned reward arguments (Bellman-based)
    parser.add_argument('--critic-model', type=str, default=None,
                        help='Path to pretrained critic model zip file for learned rewards')
    parser.add_argument('--learned-reward-ratio', type=float, default=0.0,
                        help='Blend ratio for learned rewards [0, 1]. 0=env only, 1=learned only (default: 0.0)')

    # VIP reward arguments
    parser.add_argument('--use-vip-rewards', action='store_true',
                        help='Use VIP (goal-conditioned) rewards instead of Bellman-based rewards')
    parser.add_argument('--vip-critic', type=str, default=None,
                        help='Path to trained VIP critic checkpoint (.pt file)')
    parser.add_argument('--vip-demos', type=str, default=None,
                        help='Path to expert demonstrations HDF5 file for goal extraction')
    parser.add_argument('--vip-reward-mode', type=str, default='value',
                        choices=['value', 'td', 'blend'],
                        help='VIP reward mode: "value" (V(s,g)), "td" (temporal difference), "blend" (mix with env) (default: value)')
    parser.add_argument('--vip-blend-ratio', type=float, default=0.5,
                        help='Blend ratio for VIP blend mode [0, 1] (default: 0.5)')
    parser.add_argument('--vip-goal-mode', type=str, default='episode',
                        choices=['episode', 'step'],
                        help='Goal sampling mode: "episode" (per episode) or "step" (per step) (default: episode)')
    parser.add_argument('--vip-goal-sample', type=str, default='final',
                        choices=['final', 'random', 'all'],
                        help='Goal extraction mode from demos: "final" (last state), "random", "all" (default: final)')
    parser.add_argument('--vip-n-demos', type=int, default=100,
                        help='Number of demos to use for goal extraction (default: 100)')
    parser.add_argument('--vip-single-goal', action='store_true',
                        help='Use single best goal instead of multiple goals (more efficient)')
    parser.add_argument('--vip-no-normalize', action='store_true',
                        help='Disable VIP reward normalization')

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_random_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Force CPU for MlpPolicy (faster than GPU for small networks)
    device = "cpu" if args.algo == 'ppo' and not args.use_image_obs else "cuda"
    print("Using CPU (faster than GPU for MlpPolicy)")

    # Determine number of environments
    if args.n_envs is None:
        if args.algo == 'ppo':
            # PPO benefits from more parallel environments
            num_cpus = os.cpu_count() or 4
            args.n_envs = max(2, num_cpus - 1)
        else:  # SAC
            # SAC typically works well with fewer parallel environments
            args.n_envs = 1

    # Determine batch size
    if args.batch_size is None:
        args.batch_size = 64 if args.algo == 'ppo' else 256

    # Create environment configuration
    env_config = EnvConfig(
        env_type=args.env_type,
        env_name=args.env_name,
        control_type=args.control_type if args.env_type == 'robosuite' else None,
        use_image_obs=args.use_image_obs,
        image_size=args.image_size,
        frame_stack=args.frame_stack if args.use_image_obs else 1,
        seed=args.seed,
    )

    # Print configuration
    print("\n" + "="*60)
    print("RL Training Configuration")
    print("="*60)
    print(f"Environment type: {args.env_type}")
    print(f"Environment name: {args.env_name}")
    if args.env_type == 'robosuite':
        print(f"Controller type: {args.control_type}")
    print(f"Observation type: {'Image' if args.use_image_obs else 'State'}")
    if args.use_image_obs:
        print(f"Image size: {args.image_size}x{args.image_size}")
        print(f"Frame stack: {args.frame_stack}")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Training timesteps: {args.timesteps:,}")
    print(f"Number of parallel envs: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    if args.pretrain_model:
        print(f"BC pretrain folder: {args.pretrain_model}")
    if args.critic_model:
        print(f"Critic model: {args.critic_model}")
        print(f"Learned reward ratio: {args.learned_reward_ratio:.2f}")
    if args.use_vip_rewards:
        print(f"VIP Rewards: ENABLED")
        print(f"  VIP critic: {args.vip_critic}")
        print(f"  VIP demos: {args.vip_demos}")
        print(f"  Reward mode: {args.vip_reward_mode}")
        print(f"  Goal sampling: {args.vip_goal_mode}")
        if args.vip_reward_mode == "blend":
            print(f"  Blend ratio: {args.vip_blend_ratio:.2f}")
    if args.eval_freq > 0:
        print(f"Evaluation frequency: Every {args.eval_freq:,} timesteps ({args.eval_episodes} episodes)")
    print(f"Random seed: {args.seed}")
    print(f"Device: {device}")
    print("="*60 + "\n")

    # Create save paths with timestamp for unique runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    obs_type = "img" if args.use_image_obs else "state"
    save_path = f"{args.algo}_{args.env_name.lower().replace('-', '_')}_{obs_type}_{args.timesteps}_{timestamp}"
    tensorboard_log = f"./{args.algo}_{args.env_name.lower().replace('-', '_')}_{obs_type}_tb/{timestamp}/"
    print(f"Model will be saved to: {save_path}.zip")
    print(f"Tensorboard logs will be saved to: {tensorboard_log}\n")

    # Create vectorized environment
    print(f"Creating {args.n_envs} parallel environment(s)...")
    env = env_config.create_vec_env(n_envs=args.n_envs)

    # Get policy type based on environment observation space
    policy_type = env_config.get_policy_type(env=env)
    print(f"Selected policy type: {policy_type}")

    # Convert eval_freq from timesteps to env.step() calls
    # With vectorized envs, each env.step() call advances n_envs timesteps
    if args.eval_freq > 0:
        eval_freq_timesteps = args.eval_freq
        # Convert timesteps to number of env.step() calls
        eval_freq_steps = round(args.eval_freq / args.n_envs)
        eval_freq_steps = max(1, eval_freq_steps)
        args.eval_freq = eval_freq_steps
        actual_timesteps = eval_freq_steps * args.n_envs
        if actual_timesteps != eval_freq_timesteps:
            print(f"\nNote: eval_freq {eval_freq_timesteps:,} timesteps → {eval_freq_steps:,} env steps = {actual_timesteps:,} timesteps")
        else:
            print(f"\neval_freq: {eval_freq_steps:,} env steps = {actual_timesteps:,} timesteps")

    # Create evaluation environment if periodic evaluation is enabled
    eval_env = None
    if args.eval_freq > 0:
        print(f"Creating evaluation environment for periodic evaluation...")
        eval_env = env_config.create_single_env()

    # Handle BC pretrain folder path
    ppo_init_path = None
    critic_model = None
    return_mean, return_std = None, None

    if args.pretrain_model is not None:
        pretrain_dir = Path(args.pretrain_model)
        if not pretrain_dir.exists():
            raise ValueError(f"Pretrain folder not found: {pretrain_dir}")

        # Load policy initialization model
        ppo_init_path = pretrain_dir / "ppo_init.zip"
        if not ppo_init_path.exists():
            raise ValueError(f"PPO init model not found: {ppo_init_path}")

        print(f"\nLoading pretrained policy from: {ppo_init_path}")

        # Parse normalization stats from config if available
        config_path = pretrain_dir / "config.txt"
        return_mean, return_std = parse_normalization_stats(config_path)

    # Load critic model if provided
    if args.critic_model is not None:
        critic_model_path = Path(args.critic_model)
        if not critic_model_path.exists():
            raise ValueError(f"Critic model not found: {critic_model_path}")

        print(f"\nLoading critic model from: {critic_model_path}")
        if args.algo == 'ppo':
            critic_model = PPO.load(str(critic_model_path), device=device)
        else:
            critic_model = SAC.load(str(critic_model_path), device=device)

        # Use normalization stats from pretrain config if available
        if return_mean is not None and return_std is not None:
            print(f"Using return normalization: mean={return_mean:.2f}, std={return_std:.2f}")
        else:
            print("Warning: No return normalization stats available")

        print(f"Critic model loaded. Learned reward ratio: {args.learned_reward_ratio:.2f}")

    # Load VIP critic and extract expert goals if using VIP rewards
    vip_critic_loaded = None
    vip_expert_goals = None

    if args.use_vip_rewards:
        if args.vip_critic is None:
            raise ValueError("--use-vip-rewards requires --vip-critic")

        vip_critic_path = Path(args.vip_critic)

        if not vip_critic_path.exists():
            raise ValueError(f"VIP critic not found: {vip_critic_path}")

        print(f"\n{'='*60}")
        print("Loading VIP critic...")
        print(f"{'='*60}")

        # Load VIP critic checkpoint
        checkpoint = torch.load(str(vip_critic_path), map_location=device)
        use_learned_goal = checkpoint.get("use_learned_goal", False)

        # Create environment to get observation space
        temp_env = gym.make(args.env_name)

        vip_critic_loaded = VIPCritic(
            observation_space=temp_env.observation_space,
            obs_type=checkpoint.get("obs_type", "state"),
            features_dim=checkpoint.get("features_dim", 256),
            use_learned_goal=use_learned_goal,
        ).to(device)
        vip_critic_loaded.load_state_dict(checkpoint["model_state_dict"])
        vip_critic_loaded.eval()
        temp_env.close()

        print(f"VIP critic loaded from: {vip_critic_path}")

        # Extract expert goals (only if not using learned goal)
        if use_learned_goal:
            print("VIP critic uses learned goal embedding - no need to extract expert goals!")
            vip_expert_goals = None
        else:
            if args.vip_demos is None:
                raise ValueError("--vip-demos required when VIP critic doesn't use learned goal embedding")

            vip_demos_path = Path(args.vip_demos)
            if not vip_demos_path.exists():
                raise ValueError(f"VIP demos not found: {vip_demos_path}")

            print("Extracting expert goals...")
            obs_type = "image" if args.use_image_obs else "state"
            vip_expert_goals = extract_expert_goals(
                hdf5_path=str(vip_demos_path),
                n_demos=args.vip_n_demos,
                obs_type=obs_type,
                sample_mode=args.vip_goal_sample,
                use_single_goal=args.vip_single_goal,
            )
            if args.vip_single_goal:
                print(f"Using single canonical goal (best final state)")
            else:
                print(f"Extracted {len(vip_expert_goals)} expert goals")

        print(f"VIP configuration:")
        print(f"  Reward mode: {args.vip_reward_mode}")
        if not use_learned_goal:
            print(f"  Goal sampling: {args.vip_goal_mode}")
        print(f"  Normalize: {not args.vip_no_normalize}")
        if args.vip_reward_mode == "blend":
            print(f"  Blend ratio: {args.vip_blend_ratio:.2f}")

    # Train the agent
    if args.algo == 'ppo':
        model = train_ppo(
            env=env,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            tensorboard_log=tensorboard_log,
            save_path=save_path,
            device=device,
            policy_type=policy_type,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            pretrained_model_path=str(ppo_init_path) if ppo_init_path else None,
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            critic_model=critic_model,
            learned_reward_ratio=args.learned_reward_ratio,
            return_mean=return_mean,
            return_std=return_std,
            seed=args.seed,
            vip_critic=vip_critic_loaded,
            vip_expert_goals=vip_expert_goals,
            vip_reward_mode=args.vip_reward_mode,
            vip_blend_ratio=args.vip_blend_ratio,
            vip_goal_mode=args.vip_goal_mode,
            vip_normalize=not args.vip_no_normalize,
        )
    elif args.algo == 'sac':
        model = train_sac(
            env=env,
            total_timesteps=args.timesteps,
            tensorboard_log=tensorboard_log,
            save_path=save_path,
            device=device,
            policy_type=policy_type,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            pretrained_model_path=str(ppo_init_path) if ppo_init_path else None,
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            critic_model=critic_model,
            learned_reward_ratio=args.learned_reward_ratio,
            return_mean=return_mean,
            return_std=return_std,
            seed=args.seed,
            vip_critic=vip_critic_loaded,
            vip_expert_goals=vip_expert_goals,
            vip_reward_mode=args.vip_reward_mode,
            vip_blend_ratio=args.vip_blend_ratio,
            vip_goal_mode=args.vip_goal_mode,
            vip_normalize=not args.vip_no_normalize,
        )

    # Close evaluation environment if it was created
    if eval_env is not None:
        eval_env.close()

    # Close training environment
    env.close()

    # Evaluate the agent
    if not args.no_eval:
        evaluate_agent(
            model=model,
            env_config=env_config,
            n_eval_episodes=args.eval_episodes,
            render=not args.no_render
        )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
