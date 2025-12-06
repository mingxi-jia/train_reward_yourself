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
import gymnasium as gym
from typing import Optional

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

# Import shared utilities
from train_reward_yourself.env_utils import (
    check_gpu,
    EnvConfig,
)

# Import panda_gym to register Panda environments (if available)
try:
    import panda_gym
except ImportError:
    pass  # panda_gym not installed, skip

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

    Returns:
        Trained PPO model
    """
    print("\n" + "="*60)
    if pretrained_model_path:
        print("Loading pretrained PPO agent...")
        print(f"Pretrained model: {pretrained_model_path}")
        model = PPO.load(pretrained_model_path, env=env, device=device)
        print(f"Loaded model. Continuing training on device: {model.device}")
    else:
        print("Creating PPO agent...")
        print(f"Policy type: {policy_type}")
        print("="*60)

        # Calculate rollout steps per environment
        rollout_steps_per_env = 2048 // max(n_envs, 1)

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

    callback = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=total_timesteps, log_interval=total_timesteps, progress_bar=True, callback=callback)

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

    Returns:
        Trained SAC model
    """
    print("\n" + "="*60)
    if pretrained_model_path:
        print("Loading pretrained SAC agent...")
        print(f"Pretrained model: {pretrained_model_path}")
        model = SAC.load(pretrained_model_path, env=env, device=device)
        print(f"Loaded model. Continuing training on device: {model.device}")
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

    callback = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=total_timesteps, log_interval=10, progress_bar=True, callback=callback)

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

  # Train from BC-pretrained model with evaluation
  python train_rl_agent.py --env-type gym --env-name LunarLander-v3 --algo ppo --pretrain-model test.zip --timesteps 100000 --eval-freq 5000

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
                        help='Path to pretrained model to load (e.g., test.zip)')
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

    args = parser.parse_args()

    # Check for GPU
    device = check_gpu()

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
        print(f"Pretrained model: {args.pretrain_model}")
    if args.eval_freq > 0:
        print(f"Evaluation frequency: Every {args.eval_freq:,} timesteps ({args.eval_episodes} episodes)")
    print(f"Random seed: {args.seed}")
    print(f"Device: {device}")
    print("="*60 + "\n")

    # Create save paths
    obs_type = "img" if args.use_image_obs else "state"
    save_path = f"{args.algo}_{args.env_name.lower().replace('-', '_')}_{obs_type}_{args.timesteps}"
    tensorboard_log = f"./{args.algo}_{args.env_name.lower().replace('-', '_')}_{obs_type}_tensorboard/"

    # Create vectorized environment
    print(f"Creating {args.n_envs} parallel environment(s)...")
    env = env_config.create_vec_env(n_envs=args.n_envs)

    # Get policy type based on environment observation space
    policy_type = env_config.get_policy_type(env=env)
    print(f"Selected policy type: {policy_type}")

    # Create evaluation environment if periodic evaluation is enabled
    eval_env = None
    if args.eval_freq > 0:
        print(f"Creating evaluation environment for periodic evaluation...")
        eval_env = env_config.create_single_env()

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
            pretrained_model_path=args.pretrain_model,
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
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
            pretrained_model_path=args.pretrain_model,
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
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
