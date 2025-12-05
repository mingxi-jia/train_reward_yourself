"""
RL Agent Evaluation Script

Evaluates trained PPO or SAC agents on:
- Classic control environments (LunarLander, etc.)
- Robosuite robotic manipulation environments

Features:
- Load and evaluate trained models
- Render episodes to visualize policy behavior
- Record videos of evaluation episodes
- Compute detailed statistics (mean/std reward, success rate, etc.)
- Support for both gym and robosuite environments

Usage:
    # Evaluate PPO model on LunarLander
    python eval_rl_agent.py --model-path ppo_lunarlander_v3_500000.zip --env-type gym --env-name LunarLander-v3 --algo ppo

    # Evaluate with video recording
    python eval_rl_agent.py --model-path ppo_lunarlander_v3_500000.zip --env-type gym --env-name LunarLander-v3 --algo ppo --record-video

    # Evaluate SAC model
    python eval_rl_agent.py --model-path sac_lunarlander_v3_500000.zip --env-type gym --env-name LunarLander-v3 --algo sac --episodes 20

    # Future: Evaluate on robosuite
    python eval_rl_agent.py --model-path ppo_lift_500000.zip --env-type robosuite --env-name Lift --algo ppo --control-type OSC_POSE
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from typing import Tuple, List
import json
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Import shared utilities
from train_reward_yourself.env_utils import EnvConfig

# Import panda_gym to register Panda environments (if available)
try:
    import panda_gym
except ImportError:
    pass  # panda_gym not installed, skip


def load_model(model_path: str, algo: str, env):
    """
    Load a trained model.

    Args:
        model_path: Path to the saved model (.zip file)
        algo: Algorithm used ('ppo' or 'sac')
        env: Environment for the model

    Returns:
        Loaded model
    """
    print(f"\nLoading {algo.upper()} model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if algo == 'ppo':
        model = PPO.load(model_path, env=env)
    elif algo == 'sac':
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    print(f"Model loaded successfully!")
    return model


def evaluate_episodes(
    model,
    env,
    n_episodes: int = 10,
    render: bool = True,
    deterministic: bool = True,
    verbose: bool = True
) -> Tuple[List[float], List[int], List[bool]]:
    """
    Evaluate the model for multiple episodes.

    Args:
        model: Trained RL model
        env: Environment to evaluate on
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
        verbose: Whether to print episode details

    Returns:
        Tuple of (episode_rewards, episode_lengths, episode_successes)
    """
    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    print(f"\n{'='*60}")
    print(f"Running {n_episodes} evaluation episodes...")
    print(f"Deterministic: {deterministic}")
    print(f"{'='*60}\n")

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        # Check if episode was successful (environment-specific)
        success = info.get('is_success', False) or info.get('success', False)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(success)

        if verbose:
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Length = {episode_length}, "
                  f"Success = {success}")

    return episode_rewards, episode_lengths, episode_successes


def print_evaluation_summary(
    episode_rewards: List[float],
    episode_lengths: List[int],
    episode_successes: List[bool]
):
    """Print a summary of evaluation results."""
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Number of episodes: {len(episode_rewards)}")
    print(f"\nRewards:")
    print(f"  Mean: {rewards_array.mean():.2f}")
    print(f"  Std:  {rewards_array.std():.2f}")
    print(f"  Min:  {rewards_array.min():.2f}")
    print(f"  Max:  {rewards_array.max():.2f}")
    print(f"\nEpisode Lengths:")
    print(f"  Mean: {lengths_array.mean():.1f}")
    print(f"  Std:  {lengths_array.std():.1f}")
    print(f"  Min:  {lengths_array.min()}")
    print(f"  Max:  {lengths_array.max()}")

    if any(episode_successes):
        success_rate = sum(episode_successes) / len(episode_successes) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}% ({sum(episode_successes)}/{len(episode_successes)})")

    print(f"{'='*60}\n")


def save_evaluation_results(
    results_path: str,
    episode_rewards: List[float],
    episode_lengths: List[int],
    episode_successes: List[bool],
    config: dict
):
    """Save evaluation results to a JSON file."""
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)

    results = {
        "config": config,
        "statistics": {
            "num_episodes": len(episode_rewards),
            "reward_mean": float(rewards_array.mean()),
            "reward_std": float(rewards_array.std()),
            "reward_min": float(rewards_array.min()),
            "reward_max": float(rewards_array.max()),
            "length_mean": float(lengths_array.mean()),
            "length_std": float(lengths_array.std()),
            "length_min": int(lengths_array.min()),
            "length_max": int(lengths_array.max()),
        },
        "episodes": {
            "rewards": [float(r) for r in episode_rewards],
            "lengths": [int(l) for l in episode_lengths],
            "successes": [bool(s) for s in episode_successes],
        }
    }

    if any(episode_successes):
        success_rate = sum(episode_successes) / len(episode_successes) * 100
        results["statistics"]["success_rate"] = float(success_rate)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results saved to: {results_path}")


def record_video_episodes(
    model,
    env_config: EnvConfig,
    n_episodes: int = 3,
    video_folder: str = "./videos",
    deterministic: bool = True,
):
    """
    Record video of evaluation episodes.

    Args:
        model: Trained RL model
        env_config: Environment configuration
        n_episodes: Number of episodes to record
        video_folder: Folder to save videos
        deterministic: Whether to use deterministic actions
    """
    print(f"\n{'='*60}")
    print(f"Recording {n_episodes} video episodes...")
    print(f"Video folder: {video_folder}")
    print(f"{'='*60}\n")

    # Create video folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)

    if env_config.env_type == "gym":
        # Use gymnasium's RecordVideo wrapper
        from gymnasium.wrappers import RecordVideo

        env = gym.make(env_config.env_name, render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda x: True,  # Record all episodes
            name_prefix=f"eval-{env_config.env_name}"
        )
    elif env_config.env_type == "robosuite":
        # For robosuite, we'd need custom video recording
        # This is a placeholder for future implementation
        print("Video recording for robosuite not yet implemented.")
        print("Use --render flag to visualize in real-time instead.")
        return
    else:
        raise ValueError(f"Unknown env_type: {env_config.env_type}")

    # Run episodes
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        print(f"Episode {episode + 1}/{n_episodes} recorded: Reward = {episode_reward:.2f}")

    env.close()
    print(f"\nVideos saved to: {video_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained RL agents (PPO/SAC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate PPO model on LunarLander
  python eval_rl_agent.py --model-path ppo_lunarlander_v3_500000.zip --env-type gym --env-name LunarLander-v3 --algo ppo

  # Evaluate with video recording
  python eval_rl_agent.py --model-path ppo_lunarlander_v3_500000.zip --env-type gym --env-name LunarLander-v3 --algo ppo --record-video

  # Evaluate without rendering (faster)
  python eval_rl_agent.py --model-path sac_lunarlander_v3_500000.zip --env-type gym --env-name LunarLander-v3 --algo sac --no-render --episodes 50

  # Future: Evaluate on robosuite
  python eval_rl_agent.py --model-path ppo_lift_500000.zip --env-type robosuite --env-name Lift --algo ppo --control-type OSC_POSE
        """
    )

    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model (.zip file)')
    parser.add_argument('--algo', type=str, required=True, choices=['ppo', 'sac'],
                        help='Algorithm used to train the model')
    parser.add_argument('--vecnormalize-path', type=str, default=None,
                        help='Path to VecNormalize stats (.pkl file) if used during training')

    # Environment arguments
    parser.add_argument('--env-type', type=str, required=True, choices=['gym', 'robosuite'],
                        help='Type of environment')
    parser.add_argument('--env-name', type=str, required=True,
                        help='Name of the environment')
    parser.add_argument('--control-type', type=str, default='OSC_POSE',
                        help='Controller type for robosuite (default: OSC_POSE)')

    # Observation arguments (must match training configuration)
    parser.add_argument('--use-image-obs', action='store_true',
                        help='Use image observations (must match training config)')
    parser.add_argument('--image-size', type=int, default=84,
                        help='Image size (must match training config, default: 84)')
    parser.add_argument('--frame-stack', type=int, default=4,
                        help='Frame stack size (must match training config, default: 4)')

    # Evaluation arguments
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic actions (default: True)')
    parser.add_argument('--stochastic', dest='deterministic', action='store_false',
                        help='Use stochastic actions')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')

    # Rendering and recording arguments
    parser.add_argument('--render', action='store_true', default=True,
                        help='Render episodes (default: True)')
    parser.add_argument('--no-render', dest='render', action='store_false',
                        help='Disable rendering')
    parser.add_argument('--record-video', action='store_true',
                        help='Record videos of evaluation episodes')
    parser.add_argument('--video-folder', type=str, default='./videos',
                        help='Folder to save videos (default: ./videos)')
    parser.add_argument('--video-episodes', type=int, default=3,
                        help='Number of episodes to record (default: 3)')

    # Output arguments
    parser.add_argument('--save-results', action='store_true',
                        help='Save evaluation results to JSON file')
    parser.add_argument('--results-path', type=str, default=None,
                        help='Path to save results JSON (default: auto-generated)')

    args = parser.parse_args()

    # Auto-generate results path if saving but no path specified
    if args.save_results and args.results_path is None:
        model_name = Path(args.model_path).stem
        args.results_path = f"eval_results_{model_name}.json"

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
    print("RL Agent Evaluation Configuration")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Environment type: {args.env_type}")
    print(f"Environment name: {args.env_name}")
    if args.env_type == 'robosuite':
        print(f"Controller type: {args.control_type}")
    print(f"Observation type: {'Image' if args.use_image_obs else 'State'}")
    if args.use_image_obs:
        print(f"Image size: {args.image_size}x{args.image_size}")
        print(f"Frame stack: {args.frame_stack}")
    print(f"Number of episodes: {args.episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Render: {args.render}")
    print(f"Record video: {args.record_video}")
    if args.save_results:
        print(f"Save results: {args.results_path}")
    print("="*60 + "\n")

    # Create environment
    render_mode = "human" if args.render and not args.record_video else None
    env = env_config.create_single_env(render_mode=render_mode)

    # Load VecNormalize wrapper if provided
    if args.vecnormalize_path is not None:
        print(f"Loading VecNormalize stats from: {args.vecnormalize_path}")
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(args.vecnormalize_path, env)
        env.training = False  # Don't update stats during evaluation
        env.norm_reward = False  # Don't normalize rewards during evaluation
        print("VecNormalize stats loaded!")

    # Load model
    model = load_model(args.model_path, args.algo, env)

    # Record video if requested
    if args.record_video:
        record_video_episodes(
            model=model,
            env_config=env_config,
            n_episodes=args.video_episodes,
            video_folder=args.video_folder,
            deterministic=args.deterministic,
        )

        # Recreate environment for normal evaluation if we're also doing that
        if args.episodes > 0:
            render_mode = "human" if args.render else None
            env = env_config.create_single_env(render_mode=render_mode)

            if args.vecnormalize_path is not None:
                env = DummyVecEnv([lambda: env])
                env = VecNormalize.load(args.vecnormalize_path, env)
                env.training = False
                env.norm_reward = False

    # Evaluate episodes
    if args.episodes > 0:
        episode_rewards, episode_lengths, episode_successes = evaluate_episodes(
            model=model,
            env=env,
            n_episodes=args.episodes,
            render=args.render,
            deterministic=args.deterministic,
            verbose=True
        )

        # Print summary
        print_evaluation_summary(episode_rewards, episode_lengths, episode_successes)

        # Save results if requested
        if args.save_results:
            config = {
                "model_path": args.model_path,
                "algorithm": args.algo,
                "n_episodes": args.episodes,
                "deterministic": args.deterministic,
                "env_config": env_config.to_dict(),
            }
            save_evaluation_results(
                args.results_path,
                episode_rewards,
                episode_lengths,
                episode_successes,
                config
            )

    # Close environment
    env.close()
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
