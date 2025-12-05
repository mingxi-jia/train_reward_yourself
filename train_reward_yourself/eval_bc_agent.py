#!/usr/bin/env python3
"""
Evaluate a trained Behavioral Cloning agent.

Usage:
    python -m train_reward_yourself.eval_bc_agent \
        --model bc_lunarlander.zip \
        --env-name LunarLander-v3 \
        --episodes 10
"""

import argparse
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from stable_baselines3 import PPO


def evaluate_agent(model, env: gym.Env, num_episodes: int = 10, render: bool = False):
    """Evaluate an agent in the environment."""

    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check for success if available
        if 'is_success' in info or 'success' in info:
            successes.append(info.get('is_success', info.get('success', False)))

        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }

    if successes:
        results["success_rate"] = np.mean(successes)

    return results, episode_rewards


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BC agent"
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.zip)")
    parser.add_argument("--env-name", type=str, required=True,
                        help="Environment name")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during evaluation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for evaluation")

    args = parser.parse_args()

    # Create environment
    if args.render:
        env = gym.make(args.env_name, render_mode="human")
    else:
        env = gym.make(args.env_name)

    if args.seed is not None:
        env.reset(seed=args.seed)

    # Load model
    print(f"Loading model from: {args.model}")
    model = PPO.load(args.model, env=env)

    # Evaluate
    print(f"\nEvaluating on {args.env_name} for {args.episodes} episodes...")
    results, episode_rewards = evaluate_agent(
        model, env, num_episodes=args.episodes, render=args.render
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes:      {args.episodes}")
    print(f"Mean Reward:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Min Reward:    {results['min_reward']:.2f}")
    print(f"Max Reward:    {results['max_reward']:.2f}")
    print(f"Mean Length:   {results['mean_length']:.1f} ± {results['std_length']:.1f}")

    if 'success_rate' in results:
        print(f"Success Rate:  {results['success_rate']*100:.1f}%")

    print("="*60)

    env.close()


if __name__ == "__main__":
    main()
