#!/usr/bin/env python3
"""
Fine-tune a BC-pretrained model using PPO.

This script loads a BC model and continues training with PPO,
which can improve performance beyond imitation learning.

Usage:
    python -m train_reward_yourself.finetune_bc_with_ppo \
        --bc-model bc_lunarlander.zip \
        --env-name LunarLander-v3 \
        --total-timesteps 100000 \
        --output ppo_finetuned_lunarlander.zip
"""

import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BC model with PPO"
    )

    # Model arguments
    parser.add_argument("--bc-model", type=str, required=True,
                        help="Path to BC pretrained model (.zip)")

    # Environment arguments
    parser.add_argument("--env-name", type=str, required=True,
                        help="Environment name")
    parser.add_argument("--env-type", type=str, default="gym", choices=["gym", "robosuite"],
                        help="Environment type")

    # PPO training arguments
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="Total timesteps for PPO training (default: 100000)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps per update (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of epochs per update (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--use-vecnormalize", action="store_true",
                        help="Use VecNormalize wrapper")

    # Evaluation arguments
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluation frequency in timesteps (default: 10000)")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of evaluation episodes (default: 5)")

    # Output arguments
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for finetuned model (.zip)")
    parser.add_argument("--tensorboard-log", type=str, default=None,
                        help="TensorBoard log directory")
    parser.add_argument("--save-freq", type=int, default=50000,
                        help="Save checkpoint every N timesteps (default: 50000)")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("BC + PPO Fine-tuning Configuration")
    print("="*60)
    print(f"BC Model:         {args.bc_model}")
    print(f"Environment:      {args.env_name}")
    print(f"Total timesteps:  {args.total_timesteps}")
    print(f"Learning rate:    {args.learning_rate}")
    print(f"Output:           {args.output}")
    print("="*60 + "\n")

    # Create environment
    def make_env():
        env = gym.make(args.env_name)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    if args.use_vecnormalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Create eval environment
    eval_env = DummyVecEnv([make_env])
    if args.use_vecnormalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Load BC model
    print("Loading BC pretrained model...")
    model = PPO.load(args.bc_model, env=env)

    # Update PPO hyperparameters for fine-tuning
    model.learning_rate = args.learning_rate
    model.n_steps = args.n_steps
    model.batch_size = args.batch_size
    model.n_epochs = args.n_epochs
    model.gamma = args.gamma

    if args.tensorboard_log:
        model.tensorboard_log = args.tensorboard_log

    print("BC model loaded successfully!")
    print("Starting PPO fine-tuning...\n")

    # Setup callbacks
    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(Path(args.output).parent / "best_model"),
        log_path=str(Path(args.output).parent / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(Path(args.output).parent / "checkpoints"),
        name_prefix="ppo_bc_finetune",
    )
    callbacks.append(checkpoint_callback)

    # Train with PPO
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    model.save(args.output)
    print(f"\nFinetuned model saved to: {args.output}")

    # Save VecNormalize stats if used
    if args.use_vecnormalize:
        vecnorm_path = args.output.replace(".zip", "_vecnormalize.pkl")
        env.save(vecnorm_path)
        print(f"VecNormalize stats saved to: {vecnorm_path}")

    env.close()
    eval_env.close()

    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()
