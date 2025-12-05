#!/bin/bash

# Example training scripts for LunarLander environment
# Make this file executable with: chmod +x train_lunar_lander.sh

echo "Training RL agents on LunarLander-v3"
echo "======================================"

# Example 1: Train with PPO (recommended for discrete/continuous action spaces)
echo -e "\n[Example 1] Training with PPO for 500k timesteps..."
python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 500000 \
    --n-envs 4 \
    --learning-rate 3e-4 \
    --batch-size 64

# Example 2: Train with SAC (good for continuous action spaces)
# Note: LunarLander-v3 has continuous actions by default
# echo -e "\n[Example 2] Training with SAC for 500k timesteps..."
# python -m train_reward_yourself.train_rl_agent \
#     --env-type gym \
#     --env-name LunarLander-v3 \
#     --algo sac \
#     --timesteps 500000 \
#     --n-envs 1 \
#     --learning-rate 3e-4 \
#     --batch-size 256 \
#     --buffer-size 1000000

# Example 3: Quick test run (shorter training)
# echo -e "\n[Example 3] Quick test run with PPO for 10k timesteps..."
# python -m train_reward_yourself.train_rl_agent \
#     --env-type gym \
#     --env-name LunarLander-v3 \
#     --algo ppo \
#     --timesteps 10000 \
#     --n-envs 2 \
#     --no-render

# Example 4: Train without evaluation
# echo -e "\n[Example 4] Training PPO without final evaluation..."
# python -m train_reward_yourself.train_rl_agent \
#     --env-type gym \
#     --env-name LunarLander-v3 \
#     --algo ppo \
#     --timesteps 500000 \
#     --no-eval

echo -e "\nTraining complete! Check the tensorboard logs:"
echo "tensorboard --logdir ./ppo_lunarlander_v3_tensorboard/"
