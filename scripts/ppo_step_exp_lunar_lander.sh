#!/bin/bash

# Example training scripts for LunarLander environment
# Make this file executable with: chmod +x train_lunar_lander.sh

echo "Training RL agents on LunarLander-v3"
echo "We want to see how many training steps are needed for ppo from scratch."

python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 500000 \
    --n-envs 4

python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 400000 \
    --n-envs 4

python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 300000 \
    --n-envs 4

python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 200000 \
    --n-envs 4

python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 100000 \
    --n-envs 4