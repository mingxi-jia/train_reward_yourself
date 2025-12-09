#!/bin/bash

# Example training scripts for LunarLander environment
# Make this file executable with: chmod +x train_lunar_lander.sh

echo "Training RL agents on LunarLander-v3"
echo "======================================"

# Baseline 
python train_reward_yourself/train_rl_agent.py --env-type gym --env-name LunarLander-v3  --n-envs 4 --algo ppo --timesteps 50000
# --- Evaluation Complete ---
# Mean reward over 10 episodes: -554.15 +/- 107.81

# pretrain optimal policy
python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 500000 \
    --n-envs 4

# collect demos
python train_reward_yourself/collect_demonstrations.py --model-path ppo_lunarlander_v3_500000.zip --env-type gym --env-name LunarLander-v3 --algo ppo --n-demos 10 --output demos_with_images.hdf5 

# Pretrained with BC
python train_reward_yourself/train_bc_agent.py --data=demos_with_images.hdf5 --env-name=LunarLander-v3 --output=ppo_lunar_demo10 --n-demos=10 --epochs=50 --pretrain-critic

# direct eval bc agent
python train_reward_yourself/eval_bc_agent.py --model=ppo_lunar_demo100.zip --env-name=LunarLander-v3 --render
# Evaluation Results (10 episodes):
#   Mean Reward: 124.57 ± 259.71
#   Mean Length: 311.3 ± 76.8

# Finetune with PPO
python train_reward_yourself/train_rl_agent.py --env-type gym --env-name LunarLander-v3  --n-envs 4 --algo ppo --timesteps 500000 --eval-freq 20000 --pretrain-model ppo_lunar_demo10.zip 

#--- Evaluation Complete (Finetune with PPO)---
#Mean reward over 10 episodes: 214.96 +/- 79.68