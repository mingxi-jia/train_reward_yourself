#!/bin/bash

# Example training scripts for LunarLander environment
# Make this file executable with: chmod +x train_lunar_lander.sh

echo "Training RL agents on LunarLander-v3"
echo "======================================"

# Baseline 
python train_reward_yourself/train_rl_agent.py --env-type gym --env-name LunarLander-v3  --n-envs 4 --algo ppo --timesteps 100000 --eval-freq 20000
# --- Evaluation Complete ---
# Mean reward over 10 episodes: -554.15 +/- 107.81

# pretrain optimal policy
python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 500000 \
    --n-envs 4 \
    --eval-freq 20000

# collect demos
python train_reward_yourself/collect_demonstrations.py --model-path ppo_lunarlander_v3_500000.zip --env-type gym --env-name LunarLander-v3 --algo ppo --n-demos 300 --output demos_with_images.hdf5 

# Pretrained with BC
python train_reward_yourself/train_bc_agent.py --data=demos_with_images.hdf5 --env-name=LunarLander-v3 --n-demos=10 --epochs=50 --pretrain-critic --output=experiments/run_001 

# direct eval bc agent
python train_reward_yourself/eval_bc_agent.py --model=ppo_lunar_demo100.zip --env-name=LunarLander-v3 --render

# Train from learned rewards only
python -m train_reward_yourself.train_rl_agent \
      --env-type gym --env-name LunarLander-v3 --n-envs 4 \
      --algo ppo --timesteps 500000 --eval-freq 20000 \
      --critic-model experiments/run_demo100/ppo_init.zip  --learned-reward-ratio 0.3 

# Finetune with PPO (actor and critic from BC)
python -m train_reward_yourself.train_rl_agent \
      --env-type gym --env-name LunarLander-v3 --n-envs 4 \
      --algo ppo --timesteps 500000 --eval-freq 20000 \
      --pretrain-model experiments/run_demo100

# Finetune with PPO + learned rewards
python -m train_reward_yourself.train_rl_agent \
      --env-type gym --env-name LunarLander-v3 --n-envs 4 \
      --algo ppo --timesteps 500000 --eval-freq 20000 \
      --critic-model experiments/run_001/ppo_init.zip  --learned-reward-ratio 0.5 \
      --pretrain-model experiments/run_001 

#--- Evaluation Complete (Finetune with PPO)---
#Mean reward over 10 episodes: 214.96 +/- 79.68


python -m train_reward_yourself.train_rl_agent --env-type gym --env-name LunarLander-v3 --algo ppo --timesteps 500000 --n-envs 4 --eval-freq 20000
python -m train_reward_yourself.train_rl_agent --env-type gym --env-name LunarLander-v3 --n-envs 4 --algo ppo --timesteps 500000 --eval-freq 20000