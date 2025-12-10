#!/bin/bash

#==================================
# pretrain optimal policy
python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 500000 \
    --n-envs 4 \
    --eval-freq 20000

# collect demos
python train_reward_yourself/collect_demonstrations.py --model-path ppo_lunarlander_v3_500000.zip --env-type gym --env-name LunarLander-v3 --algo ppo --n-demos 100 --output demos_with_images.hdf5 --min-reward=200

# Pretrained with BC
python train_reward_yourself/train_bc_agent.py --data=demos_with_images.hdf5 --env-name=LunarLander-v3 --n-demos=10 --epochs=50 --pretrain-critic --output=experiments/run_001 

# direct eval bc agent
python train_reward_yourself/eval_bc_agent.py --model=ppo_lunar_demo100.zip --env-name=LunarLander-v3 --render

#==================================

# Finetune with PPO + learned rewards
python -m train_reward_yourself.train_rl_agent \
      --env-type gym --env-name LunarLander-v3 --n-envs 4 \
      --algo ppo --timesteps 500000 --eval-freq 20000 \
      --critic-model experiments/run_demo100/ppo_init.zip --learned-reward-ratio 0.5 

python -m train_reward_yourself.train_rl_agent \
      --env-type gym --env-name LunarLander-v3 --n-envs 4 \
      --algo ppo --timesteps 500000 --eval-freq 20000 \
      --critic-model experiments/run_demo100/ppo_init.zip  --learned-reward-ratio 0.8 

python -m train_reward_yourself.train_rl_agent \
      --env-type gym --env-name LunarLander-v3 --n-envs 4 \
      --algo ppo --timesteps 500000 --eval-freq 20000 \
      --critic-model experiments/run_demo100/ppo_init.zip  --learned-reward-ratio 1.0 


#============VIP================
python -m train_reward_yourself.train_vip_critic \
      --data demos_with_images.hdf5 \
      --env-name LunarLander-v3 \
      --obs-type state \
      --output vip_critic_learned_goal \
      --n-demos 300 \
      --epochs 20 \
      --visualize \
      --vis-demo-idx 0

python -m train_reward_yourself.eval_bc_agent \
      --model experiments/run_demo100/ppo_init.zip \
      --env-name LunarLander-v3 \
      --episodes 10 \
      --save-video \
      --vip-critic vip_critic_output/vip_critic.pt \
      --vip-demos demos_with_images.hdf5 \
      --vip-single-goal \
      --video-path eval_vip.mp4

python -m train_reward_yourself.train_rl_agent \
      --env-type gym \
      --env-name LunarLander-v3 \
      --algo ppo \
      --use-vip-rewards \
      --vip-critic vip_critic_output/vip_critic.pt \
      --vip-demos demos_with_images.hdf5 \
      --vip-single-goal \
      --vip-reward-mode td \
      --vip-blend-ratio 0.8 \
      --timesteps 500000 --eval-freq 20000

