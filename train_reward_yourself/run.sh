# Aza's random scripts.

#0. Train expert

python train_reward_yourself/train_rl_agent.py --env-type gym --env-name LunarLander-v3 --algo ppo --timesteps 1

# Tensorboard
tensorboard --logdir=/home/atulep/Research/cs1952_final/train_reward_yourself/ppo_lunarlander_v3_state_tensorboard/PPO_3

# Eval expert

python train_reward_yourself/eval_rl_agent.py --model-path ppo_LunarLander-v3_combined.zip --env-type gym --env-name LunarLander-v3 --algo ppo


#1. Collect demos

python train_reward_yourself/collect_demonstrations.py \
    --model-path ppo_lunar_lander_v3_mingxi.zip \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --n-demos 5000 \
    --output demos_lunarlander_mingxi.hdf5


# Train RM

python train_reward_yourself/train_rm_from_demos.py  --demos demos_lunarlander_mingxi.hdf5  --env LunarLander-v3  --steps 500000

# Sanity check RM

python train_reward_yourself/test_experty_vs_random.py --demos demos_lunarlander.hdf5

