"""

python train_reward_yourself/train_rl_with_custom_rm.py \
    --env-name LunarLander-v3 \
    --algo ppo \
    --reward-path learned_reward_model.pt \
    --env-weight 0.5 \
    --learned-weight 0.5 \
    --timesteps 500000
"""

import os
import warnings
import argparse
import gymnasium as gym
import torch
import torch.nn.functional as F 
import numpy as np
from typing import Optional

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnvWrapper

# Import imitation libraries for the reward net
from imitation.rewards.reward_nets import BasicRewardNet

# Import shared utilities
from train_reward_yourself.env_utils import check_gpu, EnvConfig

try:
    import panda_gym
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class CombinedRewardVecWrapper(VecEnvWrapper):
    """
    Wraps a VecEnv to add a learned reward to the environment reward.
    Automatically handles One-Hot encoding for Discrete action spaces.
    """
    def __init__(self, venv, reward_net, env_weight=1.0, learned_weight=1.0, device="cpu"):
        super().__init__(venv)
        self.reward_net = reward_net.to(device)
        self.env_weight = env_weight
        self.learned_weight = learned_weight
        self.device = device
        self.reward_net.eval()
        
        # Check if action space is discrete to handle one-hot encoding
        self.is_discrete = isinstance(venv.action_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.n_actions = venv.action_space.n

    def reset(self):
        obs = self.venv.reset()
        self._last_obs = obs 
        return obs

    def step_async(self, actions):
        # Cache actions for step_wait, needed for AIRL to work correctly, since it needs access to s_prime.
        self._last_actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        with torch.no_grad():
            # Shape: (Batch_Size, Obs_Dim)
            obs_t = torch.as_tensor(self._last_obs, dtype=torch.float32).to(self.device)
            next_obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
            dones_t = torch.as_tensor(dones, dtype=torch.bool).to(self.device)

            if self.is_discrete:
                # One-Hot Encode: (Batch,) -> (Batch, Num_Actions)
                raw_actions = torch.as_tensor(self._last_actions, dtype=torch.long).to(self.device)
                act_t = F.one_hot(raw_actions, num_classes=self.n_actions).float()
            else:
                # Continuous: (Batch, Action_Dim)
                act_t = torch.as_tensor(self._last_actions, dtype=torch.float32).to(self.device)
                # Ensure it is at least 2D [Batch, Dim]
                if act_t.ndim == 1:
                    act_t = act_t.unsqueeze(1)

            learned_rewards_t = self.reward_net(obs_t, act_t, next_obs_t, dones_t)
            
            if isinstance(learned_rewards_t, tuple):
                learned_rewards_t = learned_rewards_t[0]
            
            learned_rewards = learned_rewards_t.cpu().numpy()

        # total = alpha1 * env + alpha2 * learned
        combined_rewards = (self.env_weight * rewards) + (self.learned_weight * learned_rewards)

        # Update cache for next step
        self._last_obs = obs
        
        return obs, combined_rewards, dones, infos


def train_ppo(env, total_timesteps, n_envs, tensorboard_log, save_path, device, 
              policy_type="MlpPolicy", learning_rate=3e-4, batch_size=64, 
              eval_env=None):
    
    print(f"Creating PPO agent on {device}...")
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=tensorboard_log,
                                log_path=tensorboard_log, eval_freq=1000,
                                n_eval_episodes=5, deterministic=True, render=False)

    rollout_steps_per_env = 2048 // max(n_envs, 1)

    model = PPO(
        policy_type, env, verbose=1, tensorboard_log=tensorboard_log,
        device=device, learning_rate=learning_rate, batch_size=batch_size,
        n_steps=rollout_steps_per_env, gamma=0.99
    )

    print(f"Starting PPO training ({total_timesteps:,} steps)...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    model.save(save_path)
    print(f"Model saved: {save_path}")
    return model

def main():
    parser = argparse.ArgumentParser(description='Train RL with Combined Rewards')
    parser.add_argument('--env-name', type=str, required=True)
    parser.add_argument('--env-type', type=str, default='gym')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--reward-path', type=str, default=None, 
                        help='Path to learned_reward_model.pt')
    parser.add_argument('--env-weight', type=float, default=1.0, 
                        help='Weight for original environment reward')
    parser.add_argument('--learned-weight', type=float, default=1.0, 
                        help='Weight for learned neural network reward')

    args = parser.parse_args()
    device = check_gpu()

    env_config = EnvConfig(
        env_type=args.env_type,
        env_name=args.env_name,
        seed=args.seed
    )

    print(f"Creating {args.n_envs} environments...")
    env = env_config.create_vec_env(n_envs=args.n_envs)
    eval_env = env_config.create_vec_env(n_envs=1) # Pure env for evaluation

    if args.reward_path:
        print(f"\nLOADING REWARD MODEL: {args.reward_path}")
        print(f"Weights -> Env: {args.env_weight} | Learned: {args.learned_weight}")
        
        reward_net = BasicRewardNet(
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        
        reward_net.load_state_dict(torch.load(args.reward_path, map_location=device))
        
        env = CombinedRewardVecWrapper(
            env, 
            reward_net, 
            env_weight=args.env_weight, 
            learned_weight=args.learned_weight,
            device=device
        )
        print("Environment successfully wrapped with Learned Reward!")
    else:
        print("\nNo reward model provided. Using standard environment reward only.")

    save_path = f"{args.algo}_{args.env_name}_combined"
    log_path = f"./rm_logs/{args.env_name}_combined"

    if args.algo == 'ppo':
        train_ppo(env, args.timesteps, args.n_envs, log_path, save_path, device, eval_env=eval_env)
    
    print("Done!")

if __name__ == '__main__':
    main()