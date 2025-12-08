import argparse
import h5py
import numpy as np
import gymnasium as gym
import torch
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

def load_first_demo(demo_path):
    """Extracts the first trajectory from your HDF5 file."""
    with h5py.File(demo_path, 'r') as f:
        demo_grp = f['data/demo_0']
        obs = demo_grp['obs']['state'][:]
        actions = demo_grp['actions'][:]
        # AIRL requires (s, a, s')
        # We assume obs has N+1 or N length relative to actions.
        # Usually obs is N+1, actions N.
        if len(obs) == len(actions) + 1:
            next_obs = obs[1:]
            obs = obs[:-1]
        else:
            # If lengths match, we lose the last transition
            next_obs = obs[1:]
            obs = obs[:-1]
            actions = actions[:-1]
            
    return obs, actions, next_obs

def score_trajectory(reward_net, obs, actions, next_obs, device="cpu"):
    """Feeds a full trajectory through the reward network."""
    reward_net.eval()
    obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
    
    # Handle discrete actions (one-hot)
    # Assuming LunarLander is 4 actions
    # If using continuous, remove the one-hot logic
    if actions.ndim == 1: # Discrete array
        act_t = torch.zeros(len(actions), 4).to(device)
        act_t[np.arange(len(actions)), actions.astype(int)] = 1.0
    else:
        act_t = torch.as_tensor(actions, dtype=torch.float32).to(device)
        
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32).to(device)
    dones_t = torch.zeros(len(obs), dtype=torch.bool).to(device)
    
    with torch.no_grad():
        rewards = reward_net(obs_t, act_t, next_obs_t, dones_t)
        if isinstance(rewards, tuple): rewards = rewards[0]
        
    return rewards.cpu().numpy().sum(), rewards.cpu().numpy().mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demos', type=str, required=True, help='Path to demos_lunarlander.hdf5')
    parser.add_argument('--model', type=str, default='learned_reward_model.pt')
    parser.add_argument('--env', type=str, default='LunarLander-v3')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    venv = sb3_make_vec_env(args.env, n_envs=1, vec_env_cls=DummyVecEnv)
    reward_net = BasicRewardNet(venv.observation_space, venv.action_space)
    reward_net.load_state_dict(torch.load(args.model, map_location=device))
    reward_net.to(device)

    print("Loading expert demo...")
    exp_obs, exp_acts, exp_next = load_first_demo(args.demos)
    exp_total, exp_mean = score_trajectory(reward_net, exp_obs, exp_acts, exp_next, device)

    print("Running random agent...")
    env = gym.make(args.env)
    obs, _ = env.reset()
    rand_obs, rand_acts, rand_next = [], [], []
    
    # Run for same length as expert to be fair
    for _ in range(len(exp_acts)):
        action = env.action_space.sample()
        next_obs, _, term, trunc, _ = env.step(action)
        
        rand_obs.append(obs)
        rand_acts.append(action)
        rand_next.append(next_obs)
        
        obs = next_obs
        if term or trunc: obs, _ = env.reset()

    rand_total, rand_mean = score_trajectory(
        reward_net, 
        np.array(rand_obs), 
        np.array(rand_acts), 
        np.array(rand_next), 
        device
    )

    print("\n" + "="*40)
    print(f"RESULTS: {args.env}")
    print("="*40)
    print(f"{'Metric':<15} | {'Expert':<10} | {'Random':<10}")
    print("-" * 40)
    print(f"{'Mean Reward':<15} | {exp_mean:<10.4f} | {rand_mean:<10.4f}")
    print(f"{'Total Reward':<15} | {exp_total:<10.1f} | {rand_total:<10.1f}")
    print("="*40)
    
    if exp_mean > rand_mean:
        print("SUCCESS: The model correctly prefers the Expert.")
    else:
        print("FAILURE: The model cannot distinguish Expert from Random.")

if __name__ == "__main__":
    main()