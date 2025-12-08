import h5py
import numpy as np
import gymnasium as gym
import torch
import argparse
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env

from imitation.data import types
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet

def make_custom_vec_env(env_name, n_envs, rng):
    """
    Creates a vectorized environment using Stable-Baselines3 directly.
    """
    return sb3_make_vec_env(
        env_name,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
    )

def load_demos_from_hdf5(file_path: str) -> list[types.Trajectory]:
    print(f"Loading demonstrations from {file_path}...")
    trajectories = []
    
    with h5py.File(file_path, 'r') as f:
        data_grp = f['data']
        demo_keys = [k for k in data_grp.keys() if k.startswith('demo_')]
        
        for k in tqdm(demo_keys, desc="Parsing HDF5"):
            demo = data_grp[k]
            obs = demo['obs']['state'][:]       
            acts = demo['actions'][:]           
            
            # Data alignment: (s, a, s')
            valid_obs = obs                     
            valid_acts = acts[:-1]              
            
            infos = [{}] * len(valid_acts)
            
            # Create Trajectory
            t = types.Trajectory(
                obs=valid_obs,
                acts=valid_acts,
                infos=np.array(infos),
                terminal=True
            )
            trajectories.append(t)
            
    print(f"Loaded {len(trajectories)} valid trajectories.")
    return trajectories

def train_reward_model(
    demos_path: str,
    env_name: str,
    total_timesteps: int = 500_000,
    save_path: str = "learned_reward_model.pt",
    seed: int = 42
):
    rng = np.random.default_rng(seed)

    trajectories = load_demos_from_hdf5(demos_path)
    
    venv = make_custom_vec_env(env_name, n_envs=8, rng=rng)
    
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
    )

    gen_algo = PPO("MlpPolicy", venv, verbose=1)

    airl_trainer = AIRL(
        demonstrations=trajectories,
        demo_batch_size=1024,
        gen_algo=gen_algo,
        venv=venv,
        reward_net=reward_net,
        allow_variable_horizon=True
    )

    print("\nStarting AIRL training...")
    airl_trainer.train(total_timesteps=total_timesteps)

    print(f"\nSaving reward model to {save_path}...")
    torch.save(reward_net.state_dict(), save_path)
    
    return reward_net

def test_reward_model(reward_net, env_name):
    """
    Visual sanity check: Compare learned reward vs environment reward.
    Handles both Discrete (one-hot) and Continuous action spaces.
    """
    env = gym.make(env_name)
    obs, _ = env.reset()
    
    print("\nRunning sanity check on learned reward...")
    print(f"{'Step':<6} | {'Env Reward':<12} | {'Learned Reward':<15}")
    print("-" * 40)
    
    reward_net.eval()
    
    # Check if we need to one-hot encode discrete actions
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    
    with torch.no_grad():
        for i in range(10):
            action = env.action_space.sample()
            next_obs, env_reward, term, trunc, _ = env.step(action)
            
            # (Shape: 1, Obs_Dim)
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            dones_t = torch.as_tensor([False], dtype=torch.bool)
            
            # (Shape: 1, Act_Dim)
            if is_discrete:
                # Create a zero tensor of shape (1, num_actions)
                act_t = torch.zeros(1, env.action_space.n)
                # Set the active index to 1.0
                act_t[0, action] = 1.0
            else:
                # For continuous, just add batch dimension
                act_t = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
            
            out = reward_net(obs_t, act_t, next_obs_t, dones_t)
            
            if isinstance(out, tuple):
                learned_reward = out[0].item()
            else:
                learned_reward = out.item()
            
            print(f"{i:<6} | {env_reward:<12.4f} | {learned_reward:<15.4f}")
            
            obs = next_obs
            if term or trunc:
                break

def load_reward_model(model_path, env_name, device="cpu"):
    """
    Reconstructs the reward network structure and loads weights.
    """
    print(f"Creating environment {env_name} to infer network structure...")

    venv = sb3_make_vec_env(env_name, n_envs=1, vec_env_cls=DummyVecEnv)
    
    print(f"Loading weights from {model_path}...")
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
    )
    
    reward_net.load_state_dict(torch.load(model_path, map_location=device))
    reward_net.to(device)
    reward_net.eval() # Set to inference mode
    
    print("Model loaded successfully!")
    return reward_net, venv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--demos', type=str, required=True, help='Path to .hdf5 demo file')
    parser.add_argument('--env', type=str, required=True, help='Gym environment name')
    parser.add_argument('--steps', type=int, default=200_000, help='Training timesteps')
    args = parser.parse_args()

    reward_net = train_reward_model(args.demos, args.env, args.steps)
    # reward_net, _ = load_reward_model("/home/atulep/Research/cs1952_final/train_reward_yourself/learned_reward_model.pt", args.env)
    test_reward_model(reward_net, args.env)