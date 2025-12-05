"""Quick script to check what observations the PandaPush environment provides"""
import gymnasium as gym
import panda_gym
import numpy as np

# Create the PandaPush environment
env = gym.make("PandaPush-v3")

print("="*60)
print("PandaPush-v3 Observation Space")
print("="*60)
print(f"Observation space type: {type(env.observation_space)}")
print(f"Observation space: {env.observation_space}")

# Check Dict space components
if hasattr(env.observation_space, 'spaces'):
    print("\nDict space components:")
    total_dim = 0
    for key, space in env.observation_space.spaces.items():
        print(f"  '{key}': {space}")
        if hasattr(space, 'shape'):
            dims = space.shape[0] if space.shape else 1
            total_dim += dims
            print(f"    - Shape: {space.shape}, Dimensions: {dims}")
    print(f"\nTotal dimensions across all components: {total_dim}")

# Reset and get an actual observation
print("\n" + "="*60)
print("Actual Observation from Environment")
print("="*60)
obs, info = env.reset(seed=42)

print(f"Observation type: {type(obs)}")
if isinstance(obs, dict):
    print("\nObservation components:")
    for key, value in obs.items():
        print(f"  '{key}':")
        print(f"    - Shape: {value.shape}")
        print(f"    - Dimensions: {value.shape[0] if value.shape else 1}")
        print(f"    - Sample values: {value[:5] if len(value) > 5 else value}")
        print(f"    - Min: {value.min():.4f}, Max: {value.max():.4f}")

# Check what the task goal is
print("\n" + "="*60)
print("Task Information")
print("="*60)
task = env.unwrapped.task
print(f"Task type: {type(task).__name__}")
print(f"Reward type: {task.reward_type}")
print(f"Distance threshold: {task.distance_threshold}")
print(f"Goal sample: {obs['desired_goal']}")

# Take a step and see what happens
print("\n" + "="*60)
print("Taking Random Action")
print("="*60)
action = env.action_space.sample()
print(f"Action shape: {action.shape}")
print(f"Action values: {action}")

obs, reward, terminated, truncated, info = env.step(action)
print(f"\nReward: {reward}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")
print(f"Info: {info}")

# Check observation breakdown for the actual state vector
print("\n" + "="*60)
print("Observation Component Breakdown")
print("="*60)
print("The 'observation' key likely contains:")
print("  - Robot EE position (3D): [x, y, z]")
print("  - Robot EE velocity (3D): [vx, vy, vz]")
print("  - Gripper state (2D): [gripper_pos, gripper_vel]")
print("  - Object position (3D): [x, y, z]")
print("  - Object orientation (4D): [qx, qy, qz, qw] (quaternion)")
print("  - Object velocity (3D): [vx, vy, vz]")
print(f"\nActual 'observation' dimensions: {obs['observation'].shape[0]}")

# Let's try to figure out what each component is
print("\n" + "="*60)
print("Attempting to Identify Observation Components")
print("="*60)

obs_vec = obs['observation']
print(f"Total observation vector: {obs_vec.shape}")
print(f"\nBreakdown (best guess):")

# Typical structure for Panda environments
idx = 0
# EE position
ee_pos_end = idx + 3
print(f"  [{idx:2d}:{ee_pos_end:2d}] EE Position: {obs_vec[idx:ee_pos_end]}")
idx = ee_pos_end

# EE velocity
ee_vel_end = idx + 3
print(f"  [{idx:2d}:{ee_vel_end:2d}] EE Velocity: {obs_vec[idx:ee_vel_end]}")
idx = ee_vel_end

# Gripper
gripper_end = idx + 2
print(f"  [{idx:2d}:{gripper_end:2d}] Gripper: {obs_vec[idx:gripper_end]}")
idx = gripper_end

# Object position
obj_pos_end = idx + 3
print(f"  [{idx:2d}:{obj_pos_end:2d}] Object Position: {obs_vec[idx:obj_pos_end]}")
print(f"                  (achieved_goal: {obs['achieved_goal']})")
idx = obj_pos_end

# Object orientation (quaternion)
obj_rot_end = idx + 4
print(f"  [{idx:2d}:{obj_rot_end:2d}] Object Rotation (quat): {obs_vec[idx:obj_rot_end]}")
idx = obj_rot_end

# Object velocity
if idx < len(obs_vec):
    obj_vel_end = idx + 3
    print(f"  [{idx:2d}:{obj_vel_end:2d}] Object Velocity: {obs_vec[idx:obj_vel_end]}")
    idx = obj_vel_end

# Anything remaining
if idx < len(obs_vec):
    print(f"  [{idx:2d}:{len(obs_vec):2d}] Remaining: {obs_vec[idx:]}")

env.close()
print("\n" + "="*60)
