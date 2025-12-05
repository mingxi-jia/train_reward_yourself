"""Check which observation keys GymWrapper actually uses"""
from train_reward_yourself.utils import make_robosuite_env
import numpy as np

# Create the same environment as in training
env = make_robosuite_env(control_type="OSC_POSE", env_name="Lift")

obs, info = env.reset()
print(f"Flattened observation shape: {obs.shape}")
print(f"Expected: {env.observation_space.shape}\n")

# Get raw observations from robosuite
raw_obs = env.env._get_observations()

# Try to reverse-engineer which keys are used by checking dimensions
print("="*60)
print("Attempting to identify which keys GymWrapper uses...")
print("="*60)

# The GymWrapper source code typically uses keys in a specific order
# Let's check if we can access the keys attribute
if hasattr(env.env, '_flatten_obs'):
    print("GymWrapper has _flatten_obs method")

# Check the source to see what gets flattened
import inspect
print("\nGymWrapper class methods:")
for name, method in inspect.getmembers(env.env.__class__, predicate=inspect.isfunction):
    if 'obs' in name.lower() or 'flatten' in name.lower():
        print(f"  - {name}")

# Manual reconstruction attempt
print("\n" + "="*60)
print("Manually testing observation key combinations...")
print("="*60)

# Common patterns for GymWrapper with use_object_obs=True:
# Typically includes robot proprioception + object observations
test_combinations = [
    # Robot proprio + object
    ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
    # With joint info
    ['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel',
     'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
    # Check if robot0_proprio-state is used
    ['robot0_proprio-state', 'object-state'],
    # Try individual components
    ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel',
     'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'object-state'],
]

for i, keys in enumerate(test_combinations):
    total_dims = 0
    valid = True
    for key in keys:
        if key in raw_obs:
            if hasattr(raw_obs[key], 'shape'):
                dims = raw_obs[key].shape[0] if len(raw_obs[key].shape) > 0 else 1
                total_dims += dims
            else:
                valid = False
                break
        else:
            valid = False
            break

    if valid and total_dims == 42:
        print(f"\n✓ MATCH FOUND - Combination {i+1}:")
        for key in keys:
            dims = raw_obs[key].shape[0] if len(raw_obs[key].shape) > 0 else 1
            print(f"    {key}: {dims} dims")
        print(f"  Total: {total_dims} dims")

env.close()
