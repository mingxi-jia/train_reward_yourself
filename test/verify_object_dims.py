"""Verify if first 10 dims of demo 'object' match environment 'object-state'"""
import h5py
import numpy as np
from train_reward_yourself.utils import make_robosuite_env

# Load demo file
demo_path = "/home/mingxi/data/mimicgen/reward/stack_d0.hdf5"
with h5py.File(demo_path, 'r') as f:
    demo_object = f['data/demo_0/obs/object'][0]  # First timestep
    print("="*60)
    print("Demo file 'object' observation (23 dims)")
    print("="*60)
    print(f"Shape: {demo_object.shape}")
    print(f"Values: {demo_object}")
    print(f"\nFirst 10 dims: {demo_object[:10]}")

# Create environment and get object-state
print("\n" + "="*60)
print("Environment 'object-state' observation (10 dims)")
print("="*60)
env = make_robosuite_env(control_type="OSC_POSE", env_name="Lift")
obs, info = env.reset()
raw_obs = env.env._get_observations()
object_state = raw_obs['object-state']
print(f"Shape: {object_state.shape}")
print(f"Values: {object_state}")

# Compare structure
print("\n" + "="*60)
print("Comparison")
print("="*60)
print(f"Demo 'object' first 10 dims: {demo_object[:10]}")
print(f"Env 'object-state' 10 dims:  {object_state}")
print("\nNote: Values won't match exactly (different states), but structure should be similar")

# Check what components make up object-state
print("\n" + "="*60)
print("Analyzing 'object-state' components")
print("="*60)
# object-state typically includes: cube_pos (3) + cube_quat (4) + gripper_to_cube (3) = 10
cube_keys = [k for k in raw_obs.keys() if 'cube' in k]
print(f"Keys with 'cube': {cube_keys}")
for key in cube_keys:
    print(f"  {key}: shape={raw_obs[key].shape}, values={raw_obs[key]}")

# Try to reconstruct object-state
if 'cube_pos' in raw_obs and 'cube_quat' in raw_obs and 'gripper_to_cube_pos' in raw_obs:
    reconstructed = np.concatenate([
        raw_obs['cube_pos'],
        raw_obs['cube_quat'],
        raw_obs['gripper_to_cube_pos']
    ])
    print(f"\nReconstructed from cube_pos + cube_quat + gripper_to_cube_pos:")
    print(f"  Shape: {reconstructed.shape}")
    print(f"  Values: {reconstructed}")
    print(f"\nDoes reconstructed match object-state? {np.allclose(reconstructed, object_state)}")

# Check what's in demo 'object' (23 dims)
print("\n" + "="*60)
print("Analyzing demo 'object' components (23 dims)")
print("="*60)
print("Likely structure based on common robosuite patterns:")
print("  - Object pose/state: ~10-14 dims")
print("  - Additional features: ~9-13 dims")
print("\nPossible breakdown:")
print("  cube_pos (3) + cube_quat (4) + gripper_to_cube (3) = 10 dims")
print("  + other object features (velocities, etc.) = +13 dims = 23 total")

env.close()
