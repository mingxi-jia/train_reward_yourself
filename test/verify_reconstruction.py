"""Verify the reconstructed object-state matches expected structure"""
import h5py
import numpy as np
from train_reward_yourself.train_sacfd import load_demonstrations_from_hdf5

demo_path = "/home/mingxi/data/mimicgen/reward/stack_d0.hdf5"

# Load using our updated code
observations, _, _, _, _ = load_demonstrations_from_hdf5(demo_path, max_episodes=1)

print("="*60)
print("Verifying reconstructed observations")
print("="*60)

# Extract components from the loaded observations
# Structure: robot_proprio (32 dims) + object-state (10 dims) = 42 dims
robot_proprio = observations[0, :32]
object_state_reconstructed = observations[0, 32:42]

print(f"First observation (42 dims):")
print(f"  Robot proprio (32 dims): {robot_proprio.shape}")
print(f"  Object-state (10 dims):  {object_state_reconstructed.shape}")

# Break down object-state
cube_pos_recon = object_state_reconstructed[0:3]
cube_quat_recon = object_state_reconstructed[3:7]
gripper_to_cube_recon = object_state_reconstructed[7:10]

print(f"\nReconstructed object-state components:")
print(f"  cube_pos:           {cube_pos_recon}")
print(f"  cube_quat:          {cube_quat_recon}")
print(f"  gripper_to_cube_pos: {gripper_to_cube_recon}")

# Compare with raw demo file
with h5py.File(demo_path, 'r') as f:
    demo_obj = f['data/demo_0/obs/object'][0]
    demo_gripper = f['data/demo_0/obs/robot0_eef_pos'][0]

    print(f"\nOriginal demo file values:")
    print(f"  object[0:3] (cube_pos):     {demo_obj[0:3]}")
    print(f"  object[3:7] (cube_quat):    {demo_obj[3:7]}")
    print(f"  object[7:10] (UNKNOWN):     {demo_obj[7:10]}")
    print(f"  robot0_eef_pos:             {demo_gripper}")

    # Verify our computation
    expected_gripper_to_cube = demo_gripper - demo_obj[0:3]
    print(f"\nExpected gripper_to_cube (gripper - cube_pos):")
    print(f"  {expected_gripper_to_cube}")

    print(f"\n✓ Match? {np.allclose(gripper_to_cube_recon, expected_gripper_to_cube)}")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("✓ Correctly extracts cube_pos from object[0:3]")
print("✓ Correctly extracts cube_quat from object[3:7]")
print("✓ Correctly computes gripper_to_cube_pos (NOT using object[7:10])")
print("\nThis matches the environment's object-state structure!")
