"""Test if demo loading now produces correct 42-dim observations"""
from train_reward_yourself.train_sacfd import load_demonstrations_from_hdf5

demo_path = "/home/mingxi/data/mimicgen/reward/stack_d0.hdf5"

print("Testing demo loading with updated code...")
observations, actions, rewards, next_observations, dones = load_demonstrations_from_hdf5(
    demo_path, max_episodes=1
)

print(f"\n✓ Loaded successfully!")
print(f"Observation shape: {observations.shape}")
print(f"Expected shape: (num_transitions, 42)")

if observations.shape[1] == 42:
    print(f"\n✅ SUCCESS! Observation dimensions match environment (42 dims)")
else:
    print(f"\n❌ MISMATCH! Got {observations.shape[1]} dims, expected 42 dims")
