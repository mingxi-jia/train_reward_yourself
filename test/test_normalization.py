"""Test the normalization implementation"""
from train_reward_yourself.train_sacfd import load_demonstrations_from_hdf5
import numpy as np

demo_path = '/home/mingxi/data/mimicgen/reward/stack_d0.hdf5'

print("="*60)
print("Testing Normalization Implementation")
print("="*60)

# Load demos with normalization stats
print("\n1. Loading demonstrations and computing normalization stats...")
obs, actions, rewards, next_obs, dones, (obs_mean, obs_std) = load_demonstrations_from_hdf5(
    demo_path, max_episodes=5, compute_norm_stats=True
)

print(f"\n2. Verifying normalization statistics:")
print(f"   Obs mean shape: {obs_mean.shape}")
print(f"   Obs std shape: {obs_std.shape}")
print(f"   Mean range: [{obs_mean.min():.4f}, {obs_mean.max():.4f}]")
print(f"   Std range: [{obs_std.min():.6f}, {obs_std.max():.6f}]")

# Test normalization
print(f"\n3. Testing normalization on observations:")
normalized_obs = (obs - obs_mean) / obs_std

print(f"   Original obs:")
print(f"     Mean: {obs.mean():.6f}")
print(f"     Std: {obs.std():.6f}")
print(f"     Scale variation: {obs.std(axis=0).max() / obs.std(axis=0).min():.2f}x")

print(f"\n   Normalized obs:")
print(f"     Mean: {normalized_obs.mean():.6f} (should be ~0)")
print(f"     Std: {normalized_obs.std():.6f} (should be ~1)")
print(f"     Scale variation: {normalized_obs.std(axis=0).max() / normalized_obs.std(axis=0).min():.2f}x (should be ~1)")

if abs(normalized_obs.mean()) < 0.01 and abs(normalized_obs.std() - 1.0) < 0.1:
    print(f"\n✅ SUCCESS! Normalization is working correctly")
    print(f"   Scale variation reduced from 769x to {normalized_obs.std(axis=0).max() / normalized_obs.std(axis=0).min():.2f}x")
else:
    print(f"\n❌ FAILED! Normalization not working as expected")

print("\n" + "="*60)
print("Test Complete")
print("="*60)
