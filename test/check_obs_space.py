"""Quick script to check what observations the environment provides"""
from train_reward_yourself.utils import make_robosuite_env

# Create the same environment as in training
env = make_robosuite_env(control_type="OSC_POSE", env_name="Coffee_D0")

print("="*60)
print("Environment Observation Space")
print("="*60)
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Observation space: {env.observation_space}")

# Reset and get an observation
obs, info = env.reset()
print(f"\nActual observation shape: {obs.shape}")

# Check the underlying robosuite env to see what obs keys it has
print("\n" + "="*60)
print("Raw Robosuite Observation Keys")
print("="*60)
raw_obs = env.env._get_observations()
total_dim = 0
for key in sorted(raw_obs.keys()):
    if hasattr(raw_obs[key], 'shape'):
        dim = raw_obs[key].shape[0] if len(raw_obs[key].shape) > 0 else 1
        total_dim += dim
        print(f"  {key}: shape={raw_obs[key].shape}, dims={dim}")
    else:
        print(f"  {key}: {type(raw_obs[key])}")

print(f"\nTotal dimensions in raw obs: {total_dim}")

# Check what the GymWrapper actually uses
print("\n" + "="*60)
print("GymWrapper's Observation Keys (what RL agent sees)")
print("="*60)
if hasattr(env.env, 'keys'):
    gym_keys = env.env.keys
    print(f"Keys used by GymWrapper: {gym_keys}")
    gym_total = 0
    for key in gym_keys:
        if key in raw_obs:
            dim = raw_obs[key].shape[0] if len(raw_obs[key].shape) > 0 else 1
            gym_total += dim
            print(f"  {key}: {raw_obs[key].shape} = {dim} dims")
    print(f"\nTotal dimensions used by GymWrapper: {gym_total}")
else:
    print("GymWrapper keys not accessible via .keys attribute")
    print("Checking modality_dims...")
    if hasattr(env.env, 'modality_dims'):
        print(f"Modality dims: {env.env.modality_dims}")

env.close()
