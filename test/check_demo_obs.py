"""Check what observations are in the demonstration file"""
import h5py
import sys

if len(sys.argv) < 2:
    print("Usage: python check_demo_obs.py <path_to_demo.hdf5>")
    print("\nExample: python check_demo_obs.py path/to/demo.hdf5")
    sys.exit(1)

demo_path = sys.argv[1]

with h5py.File(demo_path, 'r') as f:
    demo_keys = [key for key in f['data'].keys() if key.startswith('demo_')]
    print(f"Found {len(demo_keys)} demonstrations\n")

    # Look at first demo
    demo = f['data'][demo_keys[0]]

    print("="*60)
    print(f"Observation keys in {demo_keys[0]}")
    print("="*60)

    obs = demo['obs']
    total_dims = 0
    for key in sorted(obs.keys()):
        shape = obs[key].shape
        dims = shape[1] if len(shape) > 1 else 1
        total_dims += dims
        print(f"  {key}: {shape} -> {dims} dims")

    print(f"\nTotal observation dimensions: {total_dims}")

    # Check actions
    actions = demo['actions']
    print(f"\nActions shape: {actions.shape}")
