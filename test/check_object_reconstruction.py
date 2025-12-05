"""Check if we can reconstruct object-state from demo observations"""
import h5py
import numpy as np

demo_path = "/home/mingxi/data/mimicgen/reward/stack_d0.hdf5"

with h5py.File(demo_path, 'r') as f:
    demo = f['data/demo_0/obs']

    # Get first timestep
    cube_pos = demo['robot0_eef_pos'][0] if 'robot0_eef_pos' in demo else None
    obj = demo['object'][0]

    print("="*60)
    print("Can we reconstruct object-state from demo observations?")
    print("="*60)

    # Check if demo has cube_pos, cube_quat separately
    if 'cube_pos' in demo:
        print("✓ Demo has 'cube_pos'")
        cube_pos = demo['cube_pos'][0]
        print(f"  Values: {cube_pos}")
    else:
        print("✗ Demo does NOT have 'cube_pos' - might be in 'object' dims 0-2")
        print(f"  object[0:3]: {obj[0:3]}")

    if 'cube_quat' in demo:
        print("✓ Demo has 'cube_quat'")
        cube_quat = demo['cube_quat'][0]
        print(f"  Values: {cube_quat}")
    else:
        print("✗ Demo does NOT have 'cube_quat' - might be in 'object' dims 3-6")
        print(f"  object[3:7]: {obj[3:7]}")

    if 'gripper_to_cube_pos' in demo:
        print("✓ Demo has 'gripper_to_cube_pos'")
        gripper_to_cube = demo['gripper_to_cube_pos'][0]
        print(f"  Values: {gripper_to_cube}")
    else:
        print("✗ Demo does NOT have 'gripper_to_cube_pos'")
        # Try to compute it from gripper and cube positions
        if 'robot0_eef_pos' in demo:
            gripper_pos = demo['robot0_eef_pos'][0]
            cube_pos_est = obj[0:3]
            gripper_to_cube_computed = gripper_pos - cube_pos_est
            print(f"  Can compute from robot0_eef_pos - cube_pos:")
            print(f"    robot0_eef_pos: {gripper_pos}")
            print(f"    cube_pos (from object[0:3]): {cube_pos_est}")
            print(f"    gripper_to_cube (computed): {gripper_to_cube_computed}")
            print(f"    object[7:10] for comparison: {obj[7:10]}")

    print("\n" + "="*60)
    print("Recommendation")
    print("="*60)

    available_keys = list(demo.keys())
    print(f"Available observation keys in demo: {available_keys}")

    # Check if we have the components to reconstruct object-state
    has_components = ('robot0_eef_pos' in demo and
                     len(obj) >= 7)  # at least pos + quat

    if has_components:
        print("\n✓ We CAN reconstruct object-state properly:")
        print("  1. Extract cube_pos from object[0:3]")
        print("  2. Extract cube_quat from object[3:7]")
        print("  3. Compute gripper_to_cube_pos = robot0_eef_pos - cube_pos")
    else:
        print("\n⚠ Using first 10 dims of 'object' is risky - might not match object-state")
