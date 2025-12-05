#!/usr/bin/env python3
"""
Visualize LunarLander HDF5 demonstrations using rerun.io.
Shows trajectory, orientation, state variables, actions, and rewards in an interactive viewer.
"""

import argparse
import h5py
import numpy as np
import rerun as rr
from pathlib import Path
import sys


def visualize_episode(f, episode_key, timeline="frame"):
    """Visualize a single episode from the HDF5 file.

    Args:
        f: HDF5 file handle
        episode_key: Key of the episode (e.g., "demo_0")
        timeline: Timeline name for rerun
    """
    ep_grp = f[f"data/{episode_key}"]

    # Get episode data
    states = np.array(ep_grp["obs/state"])
    actions = np.array(ep_grp["actions"])
    rewards = np.array(ep_grp["rewards"])
    dones = np.array(ep_grp["dones"])

    # Check if images are available
    has_images = "images" in ep_grp["obs"]
    images = None
    if has_images:
        images = np.array(ep_grp["obs/images"])
        print(f"    Images shape: {images.shape}")

    num_samples = len(states)
    total_reward = np.sum(rewards)

    print(f"[*] {episode_key}: {num_samples} samples, Total Reward: {total_reward:.2f}")

    # LunarLander state: [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]
    x_pos = states[:, 0]
    y_pos = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]
    angle = states[:, 4]
    angular_vel = states[:, 5]
    left_leg = states[:, 6]
    right_leg = states[:, 7]

    # Action names for LunarLander-v3
    action_names = ['Do Nothing', 'Fire Left', 'Fire Main', 'Fire Right']

    # Log static world elements
    # Ground line
    ground_y = 0.0
    ground_points = np.array([
        [x_pos.min() - 1.0, ground_y, 0],
        [x_pos.max() + 1.0, ground_y, 0],
    ])
    rr.log(
        f"{episode_key}/world/ground",
        rr.LineStrips3D([ground_points], colors=[139, 69, 19], radii=0.02),
        static=True,
    )

    # Landing pad (center at x=0)
    pad_width = 0.5
    pad_points = np.array([
        [-pad_width/2, ground_y, 0],
        [pad_width/2, ground_y, 0],
    ])
    rr.log(
        f"{episode_key}/world/landing_pad",
        rr.LineStrips3D([pad_points], colors=[0, 255, 0], radii=0.05),
        static=True,
    )

    # World origin axes
    axis_length = 0.3
    rr.log(
        f"{episode_key}/world/origin",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0]],
            vectors=[
                [axis_length, 0, 0],  # X-axis (red)
                [0, axis_length, 0],  # Y-axis (green)
            ],
            colors=[[255, 0, 0], [0, 255, 0]],
            radii=0.01,
        ),
        static=True,
    )

    # Visualize each frame
    cumulative_reward = 0.0
    trajectory_positions = []

    for i in range(num_samples):
        rr.set_time(timeline, sequence=i)

        # Current position and orientation
        x, y = x_pos[i], y_pos[i]
        ang = angle[i]
        trajectory_positions.append([x, y, 0])

        # Log lander position as a 2D transform in 3D space
        # Create rotation matrix for z-axis rotation (2D rotation in 3D)
        cos_a = np.cos(ang)
        sin_a = np.sin(ang)
        R_mat = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ])

        rr.log(
            f"{episode_key}/lander/pose",
            rr.Transform3D(
                translation=[x, y, 0],
                mat3x3=R_mat,
            )
        )

        # Log lander body as a simple box
        # LunarLander is roughly 0.3 units wide and 0.2 units tall
        body_vertices = np.array([
            [-0.15, -0.1, 0],  # bottom left
            [0.15, -0.1, 0],   # bottom right
            [0.15, 0.1, 0],    # top right
            [-0.15, 0.1, 0],   # top left
        ])
        rr.log(
            f"{episode_key}/lander/pose/body",
            rr.LineStrips3D([body_vertices], colors=[200, 200, 200], radii=0.02)
        )

        # Log lander as a point for trajectory
        rr.log(
            f"{episode_key}/lander/pose/center",
            rr.Points3D([[0, 0, 0]], colors=[255, 165, 0], radii=0.05)
        )

        # Log legs contact indicators
        if left_leg[i] > 0.5:
            rr.log(
                f"{episode_key}/lander/pose/left_leg",
                rr.Points3D([[-0.1, -0.15, 0]], colors=[0, 255, 0], radii=0.03)
            )
        if right_leg[i] > 0.5:
            rr.log(
                f"{episode_key}/lander/pose/right_leg",
                rr.Points3D([[0.1, -0.15, 0]], colors=[0, 255, 0], radii=0.03)
            )

        # Log velocity vector
        vel_scale = 0.5
        rr.log(
            f"{episode_key}/lander/velocity",
            rr.Arrows3D(
                origins=[[x, y, 0]],
                vectors=[[vx[i] * vel_scale, vy[i] * vel_scale, 0]],
                colors=[0, 150, 255],
                radii=0.01,
            )
        )

        # Log full trajectory up to current point
        if i > 0:
            rr.log(
                f"{episode_key}/trajectory",
                rr.LineStrips3D([trajectory_positions], colors=[0, 200, 255], radii=0.01)
            )

        # Log state scalars
        rr.log(f"{episode_key}/state/position/x", rr.Scalars(np.array([x])))
        rr.log(f"{episode_key}/state/position/y", rr.Scalars(np.array([y])))
        rr.log(f"{episode_key}/state/velocity/vx", rr.Scalars(np.array([vx[i]])))
        rr.log(f"{episode_key}/state/velocity/vy", rr.Scalars(np.array([vy[i]])))
        rr.log(f"{episode_key}/state/velocity/speed", rr.Scalars(np.array([np.sqrt(vx[i]**2 + vy[i]**2)])))
        rr.log(f"{episode_key}/state/orientation/angle_deg", rr.Scalars(np.array([np.rad2deg(ang)])))
        rr.log(f"{episode_key}/state/orientation/angular_vel", rr.Scalars(np.array([angular_vel[i]])))
        rr.log(f"{episode_key}/state/contact/left_leg", rr.Scalars(np.array([left_leg[i]])))
        rr.log(f"{episode_key}/state/contact/right_leg", rr.Scalars(np.array([right_leg[i]])))

        # Log action
        action = actions[i]
        action_one_hot = np.zeros(4)
        action_one_hot[action] = 1.0
        rr.log(f"{episode_key}/action/type", rr.Scalars(np.array([action])))

        # Log action name as text
        rr.log(
            f"{episode_key}/action/name",
            rr.TextLog(action_names[action], level="INFO")
        )

        # Log individual action indicators
        for j, name in enumerate(action_names):
            safe_name = name.replace(' ', '_')
            rr.log(f"{episode_key}/action/one_hot/{safe_name}", rr.Scalars(np.array([action_one_hot[j]])))

        # Log reward
        rr.log(f"{episode_key}/reward/instant", rr.Scalars(np.array([rewards[i]])))
        cumulative_reward += rewards[i]
        rr.log(f"{episode_key}/reward/cumulative", rr.Scalars(np.array([cumulative_reward])))

        # Log done flag
        rr.log(f"{episode_key}/done", rr.Scalars(np.array([float(dones[i])])))

        # Log rendered image if available
        if has_images and images is not None and i < len(images):
            rr.log(
                f"{episode_key}/camera/image",
                rr.Image(images[i])
            )

    print(f"    Logged {num_samples} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LunarLander HDF5 demonstrations with rerun.io"
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default="demos_lunarlander_v3.hdf5",
        help="Path to the HDF5 file (default: demos_lunarlander_v3.hdf5)"
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Episodes to visualize (e.g., '0,1,2' or '0-5'). Default: all"
    )
    parser.add_argument(
        "--app-id",
        type=str,
        default="lunarlander_viz",
        help="Rerun application ID (default: lunarlander_viz)"
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Connect to remote rerun viewer instead of spawning"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save recording to .rrd file"
    )

    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        print(f"Error: HDF5 file not found: {hdf5_path}")
        sys.exit(1)

    # Initialize rerun
    if args.save:
        print(f"Recording to file: {args.save}")
        rr.init(args.app_id, recording_id=args.app_id)
        rr.save(args.save)
    elif args.connect:
        print("Connecting to remote rerun viewer...")
        rr.init(args.app_id, spawn=False)
        rr.connect()
    else:
        print("Spawning rerun viewer...")
        rr.init(args.app_id, spawn=True)

    # Set up blueprint for better default layout
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Horizontal(
            rr.blueprint.Vertical(
                rr.blueprint.Spatial3DView(name="3D View", origin="/"),
                rr.blueprint.Spatial2DView(name="Camera", origin="/"),
                row_shares=[2, 1],
            ),
            rr.blueprint.Vertical(
                rr.blueprint.TimeSeriesView(name="State", origin="/"),
                rr.blueprint.TimeSeriesView(name="Actions & Rewards", origin="/"),
                row_shares=[1, 1],
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    # Open HDF5 file
    print(f"\nOpening HDF5 file: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        # Get all episode keys
        all_episodes = sorted([k for k in f["data"].keys() if k.startswith("demo_")])

        # Parse episode selection
        if args.episodes is None:
            episodes_to_viz = all_episodes
        else:
            episodes_to_viz = []
            for part in args.episodes.split(","):
                if "-" in part:
                    # Range
                    start, end = part.split("-")
                    for i in range(int(start), int(end) + 1):
                        episodes_to_viz.append(f"demo_{i}")
                else:
                    # Single episode
                    episodes_to_viz.append(f"demo_{part}")

        print(f"\nTotal episodes available: {len(all_episodes)} (demo_0 to demo_{len(all_episodes)-1})")
        print(f"Visualizing: {len(episodes_to_viz)} episode(s)")
        if args.episodes is not None:
            print(f"Selected episodes: {', '.join([ep.replace('demo_', '') for ep in episodes_to_viz])}")
        print()

        # Visualize each episode
        for ep_key in episodes_to_viz:
            if ep_key not in all_episodes:
                print(f"Warning: Episode {ep_key} not found, skipping")
                continue
            visualize_episode(f, ep_key)

        print()
        print("="*60)
        print("Visualization complete!")
        print("="*60)
        print("\nRerun viewer controls:")
        print("  - Use the timeline slider to scrub through frames")
        print("  - Click episodes in the tree view to show/hide them")
        print("  - Select different entities to focus on specific data")
        print("  - Adjust the view in the 3D viewer")
        print("  - Inspect time series plots for state and reward data")

        print("\nTip: To visualize specific episodes, use:")
        print("  python visualize_lunarlander_rerun.py --episodes 0,1,2")
        print("  python visualize_lunarlander_rerun.py --episodes 0-5")

        if args.save:
            print(f"\nRecording saved to: {args.save}")
            print(f"You can view it later with: rerun {args.save}")


if __name__ == "__main__":
    main()
