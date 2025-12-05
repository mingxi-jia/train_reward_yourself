#!/usr/bin/env python3
"""
Visualize LunarLander HDF5 demonstrations.
Shows trajectory, orientation, state variables, actions, and rewards over time.
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.animation import FuncAnimation
from pathlib import Path
import sys


def plot_episode_overview(f, episode_key, show_images=False):
    """Create a comprehensive overview plot of a single episode.

    Args:
        f: HDF5 file handle
        episode_key: Key of the episode (e.g., "demo_0")
        show_images: Whether to display images if available
    """
    ep_grp = f[f"data/{episode_key}"]

    # Load data
    states = np.array(ep_grp["obs/state"])
    actions = np.array(ep_grp["actions"])
    rewards = np.array(ep_grp["rewards"])
    dones = np.array(ep_grp["dones"])

    # Check if images are available
    has_images = "images" in ep_grp["obs"]
    images = None
    if has_images and show_images:
        images = np.array(ep_grp["obs/images"])

    num_steps = len(states)
    timesteps = np.arange(num_steps)

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

    # Create figure with subplots
    if images is not None:
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'{episode_key} - Total Reward: {np.sum(rewards):.2f} (with images)', fontsize=16)
    else:
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'{episode_key} - Total Reward: {np.sum(rewards):.2f}', fontsize=16)

    # 1. Trajectory plot (top left)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(x_pos, y_pos, 'b-', linewidth=2, alpha=0.6)
    ax1.scatter(x_pos[0], y_pos[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(x_pos[-1], y_pos[-1], c='red', s=100, marker='x', label='End', zorder=5)
    ax1.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Ground')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. Position over time (top middle)
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(timesteps, x_pos, label='X', linewidth=2)
    ax2.plot(timesteps, y_pos, label='Y', linewidth=2)
    ax2.axhline(y=0, color='brown', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Position')
    ax2.set_title('Position over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Velocity over time (top right)
    ax3 = plt.subplot(3, 3, 3)
    speed = np.sqrt(vx**2 + vy**2)
    ax3.plot(timesteps, vx, label='Vx', linewidth=2)
    ax3.plot(timesteps, vy, label='Vy', linewidth=2)
    ax3.plot(timesteps, speed, label='Speed', linewidth=2, linestyle='--')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Velocity over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Angle and angular velocity (middle left)
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(timesteps, np.rad2deg(angle), label='Angle (deg)', linewidth=2)
    ax4.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Upright')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Angle (degrees)')
    ax4.set_title('Orientation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax4_twin = ax4.twinx()
    ax4_twin.plot(timesteps, angular_vel, label='Angular Vel', color='orange', linewidth=2, alpha=0.7)
    ax4_twin.set_ylabel('Angular Velocity (rad/s)', color='orange')
    ax4_twin.tick_params(axis='y', labelcolor='orange')

    # 5. Leg contact (middle middle)
    ax5 = plt.subplot(3, 3, 5)
    ax5.fill_between(timesteps, 0, left_leg, label='Left Leg', alpha=0.6, step='post')
    ax5.fill_between(timesteps, 0, right_leg, label='Right Leg', alpha=0.6, step='post')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Contact')
    ax5.set_title('Leg Ground Contact')
    ax5.set_ylim(-0.1, 1.1)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Actions (middle right)
    ax6 = plt.subplot(3, 3, 6)
    action_colors = ['gray', 'blue', 'red', 'green']
    for i, action_name in enumerate(action_names):
        mask = actions == i
        ax6.scatter(timesteps[mask], actions[mask], label=action_name,
                   c=action_colors[i], alpha=0.6, s=20)
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Action')
    ax6.set_title('Actions Taken')
    ax6.set_yticks([0, 1, 2, 3])
    ax6.set_yticklabels(action_names)
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 7. Rewards (bottom left)
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(timesteps, rewards, linewidth=2, color='purple')
    ax7.fill_between(timesteps, 0, rewards, alpha=0.3, color='purple')
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax7.set_xlabel('Timestep')
    ax7.set_ylabel('Reward')
    ax7.set_title(f'Rewards (Total: {np.sum(rewards):.2f})')
    ax7.grid(True, alpha=0.3)

    # 8. Cumulative reward (bottom middle)
    ax8 = plt.subplot(3, 3, 8)
    cumulative_reward = np.cumsum(rewards)
    ax8.plot(timesteps, cumulative_reward, linewidth=2, color='green')
    ax8.fill_between(timesteps, 0, cumulative_reward, alpha=0.3, color='green')
    ax8.set_xlabel('Timestep')
    ax8.set_ylabel('Cumulative Reward')
    ax8.set_title('Cumulative Reward')
    ax8.grid(True, alpha=0.3)

    # 9. Action distribution (bottom right)
    ax9 = plt.subplot(3, 3, 9)
    action_counts = [np.sum(actions == i) for i in range(4)]
    bars = ax9.bar(action_names, action_counts, color=action_colors, alpha=0.7)
    ax9.set_ylabel('Count')
    ax9.set_title('Action Distribution')
    ax9.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for bar, count in zip(bars, action_counts):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10)

    # 10. Sample images if available (bottom spanning plot)
    if images is not None:
        # Create a subplot that spans the bottom row
        ax10 = plt.subplot(3, 1, 3)

        # Display 5 evenly spaced frames
        num_samples = min(5, len(images))
        sample_indices = np.linspace(0, len(images)-1, num_samples, dtype=int)

        for idx, img_idx in enumerate(sample_indices):
            ax_img = plt.subplot(3, num_samples, 2*num_samples + idx + 1)
            ax_img.imshow(images[img_idx])
            ax_img.set_title(f'Step {img_idx}', fontsize=8)
            ax_img.axis('off')

    plt.tight_layout()
    return fig


def animate_episode(f, episode_key, save_path=None, show_images=False):
    """Create an animation of the lander trajectory.

    Args:
        f: HDF5 file handle
        episode_key: Key of the episode (e.g., "demo_0")
        save_path: Optional path to save the animation
        show_images: Whether to display rendered images if available
    """
    ep_grp = f[f"data/{episode_key}"]

    # Load data
    states = np.array(ep_grp["obs/state"])
    actions = np.array(ep_grp["actions"])
    rewards = np.array(ep_grp["rewards"])

    # Check if images are available
    has_images = "images" in ep_grp["obs"]
    images = None
    if has_images and show_images:
        images = np.array(ep_grp["obs/images"])

    num_steps = len(states)

    # Extract state components
    x_pos = states[:, 0]
    y_pos = states[:, 1]
    angle = states[:, 4]
    left_leg = states[:, 6]
    right_leg = states[:, 7]

    # Create figure
    if images is not None:
        fig = plt.figure(figsize=(18, 6))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Trajectory with lander
    ax1.set_xlim(x_pos.min() - 0.5, x_pos.max() + 0.5)
    ax1.set_ylim(-0.2, y_pos.max() + 0.5)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Lander Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Draw ground
    ax1.axhline(y=0, color='brown', linewidth=3)
    ax1.fill_between([x_pos.min() - 1, x_pos.max() + 1], -0.2, 0,
                     color='brown', alpha=0.3)

    # Initialize trajectory line
    trajectory_line, = ax1.plot([], [], 'b-', alpha=0.5, linewidth=2)

    # Initialize lander (simplified representation)
    lander_body = Rectangle((0, 0), 0.1, 0.1, fill=True, color='gray')
    ax1.add_patch(lander_body)

    # Right plot: Metrics over time
    timesteps = np.arange(num_steps)
    ax2.set_xlim(0, num_steps)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Value')
    ax2.set_title('Episode Metrics')
    ax2.grid(True, alpha=0.3)

    # Plot cumulative reward
    cumulative_reward = np.cumsum(rewards)
    ax2.plot(timesteps, cumulative_reward, 'g-', label='Cumulative Reward', alpha=0.5)
    ax2.plot(timesteps, y_pos, 'b-', label='Height', alpha=0.5)

    current_reward_line = ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.legend()

    # Text display
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Image display if available
    image_display = None
    if images is not None:
        ax3.set_title('Rendered Frame')
        ax3.axis('off')
        image_display = ax3.imshow(images[0])

    def init():
        ret = [trajectory_line, lander_body, current_reward_line, info_text]
        trajectory_line.set_data([], [])
        if image_display is not None:
            ret.append(image_display)
        return ret

    def update(frame):
        # Update trajectory
        trajectory_line.set_data(x_pos[:frame+1], y_pos[:frame+1])

        # Update lander position and orientation
        x, y = x_pos[frame], y_pos[frame]
        ang = angle[frame]

        # Rotate and position lander body
        # Simple rectangle representation
        width, height = 0.15, 0.08
        lander_body.set_xy((x - width/2, y - height/2))
        lander_body.set_angle(np.rad2deg(ang))

        # Update info text
        action_names = ['Do Nothing', 'Fire Left', 'Fire Main', 'Fire Right']
        info = f'Step: {frame}/{num_steps}\n'
        info += f'Position: ({x:.3f}, {y:.3f})\n'
        info += f'Angle: {np.rad2deg(ang):.1f}°\n'
        info += f'Action: {action_names[actions[frame]]}\n'
        info += f'Reward: {rewards[frame]:.2f}\n'
        info += f'Total: {cumulative_reward[frame]:.2f}'
        info_text.set_text(info)

        # Update current position marker
        current_reward_line.set_xdata([frame, frame])

        # Update image if available
        ret = [trajectory_line, lander_body, current_reward_line, info_text]
        if image_display is not None and frame < len(images):
            image_display.set_array(images[frame])
            ret.append(image_display)

        return ret

    anim = FuncAnimation(fig, update, frames=num_steps, init_func=init,
                        blit=True, interval=50, repeat=True)

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=20)
        print("Animation saved!")

    return fig, anim


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LunarLander HDF5 demonstrations"
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
        "--mode",
        type=str,
        choices=["overview", "animate", "both"],
        default="overview",
        help="Visualization mode: overview (static plots), animate (trajectory animation), or both"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots to PNG files"
    )
    parser.add_argument(
        "--save-anim",
        type=str,
        default=None,
        help="Save animation to GIF file (provide filename)"
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Display rendered images if available in HDF5 file"
    )

    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        print(f"Error: HDF5 file not found: {hdf5_path}")
        sys.exit(1)

    print(f"Opening HDF5 file: {hdf5_path}")
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

        print(f"Total episodes: {len(all_episodes)}")
        print(f"Visualizing {len(episodes_to_viz)} episode(s)")
        print()

        # Print episode statistics
        print("Episode Statistics:")
        print("-" * 60)
        for ep_key in episodes_to_viz:
            if ep_key not in all_episodes:
                print(f"Warning: Episode {ep_key} not found, skipping")
                continue

            ep_grp = f[f"data/{ep_key}"]
            rewards = np.array(ep_grp["rewards"])
            num_steps = len(rewards)
            total_reward = np.sum(rewards)
            print(f"{ep_key}: {num_steps} steps, Total Reward: {total_reward:.2f}")
        print("-" * 60)
        print()

        # Visualize episodes
        for ep_key in episodes_to_viz:
            if ep_key not in all_episodes:
                continue

            if args.mode in ["overview", "both"]:
                print(f"Creating overview plot for {ep_key}...")
                fig = plot_episode_overview(f, ep_key, show_images=args.show_images)
                if args.save:
                    save_path = f"{ep_key}_overview.png"
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"Saved to {save_path}")

            if args.mode in ["animate", "both"]:
                print(f"Creating animation for {ep_key}...")
                anim_save_path = None
                if args.save_anim:
                    anim_save_path = f"{ep_key}_{args.save_anim}"
                fig, anim = animate_episode(f, ep_key, save_path=anim_save_path, show_images=args.show_images)

        print()
        print("Visualization complete!")
        if not args.save and not args.save_anim:
            print("Displaying plots... Close all windows to exit.")
            plt.show()


if __name__ == "__main__":
    main()
