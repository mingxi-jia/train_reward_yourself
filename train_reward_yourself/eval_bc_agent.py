#!/usr/bin/env python3
"""
Evaluate a trained Behavioral Cloning agent.

Usage:
    # Basic evaluation
    python -m train_reward_yourself.eval_bc_agent \
        --model bc_lunarlander.zip \
        --env-name LunarLander-v3 \
        --episodes 10

    # Evaluate and save video with BC critic values visualization
    python -m train_reward_yourself.eval_bc_agent \
        --model bc_lunarlander.zip \
        --env-name LunarLander-v3 \
        --episodes 10 \
        --save-video \
        --video-path eval_bc_lunarlander.mp4

    # Evaluate with VIP critic visualization (shows both BC and VIP values + comparison plot)
    python -m train_reward_yourself.eval_bc_agent \
        --model bc_lunarlander.zip \
        --env-name LunarLander-v3 \
        --episodes 10 \
        --save-video \
        --vip-critic vip_critic_output/vip_critic.pt \
        --vip-demos demos_lunarlander_v3_state.hdf5 \
        --vip-single-goal \
        --video-path eval_with_vip.mp4
"""

import argparse
import numpy as np
import torch
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, List
from stable_baselines3 import PPO

# Import VIP modules
from train_reward_yourself.train_vip_critic import VIPCritic
from train_reward_yourself.vip_reward_wrapper import extract_expert_goals


def get_critic_value(
    model,
    obs: np.ndarray,
    device: str = "cpu",
    return_mean: Optional[float] = None,
    return_std: Optional[float] = None,
) -> float:
    """Extract critic value V(s) from the model.

    Args:
        model: PPO model with critic
        obs: Observation
        device: Device for computation
        return_mean: Mean used to normalize returns during training (for denormalization)
        return_std: Std used to normalize returns during training (for denormalization)

    Returns:
        Denormalized critic value
    """
    with torch.no_grad():
        device = model.device
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        features = model.policy.extract_features(obs_tensor)
        value = model.policy.value_net(features).item()

        # Denormalize if stats provided
        if return_mean is not None and return_std is not None:
            value = value * return_std + return_mean

        return value


def get_vip_critic_value(
    vip_critic,
    obs: np.ndarray,
    goal: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> float:
    """Extract VIP critic value V(s, g).

    Args:
        vip_critic: VIP critic model
        obs: Observation
        goal: Goal state tensor (not needed if using learned goal)
        device: Device for computation

    Returns:
        VIP critic value
    """
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        if hasattr(vip_critic, 'use_learned_goal') and vip_critic.use_learned_goal:
            # Use learned goal embedding
            value = vip_critic(obs_tensor).item()
        else:
            # Use provided goal
            if goal is None:
                raise ValueError("Goal required for VIP critic without learned goal")
            goal_batch = goal.unsqueeze(0) if goal.dim() == 1 else goal
            value = vip_critic(obs_tensor, goal_batch.to(device)).item()

        return value


def overlay_value_on_frame(
    frame: np.ndarray,
    episode: int,
    step: int,
    bc_value: Optional[float] = None,
    vip_value: Optional[float] = None,
) -> np.ndarray:
    """Overlay critic value(s) and step info on video frame.

    Args:
        frame: Video frame
        episode: Episode number
        step: Step number
        bc_value: BC critic value (optional)
        vip_value: VIP critic value (optional)
    """
    frame = frame.copy()
    h, w = frame.shape[:2]

    # Determine overlay height based on how many values we're showing
    num_values = sum([bc_value is not None, vip_value is not None])
    overlay_height = 50 + (num_values * 25)

    # Create semi-transparent overlay for text background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (w-5, overlay_height + 5), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 25
    cv2.putText(frame, f"Episode: {episode}", (10, y_pos), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y_pos += 25
    cv2.putText(frame, f"Step: {step}", (10, y_pos), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y_pos += 25

    # Show BC critic value if available
    if bc_value is not None:
        bc_color = (0, 255, 0) if bc_value >= 0 else (0, 0, 255)
        cv2.putText(frame, f"BC V(s): {bc_value:+.2f}", (10, y_pos), font, 0.6, bc_color, 2, cv2.LINE_AA)
        y_pos += 25

    # Show VIP critic value if available
    if vip_value is not None:
        vip_color = (0, 255, 255) if vip_value >= 0 else (255, 0, 255)  # Cyan/Magenta for VIP
        cv2.putText(frame, f"VIP V(s,g): {vip_value:+.2f}", (10, y_pos), font, 0.6, vip_color, 2, cv2.LINE_AA)

    return frame


def create_value_comparison_plot(
    bc_values: List[float],
    vip_values: List[float],
    episode: int,
    save_path: str,
):
    """Create a comparison plot of BC and VIP critic values over time.

    Args:
        bc_values: List of BC critic values over episode
        vip_values: List of VIP critic values over episode
        episode: Episode number
        save_path: Path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    steps = np.arange(len(bc_values))

    # Plot 1: Both values overlaid
    ax1.plot(steps, bc_values, 'g-', label='BC V(s)', linewidth=2, alpha=0.8)
    ax1.plot(steps, vip_values, 'c-', label='VIP V(s,g)', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Episode {episode}: BC Critic vs VIP Critic Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Difference (BC - VIP)
    diff = np.array(bc_values) - np.array(vip_values)
    ax2.plot(steps, diff, 'b-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.fill_between(steps, 0, diff, alpha=0.3)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Difference')
    ax2.set_title('Difference: BC V(s) - VIP V(s,g)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Scatter plot BC vs VIP
    ax3.scatter(bc_values, vip_values, alpha=0.6, s=50)
    # Add diagonal line (y=x) for reference
    min_val = min(min(bc_values), min(vip_values))
    max_val = max(max(bc_values), max(vip_values))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x', alpha=0.5)
    ax3.set_xlabel('BC V(s)')
    ax3.set_ylabel('VIP V(s,g)')
    ax3.set_title('Correlation: BC Critic vs VIP Critic')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Calculate correlation
    if len(bc_values) > 1:
        corr = np.corrcoef(bc_values, vip_values)[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved value comparison plot to: {save_path}")


def evaluate_agent(
    model,
    env: gym.Env,
    num_episodes: int = 10,
    render: bool = False,
    save_video: bool = False,
    video_path: Optional[str] = None,
    device: str = "cpu",
    return_mean: Optional[float] = None,
    return_std: Optional[float] = None,
    vip_critic = None,
    vip_goal: Optional[torch.Tensor] = None,
):
    """Evaluate an agent in the environment.

    Args:
        model: PPO model for action selection
        env: Environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation
        save_video: Whether to save video
        video_path: Path to save video
        device: Device for computation
        return_mean: Mean for BC critic denormalization
        return_std: Std for BC critic denormalization
        vip_critic: Optional VIP critic to visualize instead of BC critic
        vip_goal: Goal state for VIP critic (if not using learned goal)
    """

    episode_rewards = []
    episode_lengths = []
    successes = []
    video_writer = None

    # Track values for comparison plot
    bc_values_for_plot = []
    vip_values_for_plot = []

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        episode_frames = []

        # Track values for this episode
        episode_bc_values = []
        episode_vip_values = []

        while not done:
            # Get critic values for current state
            bc_value = None
            vip_value = None

            if save_video or vip_critic is not None:
                # Always compute BC critic value
                bc_value = get_critic_value(model, obs, device, return_mean, return_std)

                # Compute VIP critic value if provided
                if vip_critic is not None:
                    vip_value = get_vip_critic_value(vip_critic, obs, vip_goal, device)

                # Track values for plotting (only for video episode)
                if save_video and ep == 2:
                    episode_bc_values.append(bc_value)
                    if vip_value is not None:
                        episode_vip_values.append(vip_value)

            # Render frame with overlays
            if save_video:
                frame = env.render()
                if frame is not None:
                    # Overlay both values
                    frame = overlay_value_on_frame(
                        frame, ep + 1, episode_length,
                        bc_value=bc_value,
                        vip_value=vip_value
                    )
                    episode_frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        # Write frames to video and create comparison plot
        if save_video and ep == 2 and episode_frames:
            h, w = episode_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))

            for frame in episode_frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            video_writer.release()
            print(f"\nSaved video to: {video_path}")

            # Save values for final comparison plot
            bc_values_for_plot = episode_bc_values
            vip_values_for_plot = episode_vip_values

            # Create comparison plot if we have VIP values
            if vip_critic is not None and len(vip_values_for_plot) > 0:
                plot_path = video_path.replace('.mp4', '_comparison.png')
                create_value_comparison_plot(
                    bc_values_for_plot,
                    vip_values_for_plot,
                    ep + 1,
                    plot_path
                )

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check for success if available
        if 'is_success' in info or 'success' in info:
            successes.append(info.get('is_success', info.get('success', False)))

        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }

    if successes:
        results["success_rate"] = np.mean(successes)

    return results, episode_rewards


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BC agent"
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.zip)")
    parser.add_argument("--env-name", type=str, required=True,
                        help="Environment name")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during evaluation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for evaluation")
    parser.add_argument("--save-video", action="store_true",
                        help="Save video of first episode with critic values overlay")
    parser.add_argument("--video-path", type=str, default=None,
                        help="Path to save video (default: eval_<model_name>.mp4)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for critic inference (cpu/cuda, default: cpu)")

    # VIP critic visualization
    parser.add_argument("--vip-critic", type=str, default=None,
                        help="Path to VIP critic checkpoint (.pt) to visualize instead of BC critic")
    parser.add_argument("--vip-demos", type=str, default=None,
                        help="Path to demos for goal extraction (only needed if VIP critic doesn't use learned goal)")
    parser.add_argument("--vip-single-goal", action="store_true",
                        help="Use single best goal from demos")
    parser.add_argument("--vip-n-demos", type=int, default=100,
                        help="Number of demos for goal extraction (default: 100)")

    args = parser.parse_args()

    # Determine video path
    if args.save_video and args.video_path is None:
        model_name = Path(args.model).stem
        args.video_path = f"eval_{model_name}.mp4"

    # Create environment
    if args.render or args.save_video:
        env = gym.make(args.env_name, render_mode="rgb_array" if args.save_video else "human")
    else:
        env = gym.make(args.env_name)

    if args.seed is not None:
        env.reset(seed=args.seed)

    # Load model
    print(f"Loading model from: {args.model}")
    model = PPO.load(args.model, env=env)

    # Try to load normalization stats from BC training
    return_mean, return_std = None, None
    model_path = Path(args.model)
    config_path = model_path.parent / "config.txt"

    if config_path.exists():
        print(f"Loading normalization stats from: {config_path}")
        with open(config_path, 'r') as f:
            in_norm_section = False
            for line in f:
                line = line.strip()
                if line == "Return Normalization:":
                    in_norm_section = True
                elif in_norm_section:
                    if line.startswith("mean:"):
                        return_mean = float(line.split(":")[1].strip())
                    elif line.startswith("std:"):
                        return_std = float(line.split(":")[1].strip())
                        break

        if return_mean is not None and return_std is not None:
            print(f"Loaded return normalization: mean={return_mean:.2f}, std={return_std:.2f}")
        else:
            print("Warning: No return normalization found in config")
    else:
        print(f"Warning: Config file not found at {config_path}")
        print("Critic values will be shown in normalized space")

    # Load VIP critic if specified
    vip_critic_loaded = None
    vip_goal = None

    if args.vip_critic is not None:
        vip_critic_path = Path(args.vip_critic)
        if not vip_critic_path.exists():
            raise ValueError(f"VIP critic not found: {vip_critic_path}")

        print(f"\nLoading VIP critic from: {vip_critic_path}")

        # Load checkpoint
        checkpoint = torch.load(str(vip_critic_path), map_location=args.device)
        use_learned_goal = checkpoint.get("use_learned_goal", False)

        # Create VIP critic
        vip_critic_loaded = VIPCritic(
            observation_space=env.observation_space,
            obs_type=checkpoint.get("obs_type", "state"),
            features_dim=checkpoint.get("features_dim", 256),
            use_learned_goal=use_learned_goal,
        ).to(args.device)
        vip_critic_loaded.load_state_dict(checkpoint["model_state_dict"])
        vip_critic_loaded.eval()

        print("VIP critic loaded successfully")

        # Extract goal if needed
        if use_learned_goal:
            print("VIP critic uses learned goal embedding")
        else:
            if args.vip_demos is None:
                raise ValueError("--vip-demos required when VIP critic doesn't use learned goal")

            vip_demos_path = Path(args.vip_demos)
            if not vip_demos_path.exists():
                raise ValueError(f"VIP demos not found: {vip_demos_path}")

            print(f"Extracting goal from: {vip_demos_path}")
            vip_goals = extract_expert_goals(
                hdf5_path=str(vip_demos_path),
                n_demos=args.vip_n_demos,
                obs_type="state",
                sample_mode="final",
                use_single_goal=args.vip_single_goal,
            )
            vip_goal = vip_goals[0] if args.vip_single_goal else vip_goals[0]
            print(f"Using goal with shape: {vip_goal.shape}")

    # Evaluate
    print(f"\nEvaluating on {args.env_name} for {args.episodes} episodes...")
    if args.save_video:
        print(f"Video will be saved to: {args.video_path}")
    if vip_critic_loaded is not None:
        print("Visualizing VIP critic V(s,g) instead of BC critic")

    results, episode_rewards = evaluate_agent(
        model,
        env,
        num_episodes=args.episodes,
        render=args.render,
        save_video=args.save_video,
        video_path=args.video_path,
        device=args.device,
        return_mean=return_mean,
        return_std=return_std,
        vip_critic=vip_critic_loaded,
        vip_goal=vip_goal,
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes:      {args.episodes}")
    print(f"Mean Reward:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Min Reward:    {results['min_reward']:.2f}")
    print(f"Max Reward:    {results['max_reward']:.2f}")
    print(f"Mean Length:   {results['mean_length']:.1f} ± {results['std_length']:.1f}")

    if 'success_rate' in results:
        print(f"Success Rate:  {results['success_rate']*100:.1f}%")

    print("="*60)

    env.close()


if __name__ == "__main__":
    main()
