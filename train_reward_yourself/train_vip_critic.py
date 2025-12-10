#!/usr/bin/env python3
"""
Train a goal-conditioned value function using VIP (Value Implicit Pretraining).

Based on "Value Implicit Pretraining for RL" (https://arxiv.org/pdf/2210.00030)

Uses temporal contrastive learning:
- V(s, g) = "can state s reach goal g?"
- Positive pairs: (s_t, s_future) from same trajectory
- Negative pairs: (s_t, s_random) from different trajectories
- InfoNCE loss maximizes agreement for reachable goals

Usage:
    # Train VIP critic
    python -m train_reward_yourself.train_vip_critic \
        --data demos_lunarlander_v3.hdf5 \
        --env-name LunarLander-v3 \
        --obs-type state \
        --output vip_critic \
        --n-demos 100 \
        --epochs 50 \
        --horizon 10

    # Train with visualization
    python -m train_reward_yourself.train_vip_critic \
        --data demos_lunarlander_v3.hdf5 \
        --env-name LunarLander-v3 \
        --obs-type state \
        --output vip_critic \
        --n-demos 100 \
        --epochs 50 \
        --visualize \
        --vis-demo-idx 0

    # Visualize existing checkpoint
    python -m train_reward_yourself.train_vip_critic \
        --data demos_lunarlander_v3.hdf5 \
        --env-name LunarLander-v3 \
        --obs-type state \
        --output vip_critic \
        --visualize-only \
        --vis-demo-idx 0
"""

import argparse
import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
import gymnasium as gym
import cv2

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class VIPDataset(Dataset):
    """Dataset for VIP training with state pairs."""

    def __init__(
        self,
        hdf5_path: str,
        obs_type: str = "state",
        episodes: Optional[List[str]] = None,
        horizon: int = 10,
    ):
        """
        Args:
            hdf5_path: Path to HDF5 demos
            obs_type: "state" or "image"
            episodes: Episode keys to load
            horizon: Max steps ahead for positive goals
        """
        self.hdf5_path = hdf5_path
        self.obs_type = obs_type
        self.horizon = horizon
        self.observations = []
        self.episode_starts = []

        with h5py.File(hdf5_path, "r") as f:
            all_episodes = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
            if episodes is None:
                episodes = all_episodes

            print(f"Loading {len(episodes)} episodes for VIP training...")

            for ep_key in tqdm(episodes, desc="Loading"):
                if ep_key not in all_episodes:
                    continue

                ep_grp = f[f"data/{ep_key}"]
                if obs_type == "state":
                    obs = np.array(ep_grp["obs/state"])
                elif obs_type == "image":
                    obs = np.array(ep_grp["obs/images"])
                else:
                    raise ValueError(f"Unknown obs_type: {obs_type}")

                if len(self.episode_starts) == 0:
                    self.episode_starts.append(0)
                else:
                    self.episode_starts.append(self.episode_starts[-1] + len(obs))

                self.observations.append(obs)

        self.observations = np.concatenate(self.observations, axis=0)
        self.episode_starts.append(len(self.observations))

        print(f"Loaded {len(self.observations)} transitions")
        print(f"Observation shape: {self.observations.shape}")

    def __len__(self):
        return len(self.observations)

    def _get_episode_idx(self, idx):
        """Find which episode index belongs to."""
        for ep_idx in range(len(self.episode_starts) - 1):
            if self.episode_starts[ep_idx] <= idx < self.episode_starts[ep_idx + 1]:
                return ep_idx
        return len(self.episode_starts) - 2

    def __getitem__(self, idx):
        """Returns (state, positive_goal, negative_goal)."""
        obs = self.observations[idx]

        # Get positive goal: future state in same episode
        ep_idx = self._get_episode_idx(idx)
        ep_start = self.episode_starts[ep_idx]
        ep_end = self.episode_starts[ep_idx + 1]

        # Sample positive goal within horizon
        max_offset = min(self.horizon, ep_end - idx - 1)
        if max_offset < 1:
            # Last state in episode - use itself as goal
            pos_goal_idx = idx
        else:
            future_offset = np.random.randint(1, max_offset + 1)
            pos_goal_idx = idx + future_offset

        pos_goal = self.observations[pos_goal_idx]

        # Sample negative goal: random state from different episode
        neg_ep_idx = ep_idx
        while neg_ep_idx == ep_idx:
            neg_ep_idx = np.random.randint(0, len(self.episode_starts) - 1)

        neg_ep_start = self.episode_starts[neg_ep_idx]
        neg_ep_end = self.episode_starts[neg_ep_idx + 1]
        neg_goal_idx = np.random.randint(neg_ep_start, neg_ep_end)
        neg_goal = self.observations[neg_goal_idx]

        # Convert to tensors
        if self.obs_type == "image":
            obs = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
            pos_goal = torch.from_numpy(pos_goal).permute(2, 0, 1).float() / 255.0
            neg_goal = torch.from_numpy(neg_goal).permute(2, 0, 1).float() / 255.0
        else:
            obs = torch.from_numpy(obs).float()
            pos_goal = torch.from_numpy(pos_goal).float()
            neg_goal = torch.from_numpy(neg_goal).float()

        return obs, pos_goal, neg_goal


class MLPFeatureExtractor(BaseFeaturesExtractor):
    """MLP for state observations."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]

        self.mlp = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class VIPCritic(nn.Module):
    """Goal-conditioned value function V(s, g)."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        obs_type: str = "state",
        features_dim: int = 256,
        use_learned_goal: bool = False,
    ):
        super().__init__()

        self.obs_type = obs_type
        self.features_dim = features_dim
        self.use_learned_goal = use_learned_goal

        # Feature extractor
        if obs_type == "image":
            raise NotImplementedError("Image observations not yet implemented for VIP")
        else:
            self.feature_extractor = MLPFeatureExtractor(observation_space, features_dim)

        # Learned goal embedding (if enabled)
        if use_learned_goal:
            self.goal_embedding = nn.Parameter(torch.randn(1, features_dim))
            print("VIPCritic: Using learned goal embedding (no need for expert goals)")

        # Goal-conditioned value head: V(s, g) from concat(phi(s), phi(g))
        self.value_head = nn.Sequential(
            nn.Linear(features_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        """Compute V(s, g).

        Args:
            obs: States (B, obs_dim)
            goal: Goals (B, obs_dim). If use_learned_goal=True, this is ignored.

        Returns:
            Values (B, 1)
        """
        obs_features = self.feature_extractor(obs)

        if self.use_learned_goal:
            # Use learned goal embedding (broadcast to batch size)
            batch_size = obs_features.shape[0]
            goal_features = self.goal_embedding.expand(batch_size, -1)
        else:
            # Use provided goal
            goal_features = self.feature_extractor(goal)

        concat = torch.cat([obs_features, goal_features], dim=-1)
        return self.value_head(concat)

    def get_learned_goal(self) -> torch.Tensor:
        """Get the learned goal embedding (only valid if use_learned_goal=True)."""
        if not self.use_learned_goal:
            raise ValueError("Critic was not trained with learned goal embedding")
        return self.goal_embedding


def vip_loss(
    critic: VIPCritic,
    states: torch.Tensor,
    pos_goals: torch.Tensor,
    neg_goals: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """VIP InfoNCE contrastive loss.

    Args:
        critic: VIP critic network
        states: Current states (B, obs_dim)
        pos_goals: Positive (reachable) goals (B, obs_dim)
        neg_goals: Negative (unreachable) goals (B, obs_dim)
        temperature: Temperature for softmax

    Returns:
        InfoNCE loss
    """
    # Compute V(s, g+) and V(s, g-)
    pos_values = critic(states, pos_goals) / temperature  # (B, 1)
    neg_values = critic(states, neg_goals) / temperature  # (B, 1)

    # InfoNCE: log(exp(V+) / (exp(V+) + exp(V-)))
    # = V+ - log(exp(V+) + exp(V-))
    # = V+ - logsumexp([V+, V-])
    logits = torch.cat([pos_values, neg_values], dim=-1)  # (B, 2)
    labels = torch.zeros(len(states), dtype=torch.long, device=states.device)  # Positive is class 0

    loss = F.cross_entropy(logits, labels)
    return loss


def train_vip_critic(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    critic: VIPCritic,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    temperature: float = 0.1,
    weight_decay: float = 1e-4,
):
    """Train VIP critic with contrastive learning."""

    optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(num_epochs):
        # Training
        critic.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            states, pos_goals, neg_goals = batch
            states = states.to(device)
            pos_goals = pos_goals.to(device)
            neg_goals = neg_goals.to(device)

            optimizer.zero_grad()

            loss = vip_loss(critic, states, pos_goals, neg_goals, temperature)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / train_batches
        history["train_loss"].append(avg_train_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # Validation
        if val_loader is not None:
            critic.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    states, pos_goals, neg_goals = batch
                    states = states.to(device)
                    pos_goals = pos_goals.to(device)
                    neg_goals = neg_goals.to(device)

                    loss = vip_loss(critic, states, pos_goals, neg_goals, temperature)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches
            history["val_loss"].append(avg_val_loss)
            print(f"         Val Loss = {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)
        else:
            scheduler.step(avg_train_loss)

    return history


def overlay_value_on_frame(
    frame: np.ndarray,
    episode: int,
    step: int,
    value: float,
    goal_type: str = "learned",
) -> np.ndarray:
    """Overlay VIP critic value on frame.

    Args:
        frame: RGB frame (H, W, 3) in [0, 255]
        episode: Episode number
        step: Step number
        value: VIP critic value V(s, g)
        goal_type: "learned" or "expert"

    Returns:
        Frame with value overlaid
    """
    frame = frame.copy().astype(np.uint8)
    h, w = frame.shape[:2]

    # Add black bar at top for text
    bar_height = 100
    frame_with_bar = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    frame_with_bar[bar_height:] = frame

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_type = cv2.LINE_AA  # Anti-aliased for better rendering

    # Episode and step info (white)
    cv2.putText(
        frame_with_bar,
        f"Episode {episode} | Step {step}",
        (15, 35),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        line_type
    )

    # VIP value with color coding (cyan for positive, magenta for negative)
    value_color = (0, 255, 255) if value >= 0 else (255, 0, 255)
    goal_label = "learned goal" if goal_type == "learned" else "expert goal"
    cv2.putText(
        frame_with_bar,
        f"VIP V(s,g): {value:+.2f} ({goal_label})",
        (15, 70),
        font,
        font_scale,
        value_color,
        thickness,
        line_type
    )

    return frame_with_bar


def visualize_vip_on_demo(
    critic: VIPCritic,
    hdf5_path: str,
    demo_idx: int,
    goal: Optional[torch.Tensor],
    env_name: str,
    output_path: str,
    device: torch.device,
    obs_type: str = "state",
    fps: int = 30,
):
    """Create video of demo with VIP critic values overlaid.

    Args:
        critic: Trained VIP critic
        hdf5_path: Path to HDF5 demos
        demo_idx: Demo episode index to visualize
        goal: Goal tensor (N, obs_dim) or None if using learned goal
        env_name: Environment name for rendering
        output_path: Path to save video
        device: Device for inference
        obs_type: "state" or "image"
        fps: Frames per second
    """
    critic.eval()

    # Load demo episode
    with h5py.File(hdf5_path, "r") as f:
        ep_key = f"demo_{demo_idx}"
        if ep_key not in f["data"]:
            raise ValueError(f"Demo {ep_key} not found in {hdf5_path}")

        ep_grp = f[f"data/{ep_key}"]

        # Load observations and actions
        if obs_type == "state":
            observations = np.array(ep_grp["obs/state"])
        elif obs_type == "image":
            observations = np.array(ep_grp["obs/images"])
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")

        actions = np.array(ep_grp["actions"])
        rewards = np.array(ep_grp["rewards"])

    print(f"\nVisualizing demo {demo_idx}:")
    print(f"  Length: {len(observations)} steps")
    print(f"  Total reward: {rewards.sum():.2f}")

    # Determine goal type
    if critic.use_learned_goal:
        goal_type = "learned"
        print(f"  Using learned goal embedding")
    else:
        goal_type = "expert"
        if goal is None:
            raise ValueError("Goal must be provided when critic doesn't use learned goal")
        # Use first goal if multiple provided
        if len(goal.shape) > 1 and goal.shape[0] > 1:
            goal = goal[0:1]  # Take first goal
        print(f"  Using expert goal from demonstrations")

    # Always render from environment to get raw RGB frames
    env = gym.make(env_name, render_mode="rgb_array")
    print(f"  Rendering raw RGB frames from environment...")

    # Compute VIP values for all states
    vip_values = []
    with torch.no_grad():
        for obs in tqdm(observations, desc="Computing VIP values"):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

            if critic.use_learned_goal:
                value = critic(obs_tensor).item()
            else:
                value = critic(obs_tensor, goal.to(device)).item()

            vip_values.append(value)

    # Replay episode and create video frames
    print(f"  Creating video frames...")
    frames_with_values = []

    # Reset environment and replay the episode
    env.reset()

    for step, (action, vip_value) in enumerate(tqdm(zip(actions, vip_values), desc="Rendering frames", total=len(actions))):
        # Get raw RGB frame from environment
        frame = env.render()

        # Overlay VIP value
        frame_with_value = overlay_value_on_frame(
            frame, demo_idx, step, vip_value, goal_type
        )
        frames_with_values.append(frame_with_value)

        # Step environment for next frame
        env.step(action)

    # Save video
    h, w = frames_with_values[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames_with_values:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    env.close()

    print(f"  Video saved to: {output_path}")
    print(f"  VIP value range: [{min(vip_values):.2f}, {max(vip_values):.2f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Train VIP critic with contrastive learning"
    )

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to HDF5 demonstrations")
    parser.add_argument("--obs-type", type=str, default="state", choices=["state", "image"],
                        help="Observation type")
    parser.add_argument("--n-demos", type=int, default=100,
                        help="Number of demonstrations (default: 100)")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Train split fraction (default: 0.9)")

    # VIP parameters
    parser.add_argument("--horizon", type=int, default=10,
                        help="Max steps ahead for positive goals (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for InfoNCE loss (default: 0.1)")
    parser.add_argument("--use-learned-goal", action="store_true",
                        help="Use learned goal embedding instead of expert goals (no need to extract goals)")

    # Training
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--features-dim", type=int, default=256,
                        help="Feature dimension (default: 256)")

    # Environment (for creating observation space)
    parser.add_argument("--env-name", type=str, required=True,
                        help="Environment name")

    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")

    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Create video visualization of VIP critic on demo episode")
    parser.add_argument("--vis-demo-idx", type=int, default=0,
                        help="Demo episode index to visualize (default: 0)")
    parser.add_argument("--vis-fps", type=int, default=30,
                        help="Video FPS (default: 30)")
    parser.add_argument("--visualize-only", action="store_true",
                        help="Skip training and only visualize existing checkpoint")

    args = parser.parse_args()

    # Set seed
    set_random_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output)
    critic_path = output_dir / "vip_critic.pt"

    # Check if we're in visualize-only mode
    if args.visualize_only:
        if not critic_path.exists():
            raise ValueError(f"Checkpoint not found: {critic_path}\nCannot visualize without trained model.")

        print("\n" + "="*60)
        print("Loading trained VIP critic for visualization...")
        print("="*60)

        # Load checkpoint
        checkpoint = torch.load(str(critic_path), map_location=device)

        # Create environment to get observation space
        env = gym.make(args.env_name)

        # Recreate critic
        critic = VIPCritic(
            observation_space=env.observation_space,
            obs_type=checkpoint.get("obs_type", args.obs_type),
            features_dim=checkpoint.get("features_dim", args.features_dim),
            use_learned_goal=checkpoint.get("use_learned_goal", False),
        ).to(device)
        critic.load_state_dict(checkpoint["model_state_dict"])
        critic.eval()

        print(f"Loaded VIP critic from: {critic_path}")

        # Extract goal if needed
        if critic.use_learned_goal:
            goal = None
            print("Using learned goal embedding")
        else:
            from train_reward_yourself.vip_reward_wrapper import extract_expert_goals
            goal = extract_expert_goals(
                hdf5_path=args.data,
                n_demos=args.n_demos,
                obs_type=checkpoint.get("obs_type", args.obs_type),
                sample_mode="final",
                use_single_goal=True,
            )
            print("Extracted expert goal")

        # Create visualization
        video_path = output_dir / f"vip_demo_{args.vis_demo_idx}.mp4"
        visualize_vip_on_demo(
            critic=critic,
            hdf5_path=args.data,
            demo_idx=args.vis_demo_idx,
            goal=goal,
            env_name=args.env_name,
            output_path=str(video_path),
            device=device,
            obs_type=checkpoint.get("obs_type", args.obs_type),
            fps=args.vis_fps,
        )

        env.close()
        print(f"\nVisualization complete!")
        return

    # Load dataset
    print("\n" + "="*60)
    print("Loading VIP dataset...")
    print("="*60)

    episodes = [f"demo_{i}" for i in range(args.n_demos)]
    full_dataset = VIPDataset(
        args.data,
        obs_type=args.obs_type,
        episodes=episodes,
        horizon=args.horizon,
    )

    # Split train/val
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    def worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        worker_init_fn=worker_init_fn, generator=generator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        worker_init_fn=worker_init_fn
    ) if val_size > 0 else None

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create critic
    print("\n" + "="*60)
    print("Creating VIP critic...")
    print("="*60)

    env = gym.make(args.env_name)
    critic = VIPCritic(
        observation_space=env.observation_space,
        obs_type=args.obs_type,
        features_dim=args.features_dim,
        use_learned_goal=args.use_learned_goal,
    ).to(device)

    print(f"Critic architecture:")
    print(critic)
    if args.use_learned_goal:
        print("Using learned goal embedding - no need to extract expert goals during RL training!")

    # Train
    print("\n" + "="*60)
    print("Training VIP critic...")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    history = train_vip_critic(
        train_loader=train_loader,
        val_loader=val_loader,
        critic=critic,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
    )

    # Save model
    print("\n" + "="*60)
    print("Saving model...")
    print("="*60)

    critic_path = output_dir / "vip_critic.pt"
    checkpoint = {
        "model_state_dict": critic.state_dict(),
        "obs_type": args.obs_type,
        "features_dim": args.features_dim,
        "use_learned_goal": args.use_learned_goal,
        "history": history,
    }
    torch.save(checkpoint, str(critic_path))
    print(f"Saved VIP critic: {critic_path}")
    if args.use_learned_goal:
        print("Learned goal embedding is saved in the checkpoint - no expert goals needed!")

    # Save config
    config_path = output_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write("VIP Training Configuration\n")
        f.write("=" * 60 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"Saved config: {config_path}")

    # Visualization
    if args.visualize:
        print("\n" + "="*60)
        print("Creating visualization video...")
        print("="*60)

        # Extract goal if needed
        if args.use_learned_goal:
            goal = None
        else:
            # Extract goal from best demo (single canonical goal)
            from train_reward_yourself.vip_reward_wrapper import extract_expert_goals
            goal = extract_expert_goals(
                hdf5_path=args.data,
                n_demos=args.n_demos,
                obs_type=args.obs_type,
                sample_mode="final",
                use_single_goal=True,
            )

        video_path = output_dir / f"vip_demo_{args.vis_demo_idx}.mp4"
        visualize_vip_on_demo(
            critic=critic,
            hdf5_path=args.data,
            demo_idx=args.vis_demo_idx,
            goal=goal,
            env_name=args.env_name,
            output_path=str(video_path),
            device=device,
            obs_type=args.obs_type,
            fps=args.vis_fps,
        )

    env.close()
    print(f"\nTraining complete! Outputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
