#!/usr/bin/env python3
"""
Train a Behavioral Cloning agent from HDF5 demonstrations.

Supports both state-to-action (MLP) and image-to-action (CNN) mappings.
The trained model can be used to initialize PPO agents.
Optionally pretrain the critic (value function) from expert returns.

Usage:
    # Train from state observations
    python -m train_reward_yourself.train_bc_agent \
        --data demos_lunarlander_v3.hdf5 \
        --obs-type state \
        --env-name LunarLander-v3 \
        --output bc_lunarlander.zip

    # Train from image observations
    python -m train_reward_yourself.train_bc_agent \
        --data demos_lunarlander_v3.hdf5 \
        --obs-type image \
        --env-name LunarLander-v3 \
        --output bc_lunarlander_cnn.zip

    # Train with critic pretraining
    python -m train_reward_yourself.train_bc_agent \
        --data demos_lunarlander_v3.hdf5 \
        --obs-type state \
        --env-name LunarLander-v3 \
        --output bc_lunarlander.zip \
        --pretrain-critic \
        --gamma 0.99 \
        --value-loss-coef 0.5
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
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


def set_random_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_episode_returns(
    hdf5_path: str,
    n_demos: int,
    gamma: float,
    save_path: str,
) -> Dict[str, float]:
    """Plot ground truth value progression over time steps for each episode.

    Args:
        hdf5_path: Path to HDF5 demonstrations
        n_demos: Number of demonstrations to analyze
        gamma: Discount factor for return calculation
        save_path: Path to save the plot

    Returns:
        Dictionary with statistics (mean, std, min, max) of initial returns
    """
    all_episode_values = []  # List of arrays, one per episode
    initial_returns = []  # G_0 for each episode

    with h5py.File(hdf5_path, "r") as f:
        print(f"\nComputing ground truth value progression for {n_demos} episodes...")

        for i in tqdm(range(n_demos), desc="Computing returns"):
            ep_key = f"demo_{i}"
            if ep_key not in f["data"]:
                print(f"Warning: Episode {ep_key} not found, skipping")
                continue

            ep_grp = f[f"data/{ep_key}"]
            rewards = np.array(ep_grp["rewards"])

            # Compute discounted return at each time step: G_t
            returns = np.zeros_like(rewards)
            running_return = 0.0
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return
                returns[t] = running_return

            all_episode_values.append(returns)
            initial_returns.append(returns[0])

    initial_returns = np.array(initial_returns)

    # Compute statistics on initial returns
    stats = {
        "mean": float(np.mean(initial_returns)),
        "std": float(np.std(initial_returns)),
        "min": float(np.min(initial_returns)),
        "max": float(np.max(initial_returns)),
    }

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Value progression over time steps (N lines, one per episode)
    for i, values in enumerate(all_episode_values):
        ax1.plot(range(len(values)), values, alpha=0.6, linewidth=1.5)

    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.set_ylabel("Ground Truth Value V(s_t) = G_t", fontsize=12)
    ax1.set_title(f"Value Function Progression (γ={gamma}, n={len(all_episode_values)} episodes)",
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f"Initial Return (G_0) Stats:\n"
    stats_text += f"μ={stats['mean']:.2f}\n"
    stats_text += f"σ={stats['std']:.2f}\n"
    stats_text += f"min={stats['min']:.2f}\n"
    stats_text += f"max={stats['max']:.2f}"
    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Distribution of initial returns (G_0)
    ax2.hist(initial_returns, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(stats['mean'], color='red', linestyle='--',
                linewidth=2, label=f"Mean: {stats['mean']:.2f}")
    ax2.set_xlabel("Initial Ground Truth Value (G_0)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Distribution of Episode Initial Returns", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nGround Truth Value Statistics:")
    print(f"  Episodes: {len(all_episode_values)}")
    print(f"  Initial Return (G_0) Mean: {stats['mean']:.2f}")
    print(f"  Initial Return (G_0) Std:  {stats['std']:.2f}")
    print(f"  Initial Return (G_0) Min:  {stats['min']:.2f}")
    print(f"  Initial Return (G_0) Max:  {stats['max']:.2f}")
    print(f"  Saved plot to: {save_path}")

    return stats


class ImageObservationWrapper(gym.ObservationWrapper):
    """Wrapper that replaces state observations with rendered RGB images."""

    def __init__(self, env, target_shape=None, frame_stack=1, normalize=False):
        """
        Args:
            env: Gym environment with render_mode='rgb_array'
            target_shape: Target image shape (H, W) to resize to. If None, use render size.
            frame_stack: Number of frames to stack for temporal info (default: 1)
            normalize: If True, normalize to [0, 1]. If False, keep [0, 255] (default: False)
        """
        super().__init__(env)
        self.target_shape = target_shape
        self.frame_stack = frame_stack
        self.normalize = normalize

        # Render once to get image shape
        self.env.reset()
        img = self.env.render()
        h, w, c = img.shape

        if target_shape is not None:
            h, w = target_shape

        # Observation space depends on normalization
        if normalize:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(c * frame_stack, h, w), dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(c * frame_stack, h, w), dtype=np.uint8
            )

        # Frame buffer for stacking
        self.frames = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize frame buffer with first frame repeated
        frame = self._process_frame()
        self.frames = [frame.copy() for _ in range(self.frame_stack)]
        return self._get_observation(), info

    def _process_frame(self):
        """Process single frame: render, resize, optionally normalize."""
        img = self.env.render()  # (H, W, C) in [0, 255]

        # Resize if needed
        if self.target_shape is not None:
            img = cv2.resize(img, (self.target_shape[1], self.target_shape[0]), interpolation=cv2.INTER_AREA)

        # Normalize if requested
        if self.normalize:
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        else:
            img = img.astype(np.uint8)  # Keep [0, 255]

        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        return img

    def _get_observation(self):
        """Stack frames along channel dimension."""
        return np.concatenate(self.frames, axis=0)  # (C*K, H, W)

    def observation(self, observation):
        # Process new frame and add to buffer
        frame = self._process_frame()
        self.frames.append(frame)
        if len(self.frames) > self.frame_stack:
            self.frames.pop(0)

        return self._get_observation()


class DemonstrationDataset(Dataset):
    """PyTorch dataset for loading demonstrations from HDF5 file."""

    def __init__(self, hdf5_path: str, obs_type: str = "state", episodes: Optional[List[str]] = None,
                 compute_returns: bool = False, gamma: float = 0.99, frame_stack: int = 1):
        """
        Args:
            hdf5_path: Path to HDF5 file
            obs_type: Type of observation ("state" or "image")
            episodes: List of episode keys to load. If None, load all.
            compute_returns: Whether to compute returns for value function training
            gamma: Discount factor for return calculation
            frame_stack: Number of frames to stack (default: 1)
        """
        self.hdf5_path = hdf5_path
        self.obs_type = obs_type
        self.compute_returns = compute_returns
        self.frame_stack = frame_stack
        self.observations = []
        self.actions = []
        self.returns = [] if compute_returns else None
        self.episode_starts = []  # Track episode boundaries for frame stacking

        with h5py.File(hdf5_path, "r") as f:
            all_episodes = sorted([k for k in f["data"].keys() if k.startswith("demo_")])

            if episodes is None:
                episodes = all_episodes

            print(f"Loading {len(episodes)} episodes from {hdf5_path}...")

            for ep_key in tqdm(episodes, desc="Loading demos"):
                if ep_key not in all_episodes:
                    print(f"Warning: Episode {ep_key} not found, skipping")
                    continue

                ep_grp = f[f"data/{ep_key}"]

                # Load observations based on type
                if obs_type == "state":
                    obs = np.array(ep_grp["obs/state"])
                elif obs_type == "image":
                    if "images" in ep_grp["obs"]:
                        obs = np.array(ep_grp["obs/images"])
                    else:
                        raise ValueError(f"No images found in {ep_key}")
                else:
                    raise ValueError(f"Unknown obs_type: {obs_type}")

                actions = np.array(ep_grp["actions"])

                ep_len = len(obs)
                self.observations.append(obs)
                self.actions.append(actions)

                # Track episode start indices
                if len(self.episode_starts) == 0:
                    self.episode_starts.append(0)
                else:
                    self.episode_starts.append(self.episode_starts[-1] + ep_len)

                # Compute returns if needed
                if compute_returns:
                    rewards = np.array(ep_grp["rewards"])
                    ep_returns = self._compute_episode_returns(rewards, gamma)
                    self.returns.append(ep_returns)

        # Concatenate all episodes
        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        if compute_returns:
            self.returns = np.concatenate(self.returns, axis=0)

        # Add final boundary
        self.episode_starts.append(len(self.observations))

        print(f"Loaded {len(self.observations)} transitions")
        print(f"Observation shape: {self.observations.shape}")
        print(f"Action shape: {self.actions.shape}")
        if compute_returns:
            print(f"Returns shape: {self.returns.shape}")
            print(f"Returns range: [{self.returns.min():.2f}, {self.returns.max():.2f}]")

    def _compute_episode_returns(self, rewards: np.ndarray, gamma: float) -> np.ndarray:
        """Compute discounted returns for an episode."""
        returns = np.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        return returns

    def __len__(self):
        return len(self.observations)

    def _get_episode_idx(self, idx):
        """Find which episode this index belongs to."""
        for ep_idx in range(len(self.episode_starts) - 1):
            if self.episode_starts[ep_idx] <= idx < self.episode_starts[ep_idx + 1]:
                return ep_idx
        return len(self.episode_starts) - 2

    def _get_stacked_obs(self, idx):
        """Get frame-stacked observation at index idx."""
        if self.frame_stack == 1:
            return self.observations[idx]

        # Find episode boundaries
        ep_idx = self._get_episode_idx(idx)
        ep_start = self.episode_starts[ep_idx]

        # Collect frames (clamped to episode start)
        frames = []
        for i in range(self.frame_stack):
            frame_idx = max(ep_start, idx - (self.frame_stack - 1 - i))
            frames.append(self.observations[frame_idx])

        # Stack along channel dimension
        if self.obs_type == "image":
            # Image: (H, W, C) -> stack C dimension
            return np.concatenate(frames, axis=2)  # (H, W, C*K)
        else:
            # State: just concatenate
            return np.concatenate(frames, axis=0)

    def __getitem__(self, idx):
        obs = self._get_stacked_obs(idx)
        action = self.actions[idx]

        # Convert to torch tensors
        if self.obs_type == "image":
            # Image: (H, W, C*K) -> (C*K, H, W)
            obs = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0

            # Add brightness/contrast jitter (safe augmentation)
            # if np.random.rand() < 0.3:
            #     obs = obs * np.random.uniform(0.8, 1.2)  # brightness
            #     obs = torch.clamp(obs, 0.0, 1.0)
        else:
            obs = torch.from_numpy(obs).float()

        # Handle both discrete and continuous actions
        if len(action.shape) == 0 or (len(action.shape) == 1 and action.shape[0] == 1):
            # Discrete action
            action = torch.tensor(action, dtype=torch.long)
        else:
            # Continuous action
            action = torch.from_numpy(action).float()

        if self.compute_returns:
            return_val = torch.tensor(self.returns[idx], dtype=torch.float32)
            return obs, action, return_val

        return obs, action


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """CNN feature extractor for image observations."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class MLPFeatureExtractor(BaseFeaturesExtractor):
    """MLP feature extractor for state observations."""

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


class BCAgent(nn.Module):
    """Behavioral Cloning agent."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.Space,
        obs_type: str = "state",
        features_dim: int = 256,
        train_value_head: bool = False,
        value_bins: int = 0,
        value_min: float = -300.0,
        value_max: float = 300.0,
    ):
        super().__init__()

        self.obs_type = obs_type
        self.action_space = action_space
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        self.train_value_head = train_value_head
        self.value_bins = value_bins
        self.use_categorical_value = value_bins > 0

        # Feature extractor
        if obs_type == "image":
            self.feature_extractor = CNNFeatureExtractor(observation_space, features_dim)
        else:
            self.feature_extractor = MLPFeatureExtractor(observation_space, features_dim)

        # Action head
        if self.is_discrete:
            self.action_head = nn.Linear(features_dim, action_space.n)
        else:
            self.action_head = nn.Linear(features_dim, action_space.shape[0])

        # Value head (optional, for critic pretraining)
        if train_value_head:
            if self.use_categorical_value:
                # Categorical value distribution (classification)
                self.value_head = nn.Linear(features_dim, value_bins)
                # Create bin centers for value computation
                self.register_buffer(
                    "value_bin_centers",
                    torch.linspace(value_min, value_max, value_bins)
                )
            else:
                # Scalar value (regression)
                self.value_head = nn.Linear(features_dim, 1)

    def forward(self, obs, return_value=False):
        features = self.feature_extractor(obs)
        action_logits = self.action_head(features)

        if return_value and self.train_value_head:
            value_output = self.value_head(features)
            return action_logits, value_output

        return action_logits

    def discretize_values(self, continuous_values: torch.Tensor) -> torch.Tensor:
        """Convert continuous return values to discrete bin indices.

        Args:
            continuous_values: Continuous return values (B,) or (B, 1)

        Returns:
            Bin indices (B,) as long tensor
        """
        if not self.use_categorical_value:
            raise ValueError("discretize_values only works with categorical value heads")

        values = continuous_values.squeeze(-1)  # (B,)
        # Compute bin boundaries (midpoints between centers)
        bin_edges = (self.value_bin_centers[:-1] + self.value_bin_centers[1:]) / 2
        # Add edges for first and last bins
        bin_edges = torch.cat([
            torch.tensor([float('-inf')], device=bin_edges.device),
            bin_edges,
            torch.tensor([float('inf')], device=bin_edges.device)
        ])

        # Digitize: find which bin each value belongs to
        indices = torch.searchsorted(bin_edges, values, right=False) - 1
        indices = torch.clamp(indices, 0, self.value_bins - 1)

        return indices

    def compute_expected_value(self, value_logits: torch.Tensor) -> torch.Tensor:
        """Compute expected value from categorical distribution.

        Args:
            value_logits: Logits over value bins (B, num_bins)

        Returns:
            Expected values (B, 1)
        """
        if not self.use_categorical_value:
            raise ValueError("compute_expected_value only works with categorical value heads")

        # Softmax to get probabilities
        probs = F.softmax(value_logits, dim=-1)  # (B, num_bins)
        # Compute expectation
        expected = (probs * self.value_bin_centers).sum(dim=-1, keepdim=True)  # (B, 1)
        return expected

    def predict(self, obs, deterministic=True):
        """Predict action from observation (compatible with SB3 interface)."""
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.from_numpy(obs).float()

            # Add batch dimension if needed
            if len(obs.shape) == len(self.feature_extractor._observation_space.shape):
                obs = obs.unsqueeze(0)

            # Move to same device as model
            obs = obs.to(next(self.parameters()).device)

            action_logits = self.forward(obs)

            if self.is_discrete:
                if deterministic:
                    action = torch.argmax(action_logits, dim=-1)
                else:
                    probs = F.softmax(action_logits, dim=-1)
                    action = torch.multinomial(probs, 1).squeeze(-1)
            else:
                # For continuous actions, just use the network output
                action = action_logits

            # Remove batch dimension if it was added
            action = action.squeeze(0)

            return action.cpu().numpy(), None


def train_bc_agent(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    agent: BCAgent,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    value_loss_coef: float = 0.5,
    weight_decay: float = 1e-4,
) -> Tuple[Dict[str, List[float]], Optional[Tuple[float, float]]]:
    """Train the BC agent."""

    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Compute return normalization stats if training value head (for regression only)
    return_mean = None
    return_std = None
    if agent.train_value_head and not agent.use_categorical_value:
        all_returns = []
        for batch in train_loader:
            _, _, returns = batch
            all_returns.append(returns)
        all_returns = torch.cat(all_returns)
        return_mean = all_returns.mean()
        return_std = all_returns.std() + 1e-8
        print(f"Return normalization: mean={return_mean:.2f}, std={return_std:.2f}")
    elif agent.train_value_head and agent.use_categorical_value:
        print(f"Using categorical value head with {agent.value_bins} bins")
        print(f"Value range: [{agent.value_bin_centers.min().item():.2f}, {agent.value_bin_centers.max().item():.2f}]")

    # Compute class weights from training data
    class_weights = None
    if agent.is_discrete:
        # # Collect all actions from training set
        # all_actions = []
        # for batch in train_loader:
        #     if len(batch) == 3:  # with returns
        #         _, actions, _ = batch
        #     else:
        #         _, actions = batch
        #     all_actions.append(actions)
        # all_actions = torch.cat(all_actions)

        # # Compute inverse frequency weights
        # unique, counts = torch.unique(all_actions, return_counts=True)
        # class_weights = len(all_actions) / (len(unique) * counts.float())
        # class_weights = class_weights.to(device)
        # print(f"Class weights: {class_weights}")

        policy_criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        policy_criterion = nn.MSELoss()

    # Value criterion depends on whether we're using categorical or regression
    if agent.use_categorical_value:
        value_criterion = nn.CrossEntropyLoss()
    else:
        value_criterion = nn.HuberLoss(delta=1.0)

    history = {
        "train_loss": [], "val_loss": [],
        "train_policy_loss": [], "val_policy_loss": [],
        "train_value_loss": [], "val_value_loss": [],
        "train_acc": [], "val_acc": []
    }

    for epoch in range(num_epochs):
        # Training
        agent.train()
        train_loss = 0.0
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            if agent.train_value_head:
                obs, actions, returns = batch
                obs = obs.to(device)
                actions = actions.to(device)
                returns = returns.to(device)

                if agent.use_categorical_value:
                    # Discretize returns into bin indices for classification
                    value_targets = agent.discretize_values(returns)  # (B,) long tensor
                else:
                    # Normalize returns for regression
                    returns = returns.unsqueeze(-1)
                    value_targets = (returns - return_mean) / return_std
            else:
                obs, actions = batch
                obs = obs.to(device)
                actions = actions.to(device)

            optimizer.zero_grad()

            if agent.train_value_head:
                predictions, value_output = agent(obs, return_value=True)
                policy_loss = policy_criterion(predictions, actions) if not agent.is_discrete or predictions.dim() > 1 else policy_criterion(predictions, actions)
                value_loss = value_criterion(value_output, value_targets)
                loss = policy_loss + value_loss_coef * value_loss
            else:
                predictions = agent(obs)
                policy_loss = policy_criterion(predictions, actions) if not agent.is_discrete or predictions.dim() > 1 else policy_criterion(predictions, actions)
                loss = policy_loss
                value_loss = torch.tensor(0.0)

            if agent.is_discrete:
                train_correct += (predictions.argmax(dim=-1) == actions).sum().item()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_policy_loss += policy_loss.item()
            if agent.train_value_head:
                train_value_loss += value_loss.item()
            train_total += obs.size(0)

            postfix = {"loss": loss.item(), "policy": policy_loss.item()}
            if agent.train_value_head:
                postfix["value"] = value_loss.item()
            pbar.set_postfix(postfix)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_policy_loss = train_policy_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["train_policy_loss"].append(avg_train_policy_loss)

        log_str = f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} (Policy: {avg_train_policy_loss:.4f}"
        if agent.train_value_head:
            avg_train_value_loss = train_value_loss / len(train_loader)
            history["train_value_loss"].append(avg_train_value_loss)
            log_str += f", Value: {avg_train_value_loss:.4f}"
        log_str += ")"

        if agent.is_discrete:
            train_acc = train_correct / train_total
            history["train_acc"].append(train_acc)
            log_str += f", Acc = {train_acc:.4f}"

        print(log_str)

        # Validation
        if val_loader is not None:
            agent.eval()
            val_loss = 0.0
            val_policy_loss = 0.0
            val_value_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    if agent.train_value_head:
                        obs, actions, returns = batch
                        obs = obs.to(device)
                        actions = actions.to(device)
                        returns = returns.to(device)

                        if agent.use_categorical_value:
                            # Discretize returns into bin indices for classification
                            value_targets = agent.discretize_values(returns)
                        else:
                            # Normalize returns for regression
                            returns = returns.unsqueeze(-1)
                            value_targets = (returns - return_mean) / return_std
                    else:
                        obs, actions = batch
                        obs = obs.to(device)
                        actions = actions.to(device)

                    if agent.train_value_head:
                        predictions, value_output = agent(obs, return_value=True)
                        policy_loss = policy_criterion(predictions, actions)
                        value_loss = value_criterion(value_output, value_targets)
                        loss = policy_loss + value_loss_coef * value_loss
                    else:
                        predictions = agent(obs)
                        policy_loss = policy_criterion(predictions, actions)
                        loss = policy_loss
                        value_loss = torch.tensor(0.0)

                    if agent.is_discrete:
                        val_correct += (predictions.argmax(dim=-1) == actions).sum().item()

                    val_loss += loss.item()
                    val_policy_loss += policy_loss.item()
                    if agent.train_value_head:
                        val_value_loss += value_loss.item()
                    val_total += obs.size(0)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_policy_loss = val_policy_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)
            history["val_policy_loss"].append(avg_val_policy_loss)

            log_str = f"         Val Loss = {avg_val_loss:.4f} (Policy: {avg_val_policy_loss:.4f}"
            if agent.train_value_head:
                avg_val_value_loss = val_value_loss / len(val_loader)
                history["val_value_loss"].append(avg_val_value_loss)
                log_str += f", Value: {avg_val_value_loss:.4f}"
            log_str += ")"

            if agent.is_discrete:
                val_acc = val_correct / val_total
                history["val_acc"].append(val_acc)
                log_str += f", Acc = {val_acc:.4f}"

            print(log_str)

        # Step scheduler based on validation loss (or training loss if no validation)
        if val_loader is not None:
            scheduler.step(avg_val_loss)
        else:
            scheduler.step(avg_train_loss)

    # Return normalization stats along with history
    norm_stats = (return_mean.item(), return_std.item()) if return_mean is not None else None
    return history, norm_stats


def evaluate_bc_agent(agent: BCAgent, env: gym.Env, num_episodes: int = 10, save_debug: bool = False, debug_dir: Optional[Path] = None) -> Dict[str, float]:
    """Evaluate the BC agent in the environment."""

    agent.eval()
    episode_rewards = []
    episode_lengths = []

    for ep_idx in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # DEBUG: Save first eval observation
        if save_debug and ep_idx == 0:
            import matplotlib.pyplot as plt
            obs_img = obs.copy()
            if len(obs_img.shape) == 1:
                print("Warning: eval obs is flat vector, not image!")
            else:
                # Obs is (C, H, W) or (C*K, H, W)
                if obs_img.shape[0] > 3:
                    obs_img = obs_img[:3]  # Take first 3 channels
                obs_img = np.transpose(obs_img, (1, 2, 0))
                save_path = "debug_eval_sample.png" if debug_dir is None else str(debug_dir / "eval_sample.png")
                plt.imsave(save_path, np.clip(obs_img, 0, 1))
                print(f"Saved eval sample to {save_path}")

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }

    return results


def save_bc_model_for_ppo(
    agent: BCAgent,
    env: gym.Env,
    save_path: str,
    obs_type: str,
    features_dim: int = 256,
):
    """Save BC model in a format that can initialize PPO."""

    # Create a PPO model with the same architecture
    if obs_type == "image":
        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch=[],  # No additional layers after feature extractor
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=MLPFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch=[],  # No additional layers after feature extractor
        )

    ppo_model = PPO("MlpPolicy" if obs_type == "state" else "CnnPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    verbose=0)

    # Copy BC weights to PPO policy network
    bc_state_dict = agent.state_dict()
    ppo_state_dict = ppo_model.policy.state_dict()

    transferred = []

    # Map BC keys to PPO keys
    for bc_key in bc_state_dict.keys():
        if "feature_extractor" in bc_key:
            ppo_key = bc_key.replace("feature_extractor", "features_extractor")
            if ppo_key in ppo_state_dict:
                ppo_state_dict[ppo_key] = bc_state_dict[bc_key]
                transferred.append(f"{bc_key} -> {ppo_key}")
        elif "action_head" in bc_key:
            ppo_key = bc_key.replace("action_head", "action_net")
            if ppo_key in ppo_state_dict:
                ppo_state_dict[ppo_key] = bc_state_dict[bc_key]
                transferred.append(f"{bc_key} -> {ppo_key}")
        elif "value_head" in bc_key and agent.train_value_head:
            # Only transfer value head weights if not using categorical (PPO expects scalar)
            if not agent.use_categorical_value:
                ppo_key = bc_key.replace("value_head", "value_net")
                if ppo_key in ppo_state_dict:
                    ppo_state_dict[ppo_key] = bc_state_dict[bc_key]
                    transferred.append(f"{bc_key} -> {ppo_key}")
            else:
                # Categorical value head not compatible with PPO - skip transfer
                print(f"  Skipping {bc_key} (categorical value head not compatible with PPO's scalar value head)")

    ppo_model.policy.load_state_dict(ppo_state_dict, strict=False)
    ppo_model.save(save_path)

    print(f"Saved BC model for PPO initialization: {save_path}")
    print(f"Transferred weights:")
    for t in transferred:
        print(f"  {t}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Behavioral Cloning agent from HDF5 demonstrations"
    )

    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                        help="Path to HDF5 demonstration file")
    parser.add_argument("--obs-type", type=str, default="state", choices=["state", "image"],
                        help="Observation type (state or image)")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Number of frames to stack for temporal info (default: 1)")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Fraction of data for training (default: 0.9)")

    # Environment arguments
    parser.add_argument("--env-name", type=str, required=True,
                        help="Environment name for evaluation")
    parser.add_argument("--env-type", type=str, default="gym", choices=["gym", "robosuite"],
                        help="Environment type")

    # Training arguments
    parser.add_argument("--n-demos", type=int, default=100,
                        help="Number of demonstrations to use (default: 100)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay for regularization (default: 0.0)")
    parser.add_argument("--features-dim", type=int, default=256,
                        help="Feature dimension (default: 256)")

    # Critic pretraining arguments
    parser.add_argument("--pretrain-critic", action="store_true",
                        help="Pretrain value function (critic) from expert returns")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for return calculation (default: 0.99)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="Coefficient for value loss (default: 0.5)")
    parser.add_argument("--value-bins", type=int, default=0,
                        help="Number of bins for categorical value head (0=regression, >0=classification, default: 0)")
    parser.add_argument("--value-min", type=float, default=-300.0,
                        help="Minimum value for binning (default: -300.0)")
    parser.add_argument("--value-max", type=float, default=300.0,
                        help="Maximum value for binning (default: 300.0)")

    # Evaluation arguments
    parser.add_argument("--eval-episodes", type=int, default=30,
                        help="Number of episodes for evaluation (default: 10)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation after training")

    # Output arguments
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for trained model (.zip)")
    parser.add_argument("--plot-returns", action="store_true",
                        help="Plot ground truth episode returns from dataset")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto, default: auto)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility (default: 0)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_random_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    print("\n" + "="*60)
    print("Loading demonstrations...")
    if args.pretrain_critic:
        print("Critic pretraining ENABLED")
    print("="*60)

    n_demo = args.n_demos
    episodes = [f"demo_{i}" for i in range(n_demo)]
    print(episodes)

    # Only use frame stacking for images
    frame_stack = args.frame_stack if args.obs_type == "image" else 1

    full_dataset = DemonstrationDataset(
        args.data,
        obs_type=args.obs_type,
        episodes=episodes,
        compute_returns=args.pretrain_critic,
        gamma=args.gamma,
        frame_stack=frame_stack
    )

    # Split into train/val
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Use a generator with fixed seed for reproducible splits
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Worker init function for reproducibility
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        worker_init_fn=worker_init_fn, generator=generator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        worker_init_fn=worker_init_fn
    ) if val_size > 0 else None

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # Plot episode returns if requested
    if args.plot_returns:
        print("\n" + "="*60)
        print("Plotting ground truth episode returns...")
        print("="*60)

        # Determine output directory
        output_path = Path(args.output)
        if output_path.suffix in [".zip", ".pt"]:
            output_dir = output_path.with_suffix("")
        else:
            output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_path = output_dir / "episode_returns.png"
        plot_episode_returns(
            hdf5_path=args.data,
            n_demos=args.n_demos,
            gamma=args.gamma,
            save_path=str(plot_path),
        )

    # Create environment for evaluation
    if args.env_type == "gym":
        if args.obs_type == "image":
            # Get target image dimensions from dataset
            sample_obs = full_dataset.observations[0]  # (H, W, C)
            target_hw = (sample_obs.shape[0], sample_obs.shape[1])

            env = gym.make(args.env_name, render_mode="rgb_array")
            # BC training normalizes images to [0, 1] for better gradient flow
            env = ImageObservationWrapper(env, target_shape=target_hw, frame_stack=frame_stack, normalize=True)
        else:
            env = gym.make(args.env_name)

        # Seed the environment for reproducibility
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
    else:
        raise NotImplementedError("Robosuite environments not yet supported")

    # Create BC agent
    print("\n" + "="*60)
    print("Creating BC agent...")
    print("="*60)

    # For image observations, we can now use the wrapped env's observation space
    # But verify it matches the training data
    if args.obs_type == "image":
        # Get a stacked observation to determine the actual shape
        sample_stacked = full_dataset._get_stacked_obs(0)  # (H, W, C*K)
        # Dataset stores images as (H, W, C*K), need (C*K, H, W) for network
        expected_shape = (sample_stacked.shape[2], sample_stacked.shape[0], sample_stacked.shape[1])
        print(f"Expected observation shape from data: {expected_shape} (with {frame_stack} frame(s) stacked)")
        print(f"Environment observation space: {env.observation_space.shape}")

        # Use the training data shape to ensure consistency
        observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=expected_shape, dtype=np.float32
        )
    else:
        observation_space = env.observation_space

    agent = BCAgent(
        observation_space=observation_space,
        action_space=env.action_space,
        obs_type=args.obs_type,
        features_dim=args.features_dim,
        train_value_head=args.pretrain_critic,
        value_bins=args.value_bins,
        value_min=args.value_min,
        value_max=args.value_max,
    ).to(device)

    print(f"Agent architecture:")
    print(agent)

    # Train
    print("\n" + "="*60)
    print("Training BC agent...")
    print("="*60)

    # Create output directory first
    output_path = Path(args.output)
    if output_path.suffix in [".zip", ".pt"]:
        # If user provided a file path, create a directory with that name
        output_dir = output_path.with_suffix("")
    else:
        # If user provided a directory path, use it
        output_dir = output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Save sample training image if using images
    # if args.obs_type == "image":
    #     import matplotlib.pyplot as plt
    #     sample_obs, _ = train_dataset[0]
    #     sample_img = sample_obs.numpy()
    #     # Take first 3 channels if stacked
    #     if sample_img.shape[0] > 3:
    #         sample_img = sample_img[:3]
    #     sample_img = np.transpose(sample_img, (1, 2, 0))
    #     plt.imsave(str(output_dir / "train_sample.png"), np.clip(sample_img, 0, 1))
    #     print(f"Saved training sample to {output_dir / 'train_sample.png'}")

    history, norm_stats = train_bc_agent(
        train_loader=train_loader,
        val_loader=val_loader,
        agent=agent,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        value_loss_coef=args.value_loss_coef,
        weight_decay=args.weight_decay,
    )

    # Evaluate
    if not args.no_eval:
        print("\n" + "="*60)
        print("Evaluating BC agent...")
        print("="*60)

        results = evaluate_bc_agent(
            agent, env,
            num_episodes=args.eval_episodes,
            save_debug=(args.obs_type == "image"),
            debug_dir=output_dir
        )

        print(f"\nEvaluation Results ({args.eval_episodes} episodes):")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")

        # Save eval results
        eval_path = output_dir / "eval_results.txt"
        with open(eval_path, "w") as f:
            f.write(f"Evaluation Results ({args.eval_episodes} episodes)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
            f.write(f"Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}\n")
        print(f"Saved eval results to {eval_path}")

    # Save model
    print("\n" + "="*60)
    print("Saving model...")
    print("="*60)

    # Save for PPO initialization
    ppo_path = output_dir / "ppo_init.zip"
    save_bc_model_for_ppo(agent, env, str(ppo_path), args.obs_type, args.features_dim)

    # Save BC agent directly with all metadata
    bc_path = output_dir / "bc_agent.pt"
    checkpoint = {
        "model_state_dict": agent.state_dict(),
        "obs_type": args.obs_type,
        "features_dim": args.features_dim,
        "train_value_head": args.pretrain_critic,
        "value_bins": args.value_bins,
        "value_min": args.value_min,
        "value_max": args.value_max,
        "history": history,
    }

    # Add normalization stats if critic was pretrained (only for regression)
    if norm_stats is not None:
        checkpoint["return_mean"] = norm_stats[0]
        checkpoint["return_std"] = norm_stats[1]
        print(f"Saved return normalization: mean={norm_stats[0]:.2f}, std={norm_stats[1]:.2f}")
    elif args.pretrain_critic and args.value_bins > 0:
        print(f"Using categorical value head - no normalization stats to save")

    torch.save(checkpoint, str(bc_path))
    print(f"Saved BC agent: {bc_path}")

    # Save training config for reproducibility
    config_path = output_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write("Training Configuration\n")
        f.write("=" * 60 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        if norm_stats is not None:
            f.write(f"\nReturn Normalization:\n")
            f.write(f"  mean: {norm_stats[0]:.6f}\n")
            f.write(f"  std: {norm_stats[1]:.6f}\n")
    print(f"Saved config: {config_path}")

    env.close()
    print(f"\nTraining complete! All outputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
