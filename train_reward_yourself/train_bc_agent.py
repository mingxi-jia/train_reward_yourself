#!/usr/bin/env python3
"""
Train a Behavioral Cloning agent from HDF5 demonstrations.

Supports both state-to-action (MLP) and image-to-action (CNN) mappings.
The trained model can be used to initialize PPO agents.

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
"""

import argparse
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

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class DemonstrationDataset(Dataset):
    """PyTorch dataset for loading demonstrations from HDF5 file."""

    def __init__(self, hdf5_path: str, obs_type: str = "state", episodes: Optional[List[str]] = None):
        """
        Args:
            hdf5_path: Path to HDF5 file
            obs_type: Type of observation ("state" or "image")
            episodes: List of episode keys to load. If None, load all.
        """
        self.hdf5_path = hdf5_path
        self.obs_type = obs_type
        self.observations = []
        self.actions = []

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

                self.observations.append(obs)
                self.actions.append(actions)

        # Concatenate all episodes
        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)

        print(f"Loaded {len(self.observations)} transitions")
        print(f"Observation shape: {self.observations.shape}")
        print(f"Action shape: {self.actions.shape}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        action = self.actions[idx]

        # Convert to torch tensors
        if self.obs_type == "image":
            # Image: (H, W, C) -> (C, H, W)
            obs = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
        else:
            obs = torch.from_numpy(obs).float()

        # Handle both discrete and continuous actions
        if len(action.shape) == 0 or (len(action.shape) == 1 and action.shape[0] == 1):
            # Discrete action
            action = torch.tensor(action, dtype=torch.long)
        else:
            # Continuous action
            action = torch.from_numpy(action).float()

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
    ):
        super().__init__()

        self.obs_type = obs_type
        self.action_space = action_space
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

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

    def forward(self, obs):
        features = self.feature_extractor(obs)
        action_logits = self.action_head(features)
        return action_logits

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
) -> Dict[str, List[float]]:
    """Train the BC agent."""

    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    if agent.is_discrete:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Training
        agent.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for obs, actions in pbar:
            obs, actions = obs.to(device), actions.to(device)

            optimizer.zero_grad()
            predictions = agent(obs)

            if agent.is_discrete:
                loss = criterion(predictions, actions)
                train_correct += (predictions.argmax(dim=-1) == actions).sum().item()
            else:
                loss = criterion(predictions, actions)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += obs.size(0)

            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        if agent.is_discrete:
            train_acc = train_correct / train_total
            history["train_acc"].append(train_acc)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # Validation
        if val_loader is not None:
            agent.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for obs, actions in val_loader:
                    obs, actions = obs.to(device), actions.to(device)
                    predictions = agent(obs)

                    if agent.is_discrete:
                        loss = criterion(predictions, actions)
                        val_correct += (predictions.argmax(dim=-1) == actions).sum().item()
                    else:
                        loss = criterion(predictions, actions)

                    val_loss += loss.item()
                    val_total += obs.size(0)

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            if agent.is_discrete:
                val_acc = val_correct / val_total
                history["val_acc"].append(val_acc)
                print(f"         Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")
            else:
                print(f"         Val Loss = {avg_val_loss:.4f}")

    return history


def evaluate_bc_agent(agent: BCAgent, env: gym.Env, num_episodes: int = 10) -> Dict[str, float]:
    """Evaluate the BC agent in the environment."""

    agent.eval()
    episode_rewards = []
    episode_lengths = []

    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

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
    # The feature extractor and action head can be transferred
    bc_state_dict = agent.state_dict()
    ppo_state_dict = ppo_model.policy.state_dict()

    # Map BC keys to PPO keys
    for bc_key in bc_state_dict.keys():
        if "feature_extractor" in bc_key:
            ppo_key = bc_key.replace("feature_extractor", "features_extractor")
            if ppo_key in ppo_state_dict:
                ppo_state_dict[ppo_key] = bc_state_dict[bc_key]
        elif "action_head" in bc_key:
            # Map to action network (pi_features_extractor is usually the policy head)
            ppo_key = bc_key.replace("action_head", "action_net")
            if ppo_key in ppo_state_dict:
                ppo_state_dict[ppo_key] = bc_state_dict[bc_key]

    ppo_model.policy.load_state_dict(ppo_state_dict, strict=False)
    ppo_model.save(save_path)

    print(f"Saved BC model for PPO initialization: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Behavioral Cloning agent from HDF5 demonstrations"
    )

    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                        help="Path to HDF5 demonstration file")
    parser.add_argument("--obs-type", type=str, default="state", choices=["state", "image"],
                        help="Observation type (state or image)")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Fraction of data for training (default: 0.9)")

    # Environment arguments
    parser.add_argument("--env-name", type=str, required=True,
                        help="Environment name for evaluation")
    parser.add_argument("--env-type", type=str, default="gym", choices=["gym", "robosuite"],
                        help="Environment type")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--features-dim", type=int, default=256,
                        help="Feature dimension (default: 256)")

    # Evaluation arguments
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of episodes for evaluation (default: 10)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation after training")

    # Output arguments
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for trained model (.zip)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto, default: auto)")

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    print("\n" + "="*60)
    print("Loading demonstrations...")
    print("="*60)

    full_dataset = DemonstrationDataset(args.data, obs_type=args.obs_type)

    # Split into train/val
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_size > 0 else None

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create environment for evaluation
    if args.env_type == "gym":
        if args.obs_type == "image":
            env = gym.make(args.env_name, render_mode="rgb_array")
        else:
            env = gym.make(args.env_name)
    else:
        raise NotImplementedError("Robosuite environments not yet supported")

    # Create BC agent
    print("\n" + "="*60)
    print("Creating BC agent...")
    print("="*60)

    agent = BCAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        obs_type=args.obs_type,
        features_dim=args.features_dim,
    ).to(device)

    print(f"Agent architecture:")
    print(agent)

    # Train
    print("\n" + "="*60)
    print("Training BC agent...")
    print("="*60)

    history = train_bc_agent(
        train_loader=train_loader,
        val_loader=val_loader,
        agent=agent,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
    )

    # Evaluate
    if not args.no_eval:
        print("\n" + "="*60)
        print("Evaluating BC agent...")
        print("="*60)

        results = evaluate_bc_agent(agent, env, num_episodes=args.eval_episodes)

        print(f"\nEvaluation Results ({args.eval_episodes} episodes):")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")

    # Save model
    print("\n" + "="*60)
    print("Saving model...")
    print("="*60)

    # Save for PPO initialization
    save_bc_model_for_ppo(agent, env, args.output, args.obs_type, args.features_dim)

    # Also save BC agent directly
    output_path = Path(args.output)
    if output_path.suffix == ".zip":
        bc_direct_path = output_path.with_suffix("").with_suffix(".pt")
    else:
        bc_direct_path = output_path.with_name(output_path.name + "_bc.pt")

    torch.save({
        "model_state_dict": agent.state_dict(),
        "obs_type": args.obs_type,
        "features_dim": args.features_dim,
        "history": history,
    }, str(bc_direct_path))
    print(f"Saved BC agent: {bc_direct_path}")

    env.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
