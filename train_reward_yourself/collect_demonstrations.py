"""
Collect expert demonstrations from a trained RL agent and save in robomimic HDF5 format.

This script loads a trained RL agent (PPO/SAC) and collects expert demonstrations
by running the policy in the environment. The demonstrations are saved in HDF5
format compatible with robomimic for use in imitation learning/behavioral cloning.

Usage:
    python collect_expert_demos.py \
        --model-path ppo_lunarlander_v3_500000.zip \
        --env-type gym \
        --env-name LunarLander-v3 \
        --algo ppo \
        --n-demos 50 \
        --output demos_lunarlander.hdf5
"""

import os
import argparse
import h5py
import numpy as np
import gymnasium as gym
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from train_reward_yourself.env_utils import EnvConfig


def load_model(model_path: str, algo: str, env):
    """Load a trained RL model."""
    print(f"\nLoading {algo.upper()} model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if algo == 'ppo':
        model = PPO.load(model_path, env=env)
    elif algo == 'sac':
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    print(f"Model loaded successfully!")
    return model


def collect_trajectory(
    model,
    env,
    deterministic: bool = True,
    max_steps: int = 1000,
    render_env=None,
    collect_images: bool = False,
    image_size: Tuple[int, int] = (84, 84),
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[bool], Dict, Optional[List[np.ndarray]]]:
    """
    Collect a single trajectory from the environment.

    Args:
        model: Trained RL model
        env: Environment (for policy execution with state observations)
        deterministic: Whether to use deterministic actions
        max_steps: Maximum steps per episode
        render_env: DEPRECATED - no longer used
        collect_images: Whether to collect rendered images
        image_size: Tuple of (height, width) for resized images

    Returns:
        Tuple of (observations, actions, rewards, dones, info, images)
    """
    import cv2

    observations = []
    actions = []
    rewards = []
    dones = []
    images = [] if collect_images else None

    obs, _ = env.reset()
    observations.append(obs)

    # Collect initial image by rendering from the same environment
    if collect_images:
        img = env.render()
        if img is not None:
            # Resize image
            img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
            images.append(img)

    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        # Collect rendered image from the SAME environment
        if collect_images:
            img = env.render()
            if img is not None:
                # Resize image
                img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
                images.append(img)

        step += 1

    # Remove the last observation (since we have n+1 observations for n actions)
    observations = observations[:-1]
    if images is not None and len(images) > 0:
        images = images[:-1]  # Also remove last image to match observations

    return observations, actions, rewards, dones, info, images


def collect_demonstrations(
    model,
    env,
    n_demos: int,
    deterministic: bool = True,
    min_reward: float = None,
    max_attempts_per_demo: int = 10,
    collect_images: bool = False,
    image_size: Tuple[int, int] = (84, 84),
) -> List[Dict]:
    """
    Collect multiple expert demonstrations.

    Args:
        model: Trained RL model
        env: Environment (must support render_mode='rgb_array' if collect_images=True)
        n_demos: Number of demonstrations to collect
        deterministic: Whether to use deterministic actions
        min_reward: Minimum reward threshold for accepting a demo (None = accept all)
        max_attempts_per_demo: Maximum attempts to collect each demo with min_reward
        collect_images: Whether to collect rendered images
        image_size: Tuple of (height, width) for resized images

    Returns:
        List of demonstration dictionaries
    """
    demonstrations = []

    print(f"\n{'='*60}")
    print(f"Collecting {n_demos} expert demonstrations...")
    print(f"Deterministic: {deterministic}")
    if min_reward is not None:
        print(f"Minimum reward threshold: {min_reward}")
    if collect_images:
        print(f"Collecting images: Yes")
        print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"{'='*60}\n")

    pbar = tqdm(total=n_demos, desc="Collecting demos")

    while len(demonstrations) < n_demos:
        attempts = 0
        demo_collected = False

        while not demo_collected and attempts < max_attempts_per_demo:
            observations, actions, rewards, dones, info, images = collect_trajectory(
                model, env, deterministic=deterministic,
                collect_images=collect_images, image_size=image_size
            )

            total_reward = sum(rewards)

            # Check if demo meets minimum reward threshold
            if min_reward is None or total_reward >= min_reward:
                demo = {
                    'observations': np.array(observations),
                    'actions': np.array(actions),
                    'rewards': np.array(rewards),
                    'dones': np.array(dones),
                    'total_reward': total_reward,
                    'length': len(actions),
                    'success': info.get('is_success', False) or info.get('success', False),
                }
                if images is not None:
                    demo['images'] = np.array(images)
                demonstrations.append(demo)
                demo_collected = True
                pbar.update(1)
            else:
                attempts += 1

        if not demo_collected:
            print(f"\nWarning: Could not collect demo with reward >= {min_reward} "
                  f"after {max_attempts_per_demo} attempts. Accepting last attempt.")
            demo = {
                'observations': np.array(observations),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'dones': np.array(dones),
                'total_reward': total_reward,
                'length': len(actions),
                'success': info.get('is_success', False) or info.get('success', False),
            }
            if images is not None:
                demo['images'] = np.array(images)
            demonstrations.append(demo)
            pbar.update(1)

    pbar.close()

    # Print statistics
    rewards_array = np.array([d['total_reward'] for d in demonstrations])
    lengths_array = np.array([d['length'] for d in demonstrations])

    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Number of demos: {len(demonstrations)}")
    print(f"\nRewards:")
    print(f"  Mean: {rewards_array.mean():.2f}")
    print(f"  Std:  {rewards_array.std():.2f}")
    print(f"  Min:  {rewards_array.min():.2f}")
    print(f"  Max:  {rewards_array.max():.2f}")
    print(f"\nEpisode Lengths:")
    print(f"  Mean: {lengths_array.mean():.1f}")
    print(f"  Std:  {lengths_array.std():.1f}")
    print(f"  Min:  {lengths_array.min()}")
    print(f"  Max:  {lengths_array.max()}")

    if any(d['success'] for d in demonstrations):
        success_rate = sum(d['success'] for d in demonstrations) / len(demonstrations) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")

    print(f"{'='*60}\n")

    return demonstrations


def save_demonstrations_hdf5(
    demonstrations: List[Dict],
    output_path: str,
    env_name: str,
    env_config: dict,
):
    """
    Save demonstrations in robomimic-compatible HDF5 format.

    The structure follows robomimic convention:
    - data/
      - demo_0/
        - obs/
          - state: [T, obs_dim]
        - actions: [T, action_dim]
        - rewards: [T]
        - dones: [T]
      - demo_1/
        ...
      - demo_N/
        ...
      - total: N (total number of demos)
      - env: environment name

    Args:
        demonstrations: List of demonstration dictionaries
        output_path: Path to save HDF5 file
        env_name: Name of the environment
        env_config: Environment configuration dictionary
    """
    print(f"\nSaving demonstrations to: {output_path}")

    with h5py.File(output_path, 'w') as f:
        # Create data group
        data_grp = f.create_group('data')

        # Track total samples across all demos
        total_samples = 0

        # Save each demonstration
        for i, demo in enumerate(tqdm(demonstrations, desc="Saving demos")):
            demo_grp = data_grp.create_group(f'demo_{i}')

            # Create obs group and save observations
            obs_grp = demo_grp.create_group('obs')
            obs_grp.create_dataset('state', data=demo['observations'])

            # Save images if available
            if 'images' in demo:
                obs_grp.create_dataset('images', data=demo['images'], compression='gzip')

            # Save actions, rewards, dones
            demo_grp.create_dataset('actions', data=demo['actions'])
            demo_grp.create_dataset('rewards', data=demo['rewards'])
            demo_grp.create_dataset('dones', data=demo['dones'])

            # Save metadata for this demo
            demo_grp.attrs['num_samples'] = demo['length']
            demo_grp.attrs['total_reward'] = demo['total_reward']
            total_samples += demo['length']

        # Save global metadata
        data_grp.attrs['total'] = len(demonstrations)
        data_grp.attrs['num_samples'] = total_samples
        data_grp.attrs['env'] = env_name
        data_grp.attrs['env_config'] = str(env_config)

    print(f"Successfully saved {len(demonstrations)} demonstrations!")

    # Print file info
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nFile info:")
    print(f"  Total transitions: {total_samples}")
    print(f"  File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Collect expert demonstrations from trained RL agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 50 demos from PPO LunarLander agent
  python collect_expert_demos.py \\
      --model-path ppo_lunarlander_v3_500000.zip \\
      --env-type gym \\
      --env-name LunarLander-v3 \\
      --algo ppo \\
      --n-demos 50 \\
      --output demos_lunarlander.hdf5

  # Collect only high-reward demos (reward >= 200)
  python collect_expert_demos.py \\
      --model-path ppo_lunarlander_v3_500000.zip \\
      --env-type gym \\
      --env-name LunarLander-v3 \\
      --algo ppo \\
      --n-demos 100 \\
      --min-reward 200 \\
      --output demos_lunarlander_expert.hdf5

  # Collect from SAC agent
  python collect_expert_demos.py \\
      --model-path sac_lunarlander_v3_500000.zip \\
      --env-type gym \\
      --env-name LunarLander-v3 \\
      --algo sac \\
      --n-demos 50

  # Collect with rendered images (84x84)
  python collect_expert_demos.py \\
      --model-path ppo_lunarlander_v3_500000.zip \\
      --env-type gym \\
      --env-name LunarLander-v3 \\
      --algo ppo \\
      --n-demos 50 \\
      --collect-images \\
      --render-image-size 84
        """
    )

    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model (.zip file)')
    parser.add_argument('--algo', type=str, required=True, choices=['ppo', 'sac'],
                        help='Algorithm used to train the model')
    parser.add_argument('--vecnormalize-path', type=str, default=None,
                        help='Path to VecNormalize stats (.pkl file) if used during training')

    # Environment arguments
    parser.add_argument('--env-type', type=str, required=True, choices=['gym', 'robosuite'],
                        help='Type of environment')
    parser.add_argument('--env-name', type=str, required=True,
                        help='Name of the environment')
    parser.add_argument('--control-type', type=str, default='OSC_POSE',
                        help='Controller type for robosuite (default: OSC_POSE)')

    # Observation arguments (must match training configuration)
    parser.add_argument('--use-image-obs', action='store_true',
                        help='Use image observations (must match training config)')
    parser.add_argument('--image-size', type=int, default=84,
                        help='Image size (must match training config, default: 84)')
    parser.add_argument('--frame-stack', type=int, default=4,
                        help='Frame stack size (must match training config, default: 4)')

    # Collection arguments
    parser.add_argument('--n-demos', type=int, required=True,
                        help='Number of demonstrations to collect')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic actions (default: True)')
    parser.add_argument('--stochastic', dest='deterministic', action='store_false',
                        help='Use stochastic actions')
    parser.add_argument('--min-reward', type=float, default=None,
                        help='Minimum reward threshold for accepting demos (default: None)')
    parser.add_argument('--max-attempts', type=int, default=10,
                        help='Maximum attempts per demo when using --min-reward (default: 10)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')

    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output HDF5 file path (default: demos_{env_name}.hdf5)')

    # Image collection arguments
    parser.add_argument('--collect-images', type=bool, default=True,
                        help='Collect rendered images along with observations')
    parser.add_argument('--render-image-size', type=int, default=84,
                        help='Size for square rendered images (default: 84 -> 84x84)')

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output is None:
        args.output = f"demos_{args.env_name.lower().replace('-', '_')}.hdf5"

    # Create environment configuration
    env_config = EnvConfig(
        env_type=args.env_type,
        env_name=args.env_name,
        control_type=args.control_type if args.env_type == 'robosuite' else None,
        use_image_obs=args.use_image_obs,
        image_size=args.image_size,
        frame_stack=args.frame_stack if args.use_image_obs else 1,
        seed=args.seed,
    )

    # Print configuration
    print("\n" + "="*60)
    print("Expert Demonstration Collection Configuration")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Environment type: {args.env_type}")
    print(f"Environment name: {args.env_name}")
    if args.env_type == 'robosuite':
        print(f"Controller type: {args.control_type}")
    print(f"Observation type: {'Image' if args.use_image_obs else 'State'}")
    if args.use_image_obs:
        print(f"Image size: {args.image_size}x{args.image_size}")
        print(f"Frame stack: {args.frame_stack}")
    print(f"Number of demos: {args.n_demos}")
    print(f"Deterministic: {args.deterministic}")
    if args.min_reward is not None:
        print(f"Minimum reward: {args.min_reward}")
        print(f"Max attempts per demo: {args.max_attempts}")
    if args.collect_images:
        print(f"Collect images: Yes")
        print(f"Image size: {args.render_image_size}x{args.render_image_size}")
    print(f"Output file: {args.output}")
    print("="*60 + "\n")

    # Create environment with rendering enabled if collecting images
    render_mode = 'rgb_array' if args.collect_images else None
    env = env_config.create_single_env(render_mode=render_mode)

    # Load VecNormalize wrapper if provided
    if args.vecnormalize_path is not None:
        print(f"Loading VecNormalize stats from: {args.vecnormalize_path}")
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(args.vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
        print("VecNormalize stats loaded!")

    # Load model
    model = load_model(args.model_path, args.algo, env)

    # Collect demonstrations
    image_size = (args.render_image_size, args.render_image_size)
    demonstrations = collect_demonstrations(
        model=model,
        env=env,
        n_demos=args.n_demos,
        deterministic=args.deterministic,
        min_reward=args.min_reward,
        max_attempts_per_demo=args.max_attempts,
        collect_images=args.collect_images,
        image_size=image_size,
    )

    # Save demonstrations
    save_demonstrations_hdf5(
        demonstrations=demonstrations,
        output_path=args.output,
        env_name=args.env_name,
        env_config=env_config.to_dict(),
    )

    # Close environment
    env.close()
    print("\nCollection complete!")


if __name__ == '__main__':
    main()
