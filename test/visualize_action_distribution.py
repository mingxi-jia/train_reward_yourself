"""
Visualize action distribution from robomimic HDF5 demonstration files.

This script loads actions from demonstration data and creates various visualizations
to understand the action space characteristics.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List


def load_actions_from_hdf5(hdf5_path: str, max_episodes: int = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load actions from robomimic hdf5 format.

    Args:
        hdf5_path: Path to the hdf5 demonstration file
        max_episodes: Maximum number of episodes to load (None = load all)

    Returns:
        Tuple of (actions array, demo_keys list)
    """
    print(f"\nLoading actions from: {hdf5_path}")

    actions_list = []

    with h5py.File(hdf5_path, 'r') as f:
        # Get list of demonstration episodes
        demo_keys = [key for key in f['data'].keys() if key.startswith('demo_')]

        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]

        print(f"Loading actions from {len(demo_keys)} demonstration episodes...")

        for demo_key in demo_keys:
            demo_group = f['data'][demo_key]

            # Load actions
            episode_actions = np.array(demo_group['actions'])
            actions_list.append(episode_actions)

    # Concatenate all episodes
    actions = np.concatenate(actions_list, axis=0)

    print(f"Loaded {len(actions)} action samples from {len(demo_keys)} episodes")
    print(f"Action shape: {actions.shape}")
    print(f"Action dimensions: {actions.shape[1]}")

    return actions, demo_keys


def compute_action_statistics(actions: np.ndarray) -> dict:
    """Compute statistics for each action dimension."""
    stats = {
        'mean': actions.mean(axis=0),
        'std': actions.std(axis=0),
        'min': actions.min(axis=0),
        'max': actions.max(axis=0),
        'median': np.median(actions, axis=0),
        'q25': np.percentile(actions, 25, axis=0),
        'q75': np.percentile(actions, 75, axis=0),
    }
    return stats


def visualize_action_distribution(actions: np.ndarray, save_path: str = None):
    """
    Create comprehensive visualizations of action distribution.

    Args:
        actions: Array of actions (num_samples, action_dim)
        save_path: Optional path to save the figure
    """
    action_dim = actions.shape[1]

    # Compute statistics
    stats = compute_action_statistics(actions)

    # Print statistics
    print("\n" + "="*60)
    print("Action Statistics:")
    print("="*60)
    for i in range(action_dim):
        print(f"\nDimension {i}:")
        print(f"  Mean:   {stats['mean'][i]:8.4f}")
        print(f"  Std:    {stats['std'][i]:8.4f}")
        print(f"  Min:    {stats['min'][i]:8.4f}")
        print(f"  Max:    {stats['max'][i]:8.4f}")
        print(f"  Median: {stats['median'][i]:8.4f}")
        print(f"  Q25:    {stats['q25'][i]:8.4f}")
        print(f"  Q75:    {stats['q75'][i]:8.4f}")
    print("="*60)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Histograms for each action dimension
    print("\nCreating histograms...")
    for i in range(action_dim):
        ax = plt.subplot(4, action_dim, i + 1)
        ax.hist(actions[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(stats['mean'][i], color='r', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(stats['median'][i], color='g', linestyle='--', linewidth=2, label='Median')
        ax.set_title(f'Action Dim {i}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Box plots
    print("Creating box plots...")
    ax = plt.subplot(4, 1, 2)
    ax.boxplot([actions[:, i] for i in range(action_dim)],
                labels=[f'Dim {i}' for i in range(action_dim)],
                showmeans=True)
    ax.set_title('Action Distribution (Box Plots)')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

    # 3. Time series plot (sample trajectory)
    print("Creating time series plot...")
    sample_length = min(500, len(actions))
    ax = plt.subplot(4, 1, 3)
    for i in range(action_dim):
        ax.plot(actions[:sample_length, i], alpha=0.7, label=f'Dim {i}')
    ax.set_title(f'Action Time Series (First {sample_length} Steps)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right', ncol=action_dim)
    ax.grid(True, alpha=0.3)

    # 4. Correlation heatmap (if action_dim is reasonable)
    if action_dim <= 10:
        print("Creating correlation heatmap...")
        ax = plt.subplot(4, 1, 4)
        correlation_matrix = np.corrcoef(actions.T)
        im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(action_dim))
        ax.set_yticks(range(action_dim))
        ax.set_xticklabels([f'Dim {i}' for i in range(action_dim)])
        ax.set_yticklabels([f'Dim {i}' for i in range(action_dim)])
        ax.set_title('Action Correlation Matrix')

        # Add correlation values as text
        for i in range(action_dim):
            for j in range(action_dim):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax, label='Correlation')
    else:
        # For high-dimensional actions, show action magnitudes over time
        print("Creating action magnitude plot...")
        ax = plt.subplot(4, 1, 4)
        action_magnitudes = np.linalg.norm(actions, axis=1)
        ax.plot(action_magnitudes[:sample_length], alpha=0.7)
        ax.set_title(f'Action Magnitude Over Time (First {sample_length} Steps)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('L2 Norm')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        print("\nDisplaying figure...")
        plt.show()

    plt.close()


def visualize_per_dimension_details(actions: np.ndarray, save_dir: str = None):
    """
    Create detailed visualizations for each action dimension separately.

    Args:
        actions: Array of actions (num_samples, action_dim)
        save_dir: Optional directory to save individual figures
    """
    action_dim = actions.shape[1]

    for i in range(action_dim):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Action Dimension {i} - Detailed Analysis', fontsize=16)

        action_values = actions[:, i]

        # Histogram with KDE
        ax = axes[0, 0]
        ax.hist(action_values, bins=50, alpha=0.7, edgecolor='black', density=True, label='Histogram')
        # Add KDE estimate
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(action_values)
        x_range = np.linspace(action_values.min(), action_values.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        ax.set_title('Distribution with KDE')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Q-Q plot
        ax = axes[0, 1]
        from scipy.stats import probplot
        probplot(action_values, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normal Distribution)')
        ax.grid(True, alpha=0.3)

        # Time series
        ax = axes[1, 0]
        sample_length = min(1000, len(action_values))
        ax.plot(action_values[:sample_length], alpha=0.7)
        ax.set_title(f'Time Series (First {sample_length} Steps)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

        # Autocorrelation
        ax = axes[1, 1]
        from numpy import correlate
        max_lag = min(100, len(action_values) // 2)
        lags = range(max_lag)
        autocorr = [np.corrcoef(action_values[:-lag or None], action_values[lag:])[0, 1] if lag > 0
                   else 1.0 for lag in lags]
        ax.plot(lags, autocorr)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_title('Autocorrelation')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'action_dim_{i}_detailed.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize action distribution from HDF5 demonstrations')
    parser.add_argument('--demo-path', type=str, required=True,
                       help='Path to demonstration hdf5 file')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='Maximum number of episodes to load (default: all)')
    parser.add_argument('--save-path', type=str, default='action_distribution.png',
                       help='Path to save the main visualization (default: action_distribution.png)')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed per-dimension visualizations')
    parser.add_argument('--detailed-save-dir', type=str, default='action_details',
                       help='Directory to save detailed visualizations (default: action_details)')
    parser.add_argument('--no-save', action='store_true',
                       help='Display plots instead of saving them')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.demo_path):
        print(f"Error: File not found: {args.demo_path}")
        exit(1)

    print("="*60)
    print("Action Distribution Visualization")
    print("="*60)
    print(f"Demo file: {args.demo_path}")
    print(f"Max episodes: {args.max_episodes if args.max_episodes else 'All'}")
    print("="*60)

    # Load actions
    actions, demo_keys = load_actions_from_hdf5(args.demo_path, args.max_episodes)

    # Create main visualization
    save_path = None if args.no_save else args.save_path
    visualize_action_distribution(actions, save_path=save_path)

    # Create detailed visualizations if requested
    if args.detailed:
        print("\n" + "="*60)
        print("Creating detailed per-dimension visualizations...")
        print("="*60)
        save_dir = None if args.no_save else args.detailed_save_dir
        visualize_per_dimension_details(actions, save_dir=save_dir)
        print("Detailed visualizations complete!")

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
