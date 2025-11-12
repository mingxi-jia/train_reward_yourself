import torch
import robosuite as suite
import numpy as np
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_composite_controller_config
# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_controller_config(use_osc_position=True):
    controller_config = load_composite_controller_config(controller="BASIC")
    if use_osc_position:
        new_config = {
                                "type": "OSC_POSITION",
                                "input_max": 1,
                                "input_min": -1,
                                "output_max": [0.05, 0.05, 0.05],
                                "output_min": [-0.05, -0.05, -0.05],
                                "kp": 150,
                                "damping_ratio": 1,
                                "impedance_mode": "fixed",
                                "kp_limits": [0, 300],
                                "damping_ratio_limits": [0, 10],
                                "position_limits": None,
                                "control_delta": True,
                                "interpolation": None,
                                "ramp_ratio": 0.2,
                                'gripper': {'type': 'GRIP'}
                                }
        controller_config['body_parts']['right'] = new_config
    # delete other keys if exist
    for key in list(controller_config['body_parts'].keys()):
        if key != 'right':
            del controller_config['body_parts'][key]
    return controller_config

def create_eval_env(render=True, camera_names=None):
    """
    Create a single evaluation environment for the Lift task.

    Args:
        render: Whether to enable on-screen rendering
        camera_names: List of camera names for observation (if using camera obs)

    Returns:
        Wrapped Gym environment
    """
    # Load controller config for operational space control
    controller_config = get_controller_config(use_osc_position=False)


    eval_config = {
        "env_name": "Lift",
        "robots": "Panda",
        "controller_configs": controller_config,  # Use same controller config
        "has_renderer": render,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": True,
        "control_freq": 20,
        "horizon": 200,
        "reward_shaping": True,
    }

    if camera_names:
        eval_config["camera_names"] = camera_names

    env_robosuite = suite.make(**eval_config)
    env = GymWrapper(env_robosuite)
    return env


def evaluate_model(model_path, n_episodes=10, render=True, verbose=True):
    """
    Evaluate a trained PPO model on the Lift task.

    Args:
        model_path: Path to the saved model (without .zip extension)
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        verbose: Whether to print episode details

    Returns:
        Dictionary containing evaluation metrics
    """
    # Check for GPU
    if torch.cuda.is_available():
        print("GPU is available! Using CUDA for evaluation...")
        device = "cuda"
    else:
        print("GPU not available, using CPU...")
        device = "cpu"

    # Create evaluation environment
    print(f"Creating evaluation environment (render={render})...")
    eval_env = create_eval_env(render=render)

    # Load the trained model
    print(f"Loading model from '{model_path}'...")
    model = PPO.load(model_path, env=eval_env, device=device)
    print(f"Model loaded successfully on device: {model.device}")

    # Evaluate using SB3's built-in function
    print(f"\nEvaluating for {n_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        render=render,
        deterministic=True
    )

    # Additional detailed evaluation
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print("\n" + "="*60)
    print("Running detailed evaluation...")
    print("="*60)

    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info  = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check if task was successful (you may need to adjust this based on your reward structure)
        # Typically, a high reward indicates success in the Lift task
        is_success = episode_reward > 100  # Adjust threshold as needed
        if is_success:
            success_count += 1

        if verbose:
            success_str = " SUCCESS" if is_success else " FAILED"
            print(f"Episode {episode+1:2d}: Reward = {episode_reward:7.2f}, "
                  f"Length = {episode_length:3d}, {success_str}")

    eval_env.close()

    # Compile results
    results = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'success_rate': success_count / n_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Episodes:      {n_episodes}")
    print(f"Mean Reward:   {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Min Reward:    {results['min_reward']:.2f}")
    print(f"Max Reward:    {results['max_reward']:.2f}")
    print(f"Mean Length:   {results['mean_length']:.1f} steps")
    print(f"Success Rate:  {results['success_rate']*100:.1f}% ({success_count}/{n_episodes})")
    print("="*60 + "\n")

    return results


def rollout_episodes(model_path, n_episodes=5, render=True, save_video=False):
    """
    Rollout episodes with the trained model for visualization.

    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to rollout
        render: Whether to render on screen
        save_video: Whether to save video (requires additional setup)
    """
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment
    eval_env = create_eval_env(render=render)

    # Load model
    print(f"Loading model from '{model_path}'...")
    model = PPO.load(model_path, env=eval_env, device=device)

    print(f"\nRolling out {n_episodes} episodes...")
    print("Close the render window to continue to next episode.\n")

    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"--- Episode {episode+1}/{n_episodes} ---")

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        print(f"Episode {episode+1} finished: Reward = {total_reward:.2f}, Steps = {steps}")

    eval_env.close()
    print("\nRollout complete!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained RL policy on Robosuite Lift task')
    parser.add_argument('--model', type=str, default='ppo_lift_model',
                        help='Path to saved model (without .zip extension)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (faster evaluation)')
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'rollout'],
                        help='Evaluation mode: eval (statistical) or rollout (visualization)')

    args = parser.parse_args()

    render = not args.no_render

    if args.mode == 'eval':
        results = evaluate_model(
            model_path=args.model,
            n_episodes=args.episodes,
            render=render,
            verbose=True
        )
    else:  # rollout mode
        rollout_episodes(
            model_path=args.model,
            n_episodes=args.episodes,
            render=render
        )
