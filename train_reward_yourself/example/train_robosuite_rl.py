import torch
import robosuite as suite
import os
import warnings
import argparse
# --- NEW: Import SubprocVecEnv for parallel environments ---
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from train_reward_yourself.utils import GymWrapper
# --- FIX: Handle controller imports for different robosuite versions ---
# from robosuite.controllers import load_composite_controller_config
from train_reward_yourself.utils import get_controller_config, make_robosuite_env
# --- Explicitly import the controller class to register it ---
try:
    from robosuite.controllers.operational_space_controller import OperationalSpaceController
except ImportError:
    pass # Ignore if it fails, since we're not using it right now
# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# --- 0. Check for GPU ---
if torch.cuda.is_available():
    print("GPU is available! Using CUDA...")
    device = "cuda"
else:
    print("GPU not available, using CPU...")
    device = "cpu"
# --- 1. NEW: Environment Creation Function ---
# This function is required for SubprocVecEnv



# --- NEW: Main script must be wrapped in __name__ == '__main__' ---
if __name__ == '__main__':
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description='Train PPO agent on Robosuite Lift task')
    parser.add_argument('--controller-type', type=str, default='BASIC',
                        help='Controller type to use (default: BASIC) choices: BASIC, XYZ, XYZR')
    parser.add_argument('--timesteps', type=int, default=5,
                        help='Total training timesteps (default: 500,000)')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='Number of parallel environments (default: num_cpus - 1)')
    args = parser.parse_args()

    controller_type = args.controller_type
    output_name = f"ppo_lift_model_{controller_type}_{args.timesteps}"
    print(f"Selected controller type: {controller_type}")

    # --- 2. Create Parallel Environments ---
    # Get number of available CPUs, leave one or two free
    num_cpus = os.cpu_count() or 4
    n_envs = args.n_envs if args.n_envs is not None else max(2, num_cpus - 1)
    print(f"Using {n_envs} parallel environments...")
    # Create the vectorized environment using SubprocVecEnv
    # This creates 'n_envs' environments, each in its own CPU process
    try:
        env = SubprocVecEnv([lambda: make_robosuite_env(control_type=controller_type) for _ in range(n_envs)])
    except Exception as e:
        print(f"\n--- ERROR creating SubprocVecEnv ---")
        print(f"{e}")
        print("This can happen on some systems. Trying with 'fork' start method...")
        # 'fork' is less safe but sometimes needed. 'spawn' is default on Win/Mac
        try:
             env = SubprocVecEnv([lambda: make_robosuite_env(control_type=controller_type) for _ in range(n_envs)], start_method='fork')
        except Exception as e2:
             print(f"Failed again with 'fork': {e2}")
             print("Exiting.")
             exit()
    # --- 3. Define and Train the SB3 Agent ON GPU ---
    print("Creating PPO agent...")
    # Tune hyperparameters for parallel environments
    # Default n_steps=2048. We divide by n_envs so the total
    # rollout size is approx 2048, leading to more frequent GPU updates.
    rollout_steps_per_env = 2048 // n_envs
    # Instantiate the agent and explicitly set the device
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_lift_tensorboard/",
        device=device,
        batch_size=256,       # Size of minibatches for policy update
        n_steps=rollout_steps_per_env # Steps per env before update
    )
    print(f"Model created. Training on device: {model.device}")
    print(f"Rollout buffer size: {rollout_steps_per_env} (steps/env) * {n_envs} (envs) = {rollout_steps_per_env * n_envs} total steps")
    # --- FIX: Run a REAL training session, not a 50-step test ---
    print(f"Starting training ({args.timesteps:,} timesteps)...")
    # This will now run for a meaningful amount of time
    model.learn(total_timesteps=args.timesteps, log_interval=10) # Log every 10 updates
    model.save(output_name)
    print("Training finished. Model saved as 'ppo_lift_model.zip'")
    # --- 4. Evaluate the Trained Agent ---
    print("\nEvaluating trained agent with rendering...")
    # For evaluation, we create a single, separate env with the same controller config
    controller_config = get_controller_config(controller_type)

    eval_config = {
        "env_name": "Lift",
        "robots": "Panda",
        "controller_configs": controller_config,  # Use same controller config
        "has_renderer": True, # Render this one
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": True,
        "control_freq": 20,
        "horizon": 200,
        "reward_shaping": True,
    }
    eval_env_robosuite = suite.make(**eval_config)
    eval_env = GymWrapper(eval_env_robosuite)
    # Load the saved model
    model = PPO.load(output_name, env=eval_env, device=device)
    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)
    print(f"\n--- Evaluation Complete ---")
    print(f"Mean reward over 10 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("-----------------------------")
    # --- 5. NEW: Close all environments ---
    eval_env.close()
    env.close() # Important: close the parallel envs
    print("All environments closed.")