"""
Shared Environment Utilities for RL Training and Evaluation

This module contains common code for creating and configuring environments,
reducing duplication between training and evaluation scripts.

Features:
- GPU detection
- Gym environment creation with image/state observation support
- Robosuite environment creation
- Vectorized environment creation
- Wrapper utilities for image observations
"""

import torch
import os
import gymnasium as gym
from typing import Optional, Callable, List
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecTransposeImage


def check_gpu() -> str:
    """
    Check for GPU availability and return device string.

    Returns:
        Device string: 'cuda' if GPU available, 'cpu' otherwise
    """
    if torch.cuda.is_available():
        print("GPU is available! Using CUDA...")
        return "cuda"
    else:
        print("GPU not available, using CPU...")
        return "cpu"


def make_gym_env(
    env_name: str,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    use_image_obs: bool = False,
    image_size: int = 84,
) -> gym.Env:
    """
    Create a gymnasium environment with optional image observations.

    Args:
        env_name: Name of the gymnasium environment (e.g., 'LunarLander-v3')
        render_mode: Render mode ('human', 'rgb_array', or None)
        seed: Random seed for environment
        use_image_obs: If True, wrap environment to use image observations
        image_size: Size to resize images to (default: 84x84)

    Returns:
        Gymnasium environment instance
    """
    # Handle panda-gym environments which require render_mode to be set
    if 'Panda' in env_name and render_mode is None:
        render_mode = 'rgb_array'  # Default for panda-gym (headless)

    # Create base environment
    if use_image_obs and render_mode is None:
        # For image observations, we need rgb_array mode
        env = gym.make(env_name, render_mode="rgb_array")
    else:
        env = gym.make(env_name, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)

    # Wrap for image observations if requested
    if use_image_obs:
        from gymnasium.wrappers import ResizeObservation, GrayScaleObservation

        # Convert to grayscale and resize
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=(image_size, image_size))

    return env


def make_robosuite_env(
    env_name: str,
    control_type: str = "OSC_POSE",
    render: bool = False,
    use_image_obs: bool = False,
    camera_names: Optional[List[str]] = None,
    image_height: int = 84,
    image_width: int = 84,
) -> gym.Env:
    """
    Create a robosuite environment with optional image observations.

    Args:
        env_name: Name of the robosuite environment (e.g., 'Lift', 'Stack')
        control_type: Controller type (e.g., 'OSC_POSE', 'OSC_POSITION')
        render: Whether to enable on-screen rendering
        use_image_obs: If True, use camera observations instead of state
        camera_names: List of camera names to use (default: ['agentview'])
        image_height: Height of camera images
        image_width: Width of camera images

    Returns:
        Wrapped robosuite environment
    """
    try:
        import robosuite as suite
        from train_reward_yourself.utils import GymWrapper, get_controller_config

        controller_config = get_controller_config(control_type)

        if camera_names is None:
            camera_names = ["agentview"]

        env_config = {
            "env_name": env_name,
            "robots": "Panda",
            "controller_configs": controller_config,
            "has_renderer": render,
            "has_offscreen_renderer": use_image_obs,
            "use_camera_obs": use_image_obs,
            "camera_names": camera_names,
            "camera_heights": image_height,
            "camera_widths": image_width,
            "use_object_obs": not use_image_obs,  # Use state obs if not using images
            "control_freq": 20,
            "horizon": 200,
            "reward_shaping": True,
        }

        env = suite.make(**env_config)
        env = GymWrapper(env)
        return env

    except ImportError as e:
        raise ImportError(
            "Robosuite environments require robosuite to be installed. "
            "Install with: pip install robosuite"
        ) from e


def create_vectorized_env(
    env_type: str,
    env_name: str,
    n_envs: int,
    control_type: Optional[str] = None,
    seed: int = 0,
    use_image_obs: bool = False,
    image_size: int = 84,
    frame_stack: int = 1,
    render_mode: Optional[str] = None,
) -> SubprocVecEnv:
    """
    Create vectorized parallel environments with optional image observation support.

    Args:
        env_type: Type of environment ('gym' or 'robosuite')
        env_name: Name of the environment
        n_envs: Number of parallel environments
        control_type: Controller type for robosuite (optional)
        seed: Base random seed
        use_image_obs: If True, use image observations
        image_size: Size to resize images to (default: 84x84)
        frame_stack: Number of frames to stack (default: 1, no stacking)
        render_mode: Render mode for gym environments

    Returns:
        Vectorized environment
    """
    # Create environment factory functions
    if env_type == "gym":
        env_fns = [
            lambda i=i: make_gym_env(
                env_name,
                render_mode=render_mode if i == 0 else None,  # Only render first env
                seed=seed + i,
                use_image_obs=use_image_obs,
                image_size=image_size,
            )
            for i in range(n_envs)
        ]
    elif env_type == "robosuite":
        if control_type is None:
            control_type = "OSC_POSE"
        env_fns = [
            lambda i=i: make_robosuite_env(
                env_name,
                control_type=control_type,
                render=(i == 0),  # Only render first env
                use_image_obs=use_image_obs,
                image_height=image_size,
                image_width=image_size,
            )
            for i in range(n_envs)
        ]
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    # Create vectorized environment
    try:
        vec_env = SubprocVecEnv(env_fns)
    except Exception as e:
        print(f"\nError creating SubprocVecEnv: {e}")
        print("Trying with 'fork' start method...")
        try:
            vec_env = SubprocVecEnv(env_fns, start_method='fork')
        except Exception as e2:
            print(f"Failed again with 'fork': {e2}")
            print("Falling back to DummyVecEnv (single process)...")
            vec_env = DummyVecEnv(env_fns)

    # Apply frame stacking if using images
    if use_image_obs and frame_stack > 1:
        print(f"Applying frame stacking: {frame_stack} frames")
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
        # Transpose images to channel-first format (required by PyTorch CNNs)
        vec_env = VecTransposeImage(vec_env)

    return vec_env


def get_policy_type(use_image_obs: bool, env=None) -> str:
    """
    Get the appropriate policy type based on observation type.

    Args:
        use_image_obs: Whether using image observations
        env: Optional environment to check observation space

    Returns:
        Policy type string: 'CnnPolicy', 'MultiInputPolicy', or 'MlpPolicy'
    """
    if use_image_obs:
        return "CnnPolicy"

    # Check if environment has Dict observation space (like panda-gym)
    if env is not None:
        from gymnasium import spaces
        obs_space = env.observation_space if hasattr(env, 'observation_space') else None
        if obs_space is not None and isinstance(obs_space, spaces.Dict):
            return "MultiInputPolicy"

    return "MlpPolicy"


class EnvConfig:
    """
    Configuration class for environment settings.

    This makes it easy to pass around environment configuration
    and ensures consistency between training and evaluation.
    """

    def __init__(
        self,
        env_type: str,
        env_name: str,
        control_type: Optional[str] = None,
        use_image_obs: bool = False,
        image_size: int = 84,
        frame_stack: int = 1,
        seed: int = 0,
    ):
        """
        Initialize environment configuration.

        Args:
            env_type: Type of environment ('gym' or 'robosuite')
            env_name: Name of the environment
            control_type: Controller type for robosuite
            use_image_obs: Whether to use image observations
            image_size: Size to resize images to
            frame_stack: Number of frames to stack
            seed: Random seed
        """
        self.env_type = env_type
        self.env_name = env_name
        self.control_type = control_type
        self.use_image_obs = use_image_obs
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.seed = seed

    def create_single_env(self, render_mode: Optional[str] = None) -> gym.Env:
        """Create a single environment instance."""
        if self.env_type == "gym":
            return make_gym_env(
                self.env_name,
                render_mode=render_mode,
                seed=self.seed,
                use_image_obs=self.use_image_obs,
                image_size=self.image_size,
            )
        elif self.env_type == "robosuite":
            return make_robosuite_env(
                self.env_name,
                control_type=self.control_type or "OSC_POSE",
                render=render_mode == "human",
                use_image_obs=self.use_image_obs,
                image_height=self.image_size,
                image_width=self.image_size,
            )
        else:
            raise ValueError(f"Unknown env_type: {self.env_type}")

    def create_vec_env(self, n_envs: int, render_mode: Optional[str] = None):
        """Create a vectorized environment."""
        return create_vectorized_env(
            env_type=self.env_type,
            env_name=self.env_name,
            n_envs=n_envs,
            control_type=self.control_type,
            seed=self.seed,
            use_image_obs=self.use_image_obs,
            image_size=self.image_size,
            frame_stack=self.frame_stack,
            render_mode=render_mode,
        )

    def get_policy_type(self, env=None) -> str:
        """Get the appropriate policy type for this configuration."""
        return get_policy_type(self.use_image_obs, env)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "env_type": self.env_type,
            "env_name": self.env_name,
            "control_type": self.control_type,
            "use_image_obs": self.use_image_obs,
            "image_size": self.image_size,
            "frame_stack": self.frame_stack,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'EnvConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
