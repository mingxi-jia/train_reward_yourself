Usage Examples

  Collect demonstrations WITH images:

  python collect_expert_demos.py \
      --model-path ppo_lunarlander_v3_500000.zip \
      --env-type gym \
      --env-name LunarLander-v3 \
      --algo ppo \
      --n-demos 10 \
      --output demos_with_images.hdf5 \
      --save-images

  Collect demonstrations WITHOUT images (original behavior):

  python collect_expert_demos.py \
      --model-path ppo_lunarlander_v3_500000.zip \
      --env-type gym \
      --env-name LunarLander-v3 \
      --algo ppo \
      --n-demos 10 \
      --output demos_state_only.hdf5

  Visualize with images:

  # Overview with image samples
  python visualize_lunarlander_demos.py \
      --hdf5_path demos_with_images.hdf5 \
      --episodes 0 \
      --show-images

  # Animation with live rendered frames
  python visualize_lunarlander_demos.py \
      --hdf5_path demos_with_images.hdf5 \
      --episodes 0 \
      --mode animate \
      --show-images


python train_reward_yourself/train_bc_agent.py --data=demos_with_images.hdf5 --env-name=LunarLander-v3 --output=test