# train_reward_yourself

## TRY: Free yourself from reward shaping by Training Reward Yourself

# Installation

1. Install 
    ```
        git clone https://github.com/mingxi-jia/train_reward_yourself.git
        pip install -e .
    ```
2. Test installation integrity
    ```
        python example/train_robosuite_rl.py
    ```

# RL Agent Training

Train RL agents (PPO or SAC) on various environments including classic control (LunarLander) and robotic manipulation (Robosuite).

## Quick Start - LunarLander

### Train with PPO (State-based, recommended)
```bash
python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 500000 \
    --n-envs 4
```

### Train with SAC
```bash
python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo sac \
    --timesteps 500000 \
    --n-envs 1
```

### Train with Image Observations (CNN Policy)
```bash
python -m train_reward_yourself.train_rl_agent \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --timesteps 1000000 \
    --use-image-obs \
    --image-size 84 \
    --frame-stack 4
```

### View training progress
```bash
tensorboard --logdir ./ppo_lunarlander_v3_tensorboard/
```

### Evaluate trained model
```bash
python -m train_reward_yourself.eval_rl_agent \
    --model-path ppo_lunarlander_v3_state_500000.zip \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --episodes 10
```

### Evaluate image-based model (must match training config)
```bash
python -m train_reward_yourself.eval_rl_agent \
    --model-path ppo_lunarlander_v3_img_1000000.zip \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --use-image-obs \
    --image-size 84 \
    --frame-stack 4 \
    --episodes 10
```

### Demonstration: Collect demos from a optimal policy
```bash
python train_reward_yourself/collect_demonstrations.py --model-path ppo_lunarlander_v3_500000.zip --env-type gym --env-name LunarLander-v3 --algo ppo --n-demos 100 --output demos_with_images.hdf5
```
### Demonstration: Visualize demos
```bash
python visualize_lunarlander_rerun.py --hdf5_path=/home/mingxi/mingxi_ws/reward_learning/train_reward_yourself/demos_with_images.hdf5
```

### Train bc agent
```bash
python train_reward_yourself/train_bc_agent.py --data=demos_with_images.hdf5 --env-name=LunarLander-v3 --output=ppo_lunar_demo100
```

## Eval bc agent 
```bash
python train_reward_yourself/eval_bc_agent.py --model=test.zip --env-name=LunarLander-v3 --render
```

## PPO with pretrained bc
```bash
python train_reward_yourself/train_rl_agent.py --env-type gym --env-name LunarLander-v3   --algo ppo --timesteps 500000 --pretrain-model ppo_lunar_demo100.zip
```