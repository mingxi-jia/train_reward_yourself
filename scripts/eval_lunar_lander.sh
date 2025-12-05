#!/bin/bash

# Example evaluation scripts for trained LunarLander models
# Make this file executable with: chmod +x eval_lunar_lander.sh

echo "Evaluating trained RL agents on LunarLander-v3"
echo "==============================================="

# Set the model path (update this to your trained model)
MODEL_PATH="ppo_lunarlander_v3_500000.zip"

# Example 1: Basic evaluation with rendering
echo -e "\n[Example 1] Evaluating PPO model with rendering..."
python -m train_reward_yourself.eval_rl_agent \
    --model-path $MODEL_PATH \
    --env-type gym \
    --env-name LunarLander-v3 \
    --algo ppo \
    --episodes 10 \
    --deterministic

# Example 2: Evaluation without rendering (faster, for statistics)
# echo -e "\n[Example 2] Evaluating without rendering (50 episodes for statistics)..."
# python -m train_reward_yourself.eval_rl_agent \
#     --model-path $MODEL_PATH \
#     --env-type gym \
#     --env-name LunarLander-v3 \
#     --algo ppo \
#     --episodes 50 \
#     --no-render \
#     --deterministic

# Example 3: Record videos of evaluation episodes
# echo -e "\n[Example 3] Recording videos of evaluation episodes..."
# python -m train_reward_yourself.eval_rl_agent \
#     --model-path $MODEL_PATH \
#     --env-type gym \
#     --env-name LunarLander-v3 \
#     --algo ppo \
#     --record-video \
#     --video-episodes 5 \
#     --video-folder ./eval_videos

# Example 4: Evaluation with stochastic policy
# echo -e "\n[Example 4] Evaluating with stochastic policy..."
# python -m train_reward_yourself.eval_rl_agent \
#     --model-path $MODEL_PATH \
#     --env-type gym \
#     --env-name LunarLander-v3 \
#     --algo ppo \
#     --episodes 20 \
#     --stochastic \
#     --render

# Example 5: Evaluation with results saving
# echo -e "\n[Example 5] Evaluating and saving results to JSON..."
# python -m train_reward_yourself.eval_rl_agent \
#     --model-path $MODEL_PATH \
#     --env-type gym \
#     --env-name LunarLander-v3 \
#     --algo ppo \
#     --episodes 100 \
#     --no-render \
#     --deterministic \
#     --save-results \
#     --results-path eval_results_ppo_lunarlander.json

# Example 6: Evaluate SAC model
# SAC_MODEL_PATH="sac_lunarlander_v3_500000.zip"
# echo -e "\n[Example 6] Evaluating SAC model..."
# python -m train_reward_yourself.eval_rl_agent \
#     --model-path $SAC_MODEL_PATH \
#     --env-type gym \
#     --env-name LunarLander-v3 \
#     --algo sac \
#     --episodes 10 \
#     --deterministic

echo -e "\nEvaluation complete!"
