# train_reward_yourself

## TRY: Free yourself from reward shaping by Training Reward Yourself

# Installation

1. Install dependencies
    ```
        mamba env create -f conda_environment.yaml
    ```
2. Test installation integrity
    ```
        python example/train_robosuite_rl.py
    ```

# how to read the robomimic dataset

1. Download from drive (https://drive.google.com/file/d/1TqEY731OBs2v7I6Bmh1NBlQzcE1GhoBC/view?usp=sharing)
2. Read the state and actions
    ```
        python test/test_dataset.py
    ```