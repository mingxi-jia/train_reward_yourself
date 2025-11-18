import h5py
from tqdm import tqdm
import numpy as np
import sys
import open3d as o3d
import os
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

dataset_path = 'block_lifting_robomimic.hdf5'
dataset = h5py.File(dataset_path, 'r')

num_traj = len(dataset['data'].keys())
print(f"Number of trajectories in the dataset: {num_traj}")

# get data
trajectory_index = 0
demo = dataset[f'data/demo_{trajectory_index}']

states = demo['obs']['object'][:]
actions = demo['actions'][:]

dataset.close()