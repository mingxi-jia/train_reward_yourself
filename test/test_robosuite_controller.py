import robosuite as suite
import numpy as np
from train_reward_yourself.utils import get_controller_config

# controller type
controller = 'BASIC' #help='Controller type to use (default: BASIC) choices: BASIC, XYZ, XYZR'

env = suite.make(
    "Lift",
    robots=["Panda"],             # load a Sawyer robot and a Panda robot
    # gripper_types="RethinkGripper",       # use default grippers per robot arm
    controller_configs=get_controller_config(controller),   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    env_configuration="opposed",            # (two-arm envs only) arms face each other
    has_renderer=True,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display