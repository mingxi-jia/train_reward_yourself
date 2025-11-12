import robosuite as suite
import numpy as np
from robosuite.controllers import load_composite_controller_config

# BASIC controller: arms controlled using OSC, mobile base (if present) using JOINT_VELOCITY, other parts controlled using JOINT_POSITION 
def get_controller_config(use_osc_position=False):
    controller_config = load_composite_controller_config(controller="BASIC")

    if use_osc_position:
        print("Using OSC_POSE controller configuration...")
        new_config = {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.0, 0.0, 0.5],
            "output_min": [-0.05, -0.05, -0.05, 0.0, 0.0, -0.5],
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
    else:
        print("Using default BASIC controller configuration...")
    # delete other keys if exist
    for key in list(controller_config['body_parts'].keys()):
        if key != 'right':
            del controller_config['body_parts'][key]
    return controller_config

env = suite.make(
    "Lift",
    robots=["Panda"],             # load a Sawyer robot and a Panda robot
    # gripper_types="RethinkGripper",       # use default grippers per robot arm
    controller_configs=get_controller_config(True),   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
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