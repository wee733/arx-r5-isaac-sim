"""Numeric-only replay instrumentation; never supplies training RGB evidence."""
import numpy as np
from arx_door.envs.acceptance import acceptance_state, as_numpy
from arx_door.envs.recorders import _expected_camera_pose, read_physical_state
from arx_door.envs.b2arx import arm_state


def capture_numeric_pre_step(env):
    state = acceptance_state(env)
    physical = read_physical_state(env)
    ids = np.arange(env.num_envs)
    state.observe_safety(ids, physical['arm_contacts'], physical['camera_clearance'])
    frame = state.frames.copy()
    env._arx_pre_action_camera = {
        'wrist_camera_expected_pose': _expected_camera_pose(env),
        'timestamp': frame[:, None] / 30., 'action_index': frame[:, None],
    }
    env._arx_joint_target_substeps = []
    return {'joint_pos': as_numpy(arm_state(env))}
