"""Lossless pose-action encoding for the official GR00T XYZ_ROTVEC interface."""
import numpy as np
from scipy.spatial.transform import Rotation


def encode(actions):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError('expected [xyz, quaternion_xyzw, finger_m]')
    return np.concatenate((actions[:, :3], Rotation.from_quat(actions[:, 3:7]).as_rotvec(), actions[:, 7:8]), axis=1).astype(np.float32)


def decode(actions):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError('expected [xyz, rotation_vector, finger_m]')
    return np.concatenate((actions[:, :3], Rotation.from_rotvec(actions[:, 3:6]).as_quat(), actions[:, 6:7]), axis=1).astype(np.float32)
