"""Boundary for the B2ARX checkpoint: decoded absolute EEFs, never raw network output.

No ROS, CUDA, or simulation import here. Call from the model process and publish
the resulting JSON on /vla/chunk. The executor owns timing and all CAN commands.
"""
from __future__ import annotations

import numpy as np


def observation(state_message: dict, rgb: dict, language: str) -> dict:
    """Pack the measured 13D state and future two-camera input for Gr00tPolicy."""
    if state_message.get("schema") != "arx_vla_observation_v1":
        raise ValueError("unknown observation schema")
    if not state_message.get("calibrated"):
        raise ValueError("real/model frame and gripper calibration is not verified")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language is required")
    if set(rgb) != {"wrist", "ego"}:
        raise ValueError("both wrist and ego RGB are required; no placeholder camera input")
    video = {}
    for key, value in rgb.items():
        a = np.asarray(value)
        if a.shape != (480, 640, 3) or a.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 RGB (480,640,3)")
        video[key] = np.ascontiguousarray(a)[None, None]
    state = {}
    for key, size in (("eef", 6), ("gripper", 1), ("joints", 6)):
        a = np.asarray(state_message["state"][key], dtype=np.float32)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(f"invalid state.{key}")
        state[key] = a[None, None].copy()
    return {"video": video, "state": state,
            "language": {"annotation.human.task_description": [[language]]}}


def from_policy_reply(reply: dict, state_message: dict, sequence: int, *,
                      execute_steps: int = 8, control_gripper: bool = True,
                      training_finger_max_m: float = 0.044) -> dict:
    """Encode official Policy API output after denormalization/relative restoration.

    Actions are due at observation time + index/30 s. Inference latency
    consumes the chunk; an expired reply is rejected, never restarted at t=0.
    """
    if not state_message.get("calibrated"):
        raise ValueError("unverified real/model calibration")
    if set(reply) != {"eef", "gripper"}:
        raise ValueError("expected eef/gripper, not the legacy single_arm protocol")
    eef, grip = np.asarray(reply["eef"]), np.asarray(reply["gripper"])
    if eef.shape != (1, 16, 6) or grip.shape != (1, 16, 1):
        raise ValueError(f"expected (1,16,6)/(1,16,1), got {eef.shape}/{grip.shape}")
    if not np.isfinite(eef).all() or not np.isfinite(grip).all():
        raise ValueError("nonfinite action")
    if isinstance(execute_steps, bool) or not isinstance(execute_steps, int) or not 1 <= execute_steps <= 16:
        raise ValueError("execute_steps must be 1..16")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        raise ValueError("sequence must be a positive integer")
    if control_gripper and (np.any(grip < -0.001) or np.any(grip > training_finger_max_m + 0.001)):
        raise ValueError("gripper target exceeds the trained metre range")
    # Keep the original physical targets. The executor saturates to its measured
    # endpoint/nominal opening reference and reports clipped sample counts.
    return {"schema": "arx_vla_chunk_v1", "session_id": state_message["session_id"],
            "observation_id": state_message["observation_id"], "sequence": sequence,
            "semantics": "absolute_xyz_rotvec", "frame": "model_mount_link6",
            "dt_s": 1 / 30, "control_gripper": control_gripper,
            "eef": eef[0, :execute_steps].tolist(),
            "gripper": grip[0, :execute_steps, 0].tolist()}
