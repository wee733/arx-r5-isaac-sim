# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Strict AutoData acceptance; importing this module does not start Isaac Lab.

The historical quality profile is deliberately independent. Runtime samples are
post-action physics samples. The terminal waypoint flag controls AutoData's OR
latch only; finalization must also call :meth:`RuntimeAcceptance.accepted`.
"""

from __future__ import annotations

import numpy as np

from ..contract import EDGE_REGRASP, KNOB_HOLD_OPEN, normalize_behavior_family


STRICT_THRESHOLDS = {
    'final_door_angle_deg_exclusive': 29.0,
    'finger_force_n': 3.0,
    'unlock_handle_angle_deg': 43.5,
    'unlock_latch_m': .0154,
    'knob_gap_angle_deg': 10.0,
    'min_camera_clearance_m': .01,
    'max_arm_contact_n': 2.0,
}


def as_numpy(value) -> np.ndarray:
    """Cross the tensor/ProxyArray boundary explicitly, keeping state NumPy-only."""
    value = getattr(value, 'torch', value)
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class RuntimeAcceptance:
    """Accumulate irreversible episode violations separately for every env.

    Door columns are hinge radians, handle radians, latch metres. Contacts have
    shape ``(N, 2 fingers, 2 objects, 3)``; object order is knob, panel. An update
    contains exactly the rows identified by env_ids, including reordered subsets.
    """

    def __init__(self, num_envs: int, *, behavior_family: str = KNOB_HOLD_OPEN):
        if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs < 1:
            raise ValueError('num_envs must be a positive integer')
        self.num_envs = int(num_envs)
        self.behavior_family = normalize_behavior_family(behavior_family)
        self.frames = np.zeros(self.num_envs, dtype=np.int64)
        self.safety_samples = np.zeros(self.num_envs, dtype=np.int64)
        self.finite = np.ones(self.num_envs, dtype=bool)
        self.max_arm = np.zeros(self.num_envs)
        self.min_clearance = np.full(self.num_envs, np.inf)
        self.final_door = np.zeros((self.num_envs, 3))
        self.final_forces = np.zeros((self.num_envs, 2, 3))
        self.final_normal_forces = np.zeros((self.num_envs, 2))
        self.knob_grasp = np.full(self.num_envs, -1, dtype=np.int64)
        self.unlock = np.full(self.num_envs, -1, dtype=np.int64)
        self.gap = np.full(self.num_envs, -1, dtype=np.int64)
        self.first_opening_gap = np.full(self.num_envs, -1, dtype=np.int64)
        self.hold_checked = np.zeros(self.num_envs, dtype=np.int64)
        self.hold_valid = np.zeros(self.num_envs, dtype=np.int64)
        self.first_loss = np.full(self.num_envs, -1, dtype=np.int64)
        self.hold_min = np.full((self.num_envs, 2), np.inf)

    def _ids(self, env_ids=None) -> np.ndarray:
        if env_ids is None:
            return np.arange(self.num_envs, dtype=np.int64)
        raw = as_numpy(env_ids).reshape(-1)
        ids = raw.astype(np.int64)
        if (not np.array_equal(raw, ids) or len(np.unique(ids)) != len(ids)
                or np.any(ids < 0) or np.any(ids >= self.num_envs)):
            raise ValueError('env_ids must be unique valid environment indices')
        return ids

    def reset(self, env_ids=None) -> None:
        ids = self._ids(env_ids)
        self.frames[ids] = 0
        self.safety_samples[ids] = 0
        self.finite[ids] = True
        self.max_arm[ids] = 0
        self.min_clearance[ids] = np.inf
        self.final_door[ids] = 0
        self.final_forces[ids] = 0
        self.final_normal_forces[ids] = 0
        for events in (self.knob_grasp, self.unlock, self.gap,
                       self.first_opening_gap, self.first_loss):
            events[ids] = -1
        self.hold_checked[ids] = 0
        self.hold_valid[ids] = 0
        self.hold_min[ids] = np.inf

    def observe_safety(self, env_ids, arm_forces, camera_clearance) -> None:
        """Include pre-action bounds without advancing post-action event indices."""
        ids = self._ids(env_ids)
        count = len(ids)
        arms = np.asarray(as_numpy(arm_forces), dtype=float)
        clearance = np.asarray(as_numpy(camera_clearance), dtype=float)
        if arms.ndim != 3 or arms.shape[0] != count or arms.shape[1] < 1 or arms.shape[2] != 3:
            raise ValueError('arm_forces must have shape (N, links, 3) with at least one link')
        if clearance.shape not in ((count,), (count, 1)):
            raise ValueError('camera_clearance must have shape (N,) or (N, 1)')
        clearance = clearance.reshape(count)
        self.finite[ids] &= np.isfinite(arms).all(axis=(1, 2)) & np.isfinite(clearance)
        self.max_arm[ids] = np.maximum(self.max_arm[ids], np.nan_to_num(
            np.linalg.norm(arms, axis=-1).max(axis=1), nan=np.inf))
        self.min_clearance[ids] = np.minimum(self.min_clearance[ids], np.nan_to_num(
            clearance, nan=-np.inf))
        self.safety_samples[ids] += 1

    def update(self, env_ids, door_joint_pos, contact_forces, arm_forces,
               camera_clearance, *, panel_normal=None) -> None:
        ids = self._ids(env_ids)
        count = len(ids)
        door = np.asarray(as_numpy(door_joint_pos), dtype=float)
        contacts = np.asarray(as_numpy(contact_forces), dtype=float)
        arms = np.asarray(as_numpy(arm_forces), dtype=float)
        clearance = np.asarray(as_numpy(camera_clearance), dtype=float)
        if door.shape != (count, 3):
            raise ValueError('door_joint_pos must have shape (N, 3)')
        if contacts.shape != (count, 2, 2, 3):
            raise ValueError('contact_forces must have shape (N, 2, 2, 3)')
        if arms.ndim != 3 or arms.shape[0] != count or arms.shape[1] < 1 or arms.shape[2] != 3:
            raise ValueError('arm_forces must have shape (N, links, 3) with at least one link')
        if clearance.shape not in ((count,), (count, 1)):
            raise ValueError('camera_clearance must have shape (N,) or (N, 1)')
        clearance = clearance.reshape(count)
        if panel_normal is None:
            # Reviewed closed-panel frame: outward local Y rotates Rz(-hinge).
            normal = np.stack((np.sin(door[:, 0]), np.cos(door[:, 0]), np.zeros(count)), axis=-1)
        else:
            normal = np.asarray(as_numpy(panel_normal), dtype=float)
            if normal.shape != (count, 3):
                raise ValueError('panel_normal must have shape (N, 3)')
            length = np.linalg.norm(normal, axis=-1)
            if np.any(np.isfinite(length) & (length <= 1e-8)):
                raise ValueError('panel_normal must have nonzero length')
            normal = normal / length[:, None]
        finite = (np.isfinite(door).all(axis=1) & np.isfinite(contacts).all(axis=(1, 2, 3))
                  & np.isfinite(arms).all(axis=(1, 2)) & np.isfinite(clearance)
                  & np.isfinite(normal).all(axis=1))
        self.finite[ids] &= finite
        # Invalid samples reject the episode; do not let NaN erase prior extrema.
        self.observe_safety(ids, arms, clearance)
        self.final_door[ids] = door
        knob_norm = np.linalg.norm(contacts[:, :, 0], axis=-1)
        knob_ok = (knob_norm >= 3).all(axis=1) & finite
        unlocked = ((door[:, 1] >= np.deg2rad(43.5)) & (door[:, 2] >= .0154) & finite)
        opened_gap = (door[:, 0] >= np.deg2rad(10)) & finite
        frame = self.frames[ids]
        for events, flags in ((self.knob_grasp, knob_ok), (self.unlock, unlocked),
                              (self.gap, opened_gap & knob_ok),
                              (self.first_opening_gap, opened_gap)):
            rows = (events[ids] < 0) & flags
            events[ids[rows]] = frame[rows]
        holding = self.first_opening_gap[ids] >= 0
        self.hold_checked[ids] += holding
        self.hold_valid[ids] += holding & knob_ok
        lost = holding & ~knob_ok & (self.first_loss[ids] < 0)
        self.first_loss[ids[lost]] = frame[lost]
        self.hold_min[ids[holding]] = np.minimum(self.hold_min[ids[holding]],
                                                np.nan_to_num(knob_norm[holding], nan=0))
        object_index = 0 if self.behavior_family == KNOB_HOLD_OPEN else 1
        self.final_forces[ids] = contacts[:, :, object_index]
        self.final_normal_forces[ids] = np.einsum('nfc,nc->nf', self.final_forces[ids], normal)
        self.frames[ids] += 1

    def _checks(self, ids) -> dict[str, np.ndarray]:
        final_norms = np.linalg.norm(self.final_forces[ids], axis=-1)
        if self.behavior_family == EDGE_REGRASP:
            components = self.final_normal_forces[ids]
            opposed = ((np.abs(components) >= 3).all(axis=1)
                       & (components[:, 0] * components[:, 1] < 0))
        else:
            opposed = np.einsum('nc,nc->n', self.final_forces[ids, 0],
                                self.final_forces[ids, 1]) < 0
        checks = {
            'samples': self.frames[ids] > 0,
            'finite': self.finite[ids],
            'final_angle': self.final_door[ids, 0] > np.deg2rad(29),
            'final_grip': (final_norms >= 3).all(axis=1) & opposed,
            'arm_contact': self.max_arm[ids] <= 2,
            'camera_clearance': self.min_clearance[ids] >= .01,
            'grasp_unlock_gap_order': ((self.knob_grasp[ids] >= 0)
                & (self.knob_grasp[ids] < self.unlock[ids]) & (self.unlock[ids] < self.gap[ids])),
        }
        if self.behavior_family == KNOB_HOLD_OPEN:
            checks['continuous_knob_hold'] = (self.hold_checked[ids] > 0) & (self.first_loss[ids] < 0)
        return checks

    def accepted(self, env_ids=None) -> np.ndarray:
        ids = self._ids(env_ids)
        return np.logical_and.reduce(list(self._checks(ids).values())).copy()

    def summary(self, env_id: int) -> dict:
        ids = self._ids([env_id])
        index = int(ids[0])
        checks = self._checks(ids)
        event = lambda values: None if values[index] < 0 else int(values[index])
        finite_float = lambda value: float(value) if np.isfinite(value) else None
        result = {
            'env_id': index, 'behavior_family': self.behavior_family,
            'final_contact_object': 'knob' if self.behavior_family == KNOB_HOLD_OPEN else 'panel',
            'accepted': bool(np.logical_and.reduce(list(checks.values()))[0]),
            'frames': int(self.frames[index]),
            'safety_samples': int(self.safety_samples[index]),
            'sample_alignment': 'post_action_physics', 'thresholds': dict(STRICT_THRESHOLDS),
            'reasons': [name for name, flags in checks.items() if not flags[0]],
            'knob_grasp_frame': event(self.knob_grasp), 'unlock_frame': event(self.unlock),
            'gap_frame': event(self.gap),
            'grasp_unlock_gap_ordered': bool(checks['grasp_unlock_gap_order'][0]),
            'final_door_angle_deg': finite_float(np.rad2deg(self.final_door[index, 0])),
            'final_finger_force_n': [finite_float(v) for v in np.linalg.norm(self.final_forces[index], axis=-1)],
            'max_arm_contact_n': finite_float(self.max_arm[index]),
            'min_camera_clearance_m': finite_float(self.min_clearance[index]),
            'finite_evidence': bool(self.finite[index]),
        }
        if self.behavior_family == KNOB_HOLD_OPEN:
            checked = int(self.hold_checked[index])
            result.update({
                'knob_hold_start_frame': event(self.first_opening_gap),
                'knob_hold_checked_frames': checked,
                'knob_hold_contact_fraction': float(self.hold_valid[index] / checked) if checked else 0.,
                'knob_hold_first_loss_frame': event(self.first_loss),
                'knob_hold_min_finger_force_n': [finite_float(v) for v in self.hold_min[index]],
                'final_knob_force_dot_n2': finite_float(np.dot(*self.final_forces[index])),
            })
        else:
            result['final_panel_normal_force_n'] = [finite_float(v) for v in self.final_normal_forces[index]]
        return result

    def summaries(self, env_ids=None) -> list[dict]:
        return [self.summary(int(index)) for index in self._ids(env_ids)]


def acceptance_state(env) -> RuntimeAcceptance:
    """Return the recorder and success predicate's shared per-environment state."""
    state = getattr(env, '_arx_acceptance', None)
    if state is None:
        family = getattr(env.cfg, 'behavior_family', KNOB_HOLD_OPEN)
        state = RuntimeAcceptance(env.num_envs, behavior_family=family)
        env._arx_acceptance = state
    return state


def reset_acceptance(env, env_ids=None) -> None:
    """Reset only requested episodes; usable as an Arena reset event or recorder hook."""
    state = acceptance_state(env)
    ids = state._ids(env_ids)
    state.reset(ids)
    terminal = getattr(env, '_arx_terminal', None)
    if terminal is not None:
        if hasattr(terminal, 'device'):
            import torch
            terminal[torch.as_tensor(ids, device=terminal.device, dtype=torch.long)] = False
        else:
            terminal[ids] = False


def domain_success(env):
    """Gate the current/cumulative result with the final executed waypoint flag.

    The post-step recorder must run before the official AutoData env interface
    queries this predicate. Arena progress and historical online thresholds are
    never used as success evidence. Missing terminal qualification fails closed.
    """
    accepted = acceptance_state(env).accepted()
    terminal = as_numpy(getattr(env, '_arx_terminal', np.zeros(env.num_envs, dtype=bool)))
    if terminal.shape != (env.num_envs,) or terminal.dtype != np.bool_:
        raise ValueError('_arx_terminal must be a boolean vector indexed by environment')
    success = accepted & terminal
    if hasattr(env, 'device'):
        import torch
        return torch.as_tensor(success, device=env.device, dtype=torch.bool)
    return success
