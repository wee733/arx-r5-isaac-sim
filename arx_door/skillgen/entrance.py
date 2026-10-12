# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""CPU selection of a reachable entrance within a proven free-space prefix.

This only selects and trims source waypoints. It does not execute motion or
replace the collision-checked cuMotion connection to the selected entrance.
"""

import json
from pathlib import Path

import numpy as np


class EntranceSelectionError(RuntimeError):
    """Keep the failed entrance audit available to the caller."""

    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


def record_connection_audit(metrics, *, connected, result=None, error=None):
    """Distinguish a reachable entrance selection from an executable path."""
    metrics['entrance_selection_status'] = metrics['status']
    metrics['connection_success'] = bool(connected)
    if error is not None:
        metrics.update(connection_status='ERROR', connection_reason=str(error),
                       connection_error_type=type(error).__name__)
    else:
        result = result or {}
        metrics['connection_status'] = result.get('status', 'UNKNOWN')
        if 'reason' in result:
            metrics['connection_reason'] = result['reason']
        for source_key, audit_key in (
            ('status_code', 'connection_status_code'),
            ('backend', 'connection_backend'),
            ('collision_checking', 'connection_collision_checking'),
            ('target_diagnostics', 'target_collision_diagnostics'),
        ):
            if source_key in result:
                metrics[audit_key] = result[source_key]
    if not connected:
        metrics['status'] = 'connection_failed'


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def load_source_approach_masks(path: Path, demo_keys=None):
    """Read the same episode order as native DataGenInfoPool.

    A fully open command alone does not prove absence of contact. Source phase
    annotations and measured finger/arm forces must also identify the approach.
    Missing annotations disable trimming, while reachable original entrances
    remain valid without this optional fallback.
    """
    import h5py

    masks = []
    with h5py.File(path, 'r') as handle:
        for name in handle['data'].keys():
            if demo_keys is not None and name not in demo_keys:
                continue
            demo = handle['data'][name]
            diagnostics = demo['obs/diagnostics']
            names = json.loads(diagnostics.attrs['phase_names'])
            phase_id = np.asarray(diagnostics['phase_id'][:])
            fingers = np.asarray(diagnostics['finger_contacts'][:])
            arms = np.asarray(diagnostics['arm_contacts'][:])
            actions = np.asarray(demo['actions'][:])
            count = len(actions)
            if (
                phase_id.shape != (count,)
                or fingers.shape != (count, 2, 2, 3)
                or arms.shape != (count, 6, 3)
                or actions.shape != (count, 7)
                or not np.isfinite(fingers).all()
                or not np.isfinite(arms).all()
                or not np.isfinite(actions).all()
            ):
                raise ValueError(f'invalid non-contact source diagnostics in {name}')
            approach = names.index('approach_door_edge')
            mask = (
                (phase_id == approach)
                & (actions[:, 6] >= 0.04)
                & (np.linalg.norm(fingers, axis=-1).max(axis=(1, 2)) <= 1e-3)
                & (np.linalg.norm(arms, axis=-1).max(axis=1) <= 1e-3)
            )
            masks.append({'demo_key': name, 'noncontact_approach': mask})
    return masks


def select_reachable_entrance(
    trajectory, *, kinematics, base_world, start_joints,
    noncontact_mask=None, min_joint_margin_rad=0.02, open_gripper_min=0.04,
):
    """Return an entrance index, preserving every reachable original entrance.

    Only after the strict existing IK rejects index zero may a later waypoint
    be selected. Search stops at the first waypoint not proven to be a fully
    open, non-contact approach; it never skips a grasp or contact phase.
    """
    waypoints = [point for seq in trajectory.waypoint_sequences for point in seq.sequence]
    if not waypoints:
        raise ValueError('cannot select an entrance for an empty trajectory')
    base_world = _array(base_world)
    start_joints = _array(start_joints)
    if base_world.shape != (4, 4) or not np.isfinite(base_world).all():
        raise ValueError('robot base pose must be a finite 4x4 matrix')
    if start_joints.shape != (6,) or not np.isfinite(start_joints).all():
        raise ValueError('current arm state must contain six finite joints')
    if not np.isfinite(min_joint_margin_rad) or min_joint_margin_rad < 0:
        raise ValueError('joint limit margin must be finite and nonnegative')
    world_to_base = np.linalg.inv(base_world)
    metrics = {
        'original_waypoint_count': len(waypoints),
        'retained_waypoint_count': len(waypoints),
        'trimmed_waypoint_count': 0,
        'selected_waypoint_index': 0,
        'minimum_fallback_joint_margin_rad': float(min_joint_margin_rad),
        'candidates_examined': 0,
    }

    def solve(index):
        target = _array(waypoints[index].pose)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError('entrance waypoint pose must be a finite 4x4 matrix')
        joints = _array(kinematics.ik(world_to_base @ target, start_joints))
        margin = np.minimum(joints - kinematics.lower, kinematics.upper - joints).min()
        return float(margin)

    try:
        margin = solve(0)
    except ValueError as exc:
        metrics.update(original_ik_reachable=False, original_ik_error=str(exc))
    else:
        metrics.update(
            original_ik_reachable=True, selected_joint_margin_rad=margin,
            selected_gripper_action=_array(waypoints[0].gripper_action).reshape(-1).tolist(),
            status='original_entrance',
        )
        return metrics

    if noncontact_mask is None:
        metrics.update(status='missing_noncontact_source_annotations', noncontact_prefix_length=0)
        raise EntranceSelectionError('unreachable panel entrance; source non-contact annotations unavailable', metrics)
    noncontact_mask = np.asarray(noncontact_mask, dtype=bool)
    if noncontact_mask.shape != (len(waypoints),):
        raise ValueError('non-contact source mask must match native waypoint count')
    prefix_length = 0
    for point, proven_noncontact in zip(waypoints, noncontact_mask):
        gripper = _array(point.gripper_action).reshape(-1)
        if (
            not proven_noncontact or gripper.size != 1
            or not np.isfinite(gripper).all() or gripper[0] < open_gripper_min
        ):
            break
        prefix_length += 1
    metrics['noncontact_prefix_length'] = prefix_length
    for index in range(1, prefix_length):
        metrics['candidates_examined'] += 1
        try:
            margin = solve(index)
        except ValueError:
            continue
        if margin < min_joint_margin_rad:
            continue
        metrics.update(
            retained_waypoint_count=len(waypoints) - index,
            trimmed_waypoint_count=index, selected_waypoint_index=index,
            selected_joint_margin_rad=margin,
            selected_gripper_action=_array(waypoints[index].gripper_action).reshape(-1).tolist(),
            status='trimmed_noncontact_approach',
        )
        return metrics
    metrics['status'] = 'no_reachable_noncontact_entrance'
    raise EntranceSelectionError(
        'no reachable panel entrance in the fully open non-contact approach prefix', metrics,
    )


def trim_trajectory_prefix(trajectory, count):
    """Drop only selected prefix waypoints without changing retained objects."""
    if not isinstance(count, int) or count < 0 or count >= len(trajectory):
        raise ValueError('trim count must leave at least one trajectory waypoint')
    if count == 0:
        return
    retained_sequences = []
    remaining = count
    for sequence in trajectory.waypoint_sequences:
        if remaining >= len(sequence.sequence):
            remaining -= len(sequence.sequence)
            continue
        if remaining:
            sequence.sequence = sequence.sequence[remaining:]
            remaining = 0
        retained_sequences.append(sequence)
    trajectory.waypoint_sequences = retained_sequences
