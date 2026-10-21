# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Explicit Isaac Lab 2.3 / 3.0 boundaries for the PhysX door task.

Project manifests and SciPy use XYZW; cuMotion keeps its own WXYZ contract.
Only values crossing an Isaac Lab API are reordered here. Importing this
module does not initialize Isaac Lab, Kit, Torch, or a physics backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
import re
from typing import Any, Literal


@dataclass(frozen=True)
class LabProfile:
    """The API contracts of a supported release family, not a validation claim."""

    release_version: str
    quaternion_order: Literal['wxyz', 'xyzw']
    proxy_arrays: bool
    expected_sim: str
    expected_python: tuple[int, int]


def lab_profile(version: str | None = None) -> LabProfile:
    """Resolve the supported release family, preferring source VERSION.

    Lab 3.0's distribution metadata can read ``24.2.4``. Treating that value
    as a release major would select the wrong quaternion and physics API.
    Unknown families fail before data generation instead of guessing.
    """
    if version is None:
        return _installed_profile()
    match = re.fullmatch(r'(\d+)\.(\d+)(?:\.\d+)?(?:[-+a-zA-Z].*)?', version.strip())
    family = tuple(map(int, match.groups()[:2])) if match else None
    if family == (2, 3):
        return LabProfile(version, 'wxyz', False, '5.1', (3, 11))
    if family == (3, 0):
        return LabProfile(version, 'xyzw', True, '6.1', (3, 12))
    raise RuntimeError(
        f'Unsupported Isaac Lab release {version!r}; the door task supports '
        'Lab 2.3 / Sim 5.1 and Lab 3.0 / Sim 6.1. Use a configured runtime.'
    )


@lru_cache(maxsize=1)
def _installed_profile() -> LabProfile:
    spec = importlib.util.find_spec('isaaclab')
    if spec is None:
        raise RuntimeError('Isaac Lab is unavailable in this Python environment')
    if spec.origin:
        package_dir = Path(spec.origin).resolve().parent
        for parent in (package_dir, *package_dir.parents[:4]):
            version_file = parent / 'VERSION'
            if version_file.is_file():
                return lab_profile(version_file.read_text(encoding='utf-8').strip())
    try:
        version = importlib.metadata.version('isaaclab')
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError('Cannot determine the Isaac Lab source release') from exc
    return lab_profile(version)


def as_torch(value: Any):
    """Return a Tensor or the explicit zero-copy view of a Lab 3 ProxyArray.

    Unwrap before indexing: ProxyArray.shape describes its Warp storage,
    whereas Tensor.shape includes vector/quaternion component dimensions.
    Callers requiring an independent snapshot must clone the returned view.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, 'torch', None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    raise TypeError(f'Expected Tensor or ProxyArray, received {type(value).__name__}')


def _resolve_profile(profile: LabProfile | str | None) -> LabProfile:
    return profile if isinstance(profile, LabProfile) else lab_profile(profile)


def _reorder_quaternion(values: Any, indices: tuple[int, ...]):
    if hasattr(values, 'shape'):
        if values.shape[-1:] != (4,):
            raise ValueError('Quaternion must have four components on its last axis')
        return values[..., list(indices)]
    if isinstance(values, (tuple, list)) and len(values) == 4:
        return type(values)(values[index] for index in indices)
    raise ValueError('Quaternion must be an array or a sequence of four components')


def quaternion_from_xyzw(values: Any, profile: LabProfile | str | None = None):
    """Convert project/SciPy XYZW values to the selected Lab API's order."""
    order = _resolve_profile(profile).quaternion_order
    return _reorder_quaternion(values, (3, 0, 1, 2) if order == 'wxyz' else (0, 1, 2, 3))


def quaternion_to_xyzw(values: Any, profile: LabProfile | str | None = None):
    """Convert Lab state/camera quaternions to project/SciPy XYZW values."""
    order = _resolve_profile(profile).quaternion_order
    return _reorder_quaternion(values, (1, 2, 3, 0) if order == 'wxyz' else (0, 1, 2, 3))


def configure_physx(sim_cfg, profile: LabProfile | str | None = None) -> None:
    """Preserve reviewed contact settings and select Isaac Sim PhysX explicitly."""
    selected = _resolve_profile(profile)
    if selected.proxy_arrays:
        from isaaclab_physx.physics import PhysxCfg

        sim_cfg.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
        )
    else:
        sim_cfg.physx.bounce_threshold_velocity = 0.01
        sim_cfg.physx.friction_correlation_distance = 0.00625


def contact_force_matrix(data, profile: LabProfile | str | None = None):
    """Return filtered PhysX normal contact forces [N] as a Tensor.

    Lab 2.3's force_matrix_w also reports normal forces. Lab 3.0 exposes the
    explicit name; its total-force compatibility alias warns on every read.
    """
    field = 'normal_force_matrix_w' if _resolve_profile(profile).proxy_arrays else 'force_matrix_w'
    value = getattr(data, field)
    if value is None:
        raise RuntimeError('Filtered contact sensor data is unavailable')
    return as_torch(value)


def net_contact_forces(data, profile: LabProfile | str | None = None):
    """Return unfiltered PhysX normal contact forces [N] as a Tensor."""
    field = 'net_normal_forces_w' if _resolve_profile(profile).proxy_arrays else 'net_forces_w'
    value = getattr(data, field)
    if value is None:
        raise RuntimeError('Contact sensor data is unavailable')
    return as_torch(value)


def set_joint_position_target(
    asset, targets, *, joint_ids, env_ids=None,
    profile: LabProfile | str | None = None,
) -> None:
    """Set joint targets [m or rad] through the release's actuator API."""
    if _resolve_profile(profile).proxy_arrays:
        asset.actuators.target_command.set_position_index(
            value=targets, joint_ids=joint_ids, env_ids=env_ids,
        )
    else:
        asset.set_joint_position_target(targets, joint_ids=joint_ids, env_ids=env_ids)


def validate_mimic_api(generator_type: type) -> None:
    """Fail clearly if a checkout changes the overridden SkillGen method API."""
    required = {
        'generate_eef_subtask_trajectory': (
            'self', 'env_id', 'eef_name', 'subtask_ind',
            'all_randomized_subtask_boundaries', 'runtime_subtask_constraints_dict',
            'selected_src_demo_inds',
        ),
        'merge_eef_subtask_trajectory': (
            'self', 'env_id', 'eef_name', 'subtask_index',
            'prev_executed_traj', 'subtask_trajectory',
        ),
        '_convert_planned_trajectory_to_waypoints': (
            'self', 'motion_planner', 'gripper_action',
        ),
    }
    for name, prefix in required.items():
        method = getattr(generator_type, name, None)
        parameters = tuple(inspect.signature(method).parameters) if callable(method) else ()
        if parameters[:len(prefix)] != prefix:
            raise RuntimeError(
                f'Unsupported Isaac Lab Mimic API: {name}{parameters}; '
                f'the door adapter expects {prefix}'
            )
