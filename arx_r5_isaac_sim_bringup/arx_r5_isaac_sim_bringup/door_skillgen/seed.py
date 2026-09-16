# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Convert reviewed ARX door recordings to Isaac Lab Mimic HDF5 seeds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation


ENV_ID = 'Isaac-ARX-R5-Door-SkillGen-v0'
EEF_NAME = 'link6'
REQUIRED_FIELDS = (
    'state',
    'action',
    'timestamp',
    'phase',
    'door',
    'contacts',
    'camera_clearance',
    'arm_contacts',
)
FINE_PHASE_ORDER = (
    'cumotion_approach_knob',
    'approach_knob',
    'grasp_knob',
    'unlock_round_knob',
    'pull_knob_for_gap',
    'release_knob',
    'retreat_from_knob',
    'cumotion_transfer_to_upright_edge',
    'approach_door_edge',
    'grasp_door_edge',
    'pull_door_by_edge',
    'hold_door_open',
)

# The panel frame is also the hinge frame in the approved door asset.
PANEL_ZERO_TRANSLATION = np.asarray((-0.378, -0.331, 0.0))


class ForwardKinematics(Protocol):
    """Minimal kinematic interface required by the offline converter."""

    def fk(self, joints: np.ndarray) -> np.ndarray:
        """Return base-to-link6 as a homogeneous matrix."""


@dataclass(frozen=True)
class MacroSkill:
    """One SkillGen segment with an explicit non-contact gap around it."""

    name: str
    start_phase: str
    end_phase: str
    object_ref: str


MACRO_SKILLS = (
    MacroSkill(
        'knob_interaction',
        'approach_knob',
        'retreat_from_knob',
        'knob',
    ),
    MacroSkill(
        'edge_interaction',
        'approach_door_edge',
        'hold_door_open',
        'panel',
    ),
)


@dataclass(frozen=True)
class DoorSeed:
    """Validated source attempt and all numeric trajectories."""

    path: Path
    metadata: dict[str, Any]
    state: np.ndarray
    action: np.ndarray
    timestamp: np.ndarray
    phase: np.ndarray
    door: np.ndarray
    contacts: np.ndarray
    camera_clearance: np.ndarray
    arm_contacts: np.ndarray


@dataclass(frozen=True)
class SkillGenArrays:
    """Loader-facing arrays plus door-specific diagnostic information."""

    source_path: Path
    source_sha256: str
    source_metadata: dict[str, Any]
    actions: np.ndarray
    state: np.ndarray
    timestamp: np.ndarray
    door: np.ndarray
    contacts: np.ndarray
    camera_clearance: np.ndarray
    arm_contacts: np.ndarray
    eef_pose: dict[str, np.ndarray]
    target_eef_pose: dict[str, np.ndarray]
    object_pose: dict[str, np.ndarray]
    subtask_start_signals: dict[str, np.ndarray]
    subtask_term_signals: dict[str, np.ndarray]
    macro_boundaries: dict[str, tuple[int, int]]
    phase_id: np.ndarray
    phase_names: tuple[str, ...]


def _as_finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f'{label} must have shape {shape}, got {array.shape}')
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(f'{label} must contain finite numeric values')
    return array


def _validate_phase_order(phases: np.ndarray) -> tuple[str, ...]:
    if phases.ndim != 1:
        raise ValueError(f'phase must be one-dimensional, got {phases.shape}')
    names = tuple(dict.fromkeys(str(value) for value in phases))
    if names != FINE_PHASE_ORDER:
        raise ValueError(
            'door phase order does not match the approved seed contract: '
            f'{names}'
        )
    return names


def load_seed(path: str | Path) -> DoorSeed:
    """Load a successful, explicitly user-approved door attempt."""
    source = Path(path).expanduser().resolve()
    metadata_path = source / 'episode.json'
    trajectory_path = source / 'trajectory.npz'
    if not metadata_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(
            f'{source} must contain episode.json and trajectory.npz'
        )
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('schema') != 'arx-door-teaching-attempt-v1':
        raise ValueError('unsupported door attempt schema')
    if not metadata.get('success'):
        raise ValueError('only successful attempts can become SkillGen seeds')
    if metadata.get('quality_status') != 'user_approved_seed':
        raise ValueError('SkillGen seeds must be explicitly user-approved')

    with np.load(trajectory_path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_FIELDS).difference(archive.files))
        if missing:
            raise ValueError(f'trajectory is missing required fields: {missing}')
        values = {name: archive[name].copy() for name in REQUIRED_FIELDS}

    state = np.asarray(values['state'])
    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError(f'state must have shape (N, 7), got {state.shape}')
    count = state.shape[0]
    _as_finite(state, (count, 7), 'state')
    _as_finite(values['action'], (count, 7), 'action')
    timestamp = _as_finite(values['timestamp'], (count,), 'timestamp')
    _as_finite(values['door'], (count, 3), 'door')
    _as_finite(values['contacts'], (count, 2, 2, 3), 'contacts')
    _as_finite(values['camera_clearance'], (count,), 'camera_clearance')
    _as_finite(values['arm_contacts'], (count, 6, 3), 'arm_contacts')
    if count < 2 or not np.all(np.diff(timestamp) > 0):
        raise ValueError('timestamp must be strictly increasing')
    if int(metadata.get('frames', -1)) != count:
        raise ValueError('episode frame count does not match trajectory')
    if int(metadata.get('fps', -1)) <= 0:
        raise ValueError('episode fps must be positive')
    _validate_phase_order(np.asarray(values['phase']))

    placement = metadata.get('scene_placement', {}).get('placement', {})
    _as_finite(placement.get('position'), (3,), 'base position')
    quaternion = _as_finite(
        placement.get('quaternion_xyzw'), (4,), 'base quaternion'
    )
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-4):
        raise ValueError('base quaternion must be normalized')
    _as_finite(
        metadata.get('scene_placement', {}).get('handle_world_xyz'),
        (3,),
        'handle reference',
    )

    return DoorSeed(
        path=source,
        metadata=metadata,
        **{name: np.asarray(values[name]) for name in REQUIRED_FIELDS},
    )


def _base_pose(metadata: dict[str, Any]) -> np.ndarray:
    placement = metadata['scene_placement']['placement']
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(
        placement['quaternion_xyzw']
    ).as_matrix()
    result[:3, 3] = placement['position']
    return result


def _door_object_poses(seed: DoorSeed) -> dict[str, np.ndarray]:
    count = seed.state.shape[0]
    panel = np.repeat(np.eye(4)[None], count, axis=0)
    knob = panel.copy()
    handle_zero = np.asarray(seed.metadata['scene_placement']['handle_world_xyz'])
    handle_offset = handle_zero - PANEL_ZERO_TRANSLATION
    for index, (hinge_angle, handle_angle, _) in enumerate(seed.door):
        panel_rotation = Rotation.from_euler('z', -float(hinge_angle)).as_matrix()
        panel[index, :3, :3] = panel_rotation
        panel[index, :3, 3] = PANEL_ZERO_TRANSLATION
        knob[index, :3, :3] = (
            panel_rotation
            @ Rotation.from_euler('y', float(handle_angle)).as_matrix()
        )
        knob[index, :3, 3] = (
            PANEL_ZERO_TRANSLATION + panel_rotation @ handle_offset
        )
    return {
        'knob': knob.astype(np.float32),
        'panel': panel.astype(np.float32),
    }


def _macro_annotations(
    phases: np.ndarray,
) -> tuple[
    dict[str, tuple[int, int]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    text = np.asarray(phases).astype(str)
    count = len(text)
    boundaries: dict[str, tuple[int, int]] = {}
    starts: dict[str, np.ndarray] = {}
    terms: dict[str, np.ndarray] = {}
    previous_end = -1
    for skill in MACRO_SKILLS:
        start_matches = np.flatnonzero(text == skill.start_phase)
        end_matches = np.flatnonzero(text == skill.end_phase)
        if not len(start_matches) or not len(end_matches):
            raise ValueError(f'missing phases for macro skill {skill.name}')
        start = int(start_matches[0])
        end = int(end_matches[-1]) + 1
        if start <= previous_end or end <= start:
            raise ValueError('SkillGen macro skills must be ordered and disjoint')
        boundaries[skill.name] = (start, end)
        start_signal = np.zeros((count, 1), dtype=np.uint8)
        start_signal[start:] = 1
        term_signal = np.zeros((count, 1), dtype=np.uint8)
        term_signal[end - 1:] = 1
        starts[skill.name] = start_signal
        terms[skill.name] = term_signal
        previous_end = end
    return boundaries, starts, terms


def build_skillgen_arrays(
    seed: DoorSeed,
    kinematics: ForwardKinematics,
) -> SkillGenArrays:
    """Enrich joint recordings with SkillGen poses and annotations."""
    count = seed.state.shape[0]
    base_world = _base_pose(seed.metadata)
    actual = np.empty((count, 4, 4), dtype=np.float32)
    target = np.empty_like(actual)
    for index in range(count):
        actual[index] = base_world @ kinematics.fk(seed.state[index, :6])
        target[index] = base_world @ kinematics.fk(seed.action[index, :6])
    boundaries, starts, terms = _macro_annotations(seed.phase)
    phase_names = _validate_phase_order(seed.phase)
    phase_to_id = {name: index for index, name in enumerate(phase_names)}
    phase_id = np.asarray(
        [phase_to_id[str(value)] for value in seed.phase], dtype=np.uint8
    )
    source_hash = hashlib.sha256(
        (seed.path / 'trajectory.npz').read_bytes()
    ).hexdigest()
    return SkillGenArrays(
        source_path=seed.path,
        source_sha256=source_hash,
        source_metadata=seed.metadata,
        actions=seed.action.astype(np.float32),
        state=seed.state.astype(np.float32),
        timestamp=seed.timestamp.astype(np.float64),
        door=seed.door.astype(np.float32),
        contacts=seed.contacts.astype(np.float32),
        camera_clearance=seed.camera_clearance.astype(np.float32),
        arm_contacts=seed.arm_contacts.astype(np.float32),
        eef_pose={EEF_NAME: actual},
        target_eef_pose={EEF_NAME: target},
        object_pose=_door_object_poses(seed),
        subtask_start_signals=starts,
        subtask_term_signals=terms,
        macro_boundaries=boundaries,
        phase_id=phase_id,
        phase_names=phase_names,
    )


def _write_dataset(group: Any, name: str, value: np.ndarray) -> None:
    group.create_dataset(name, data=value, compression='gzip')


def write_skillgen_hdf5(
    arrays: SkillGenArrays,
    output: str | Path,
    *,
    env_name: str = ENV_ID,
) -> None:
    """Atomically write an Isaac Lab HDF5 source demonstration."""
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            'h5py is required; run this command with the Isaac Lab Python'
        ) from error

    destination = Path(output).expanduser().resolve()
    if destination.suffix != '.hdf5':
        raise ValueError('SkillGen output must use the .hdf5 suffix')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f'.{destination.name}.{os.getpid()}.tmp'
    )
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, 'w') as dataset:
            data = dataset.create_group('data')
            data.attrs['total'] = len(arrays.actions)
            data.attrs['env_args'] = json.dumps({
                'env_name': env_name,
                'type': 2,
                'env_kwargs': {},
            })
            demo = data.create_group('demo_0')
            demo.attrs['num_samples'] = len(arrays.actions)
            demo.attrs['seed'] = int(
                arrays.source_metadata['scene_placement'].get('seed', 0)
            )
            demo.attrs['success'] = True
            demo.attrs['source_path'] = str(arrays.source_path)
            demo.attrs['source_sha256'] = arrays.source_sha256
            demo.attrs['schema'] = 'arx-door-skillgen-seed-v1'
            demo.attrs['macro_boundaries'] = json.dumps(
                arrays.macro_boundaries
            )
            _write_dataset(demo, 'actions', arrays.actions)
            obs = demo.create_group('obs')
            datagen = obs.create_group('datagen_info')
            eef = datagen.create_group('eef_pose')
            target = datagen.create_group('target_eef_pose')
            objects = datagen.create_group('object_pose')
            starts = datagen.create_group('subtask_start_signals')
            terms = datagen.create_group('subtask_term_signals')
            for key, value in arrays.eef_pose.items():
                _write_dataset(eef, key, value)
            for key, value in arrays.target_eef_pose.items():
                _write_dataset(target, key, value)
            for key, value in arrays.object_pose.items():
                _write_dataset(objects, key, value)
            for key, value in arrays.subtask_start_signals.items():
                _write_dataset(starts, key, value)
            for key, value in arrays.subtask_term_signals.items():
                _write_dataset(terms, key, value)
            diagnostics = obs.create_group('diagnostics')
            diagnostics.attrs['phase_names'] = json.dumps(
                arrays.phase_names
            )
            _write_dataset(diagnostics, 'phase_id', arrays.phase_id)
            _write_dataset(diagnostics, 'timestamp', arrays.timestamp)
            _write_dataset(diagnostics, 'joint_state', arrays.state)
            _write_dataset(diagnostics, 'door_joint_pos', arrays.door)
            _write_dataset(diagnostics, 'finger_contacts', arrays.contacts)
            _write_dataset(
                diagnostics,
                'camera_clearance',
                arrays.camera_clearance,
            )
            _write_dataset(
                diagnostics,
                'arm_contacts',
                arrays.arm_contacts,
            )
            dataset.flush()
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def conversion_report(arrays: SkillGenArrays, output: str | Path) -> dict:
    """Return deterministic provenance and shape information for operators."""
    return {
        'schema': 'arx-door-skillgen-conversion-v1',
        'source': str(arrays.source_path),
        'source_sha256': arrays.source_sha256,
        'output': str(Path(output).expanduser().resolve()),
        'environment': ENV_ID,
        'frames': len(arrays.actions),
        'action_shape': list(arrays.actions.shape),
        'action_semantics': 'absolute_joint_target',
        'macro_boundaries': {
            key: list(value) for key, value in arrays.macro_boundaries.items()
        },
        'fine_phases': list(arrays.phase_names),
    }
