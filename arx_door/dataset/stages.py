# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Recover per-frame task stages of a generated episode from its source seed.

SkillGen replays the seed's knob skill as one rigid transform of the seed's
target link6 poses in the knob frame, after one or more planned free-space
segments. Generated ``transition_index`` only counts planner calls, so stages
are recovered here: the longest generated suffix whose targets equal the
transformed seed targets is the replay, and each replayed frame inherits the
seed's reviewed fine phase. Nothing is inferred from timing alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stage:
    """One language-conditioned stage and the seed phases it covers."""

    name: str
    instruction: str
    seed_phases: tuple[str, ...]


# Four stages, each with one dominant dynamic regime. ``hold_door_open`` is
# the motionless tail of the pull and gets the same instruction.
KNOB_HOLD_OPEN_STAGES = (
    Stage('approach', 'Move the open gripper to the front of the round doorknob.',
          ('cumotion_approach_knob', 'approach_knob')),
    Stage('grasp', 'Close the gripper around the round doorknob.', ('grasp_knob',)),
    Stage('unlock', 'Turn the round doorknob counterclockwise until the latch retracts.',
          ('unlock_round_knob',)),
    Stage('open', 'Keep holding the doorknob and pull the door open.',
          ('pull_knob_for_gap', 'pull_door_by_knob', 'hold_door_open')),
)
STAGE_NAMES = tuple(stage.name for stage in KNOB_HOLD_OPEN_STAGES)

# Per-frame origin of a generated command target.
PLANNED_TRANSITION, RECOVERY, SEED_REPLAY = 0, 1, 2
FRAME_SOURCES = ('planned_transition', 'recovery', 'seed_replay')


@dataclass(frozen=True)
class Alignment:
    """Map every generated frame to its origin and, for replays, its seed frame."""

    seed_frame: np.ndarray      # (N,) seed frame index, -1 for planned frames
    source: np.ndarray          # (N,) PLANNED_TRANSITION | RECOVERY | SEED_REPLAY
    replay_start: int           # first generated replay frame
    recovery_start: int | None  # first frame of a second planned segment, if any

    def to_json(self) -> dict:
        return {'replay_start_frame': self.replay_start,
                'recovery_start_frame': self.recovery_start,
                'first_replayed_seed_frame': int(self.seed_frame[self.replay_start]),
                'frame_source_counts': {name: int((self.source == index).sum())
                                        for index, name in enumerate(FRAME_SOURCES)}}


def _poses(value, label):
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f'{label} must be finite (N, 4, 4) poses')
    return poses


def align_to_seed(gen_target, gen_knob, seed_target, seed_knob, *, seed_start, seed_end,
                  transition_index, position_atol=1e-4, rotation_atol=1e-4):
    """Align a generated episode with the seed skill ``[seed_start, seed_end)``.

    The skill is anchored at the knob pose before any contact, which is the
    first frame of both recordings. ``transition_index`` separates the first
    planned transition from later corrective (recovery) segments.
    """
    gen_target, gen_knob = _poses(gen_target, 'generated targets'), _poses(gen_knob, 'generated knob')
    seed_target, seed_knob = _poses(seed_target, 'seed targets'), _poses(seed_knob, 'seed knob')
    count = len(gen_target)
    transition_index = np.asarray(transition_index).reshape(-1)
    if len(gen_knob) != count or transition_index.shape != (count,):
        raise ValueError('generated targets, knob poses and transition indices must align')
    if not 0 <= seed_start < seed_end <= len(seed_target) or len(seed_knob) != len(seed_target):
        raise ValueError('seed skill boundaries are outside the seed recording')
    anchor = gen_knob[0] @ np.linalg.inv(seed_knob[seed_start])
    expected = anchor @ seed_target[seed_start:seed_end]
    longest = min(count, seed_end - seed_start)
    gen_tail = gen_target[count - longest:]
    seed_tail = expected[len(expected) - longest:]
    match = (
        (np.linalg.norm(gen_tail[:, :3, 3] - seed_tail[:, :3, 3], axis=-1) <= position_atol)
        & (np.abs(gen_tail[:, :3, :3] - seed_tail[:, :3, :3]).max(axis=(1, 2)) <= rotation_atol)
    )
    mismatched = np.flatnonzero(~match)
    length = longest - (mismatched[-1] + 1 if len(mismatched) else 0)
    if length < 1:
        raise ValueError('generated episode does not end with a replay of the seed skill')
    replay_start = int(count - length)
    seed_frame = np.full(count, -1, dtype=np.int64)
    seed_frame[replay_start:] = np.arange(seed_end - length, seed_end)
    source = np.full(count, PLANNED_TRANSITION, dtype=np.int64)
    source[replay_start:] = SEED_REPLAY
    first_plan = transition_index[0]
    later = np.flatnonzero(transition_index[:replay_start] != first_plan)
    recovery_start = int(later[0]) if len(later) else None
    if recovery_start is not None:
        source[recovery_start:replay_start] = RECOVERY
    return Alignment(seed_frame, source, replay_start, recovery_start)


def stage_labels(alignment: Alignment, seed_phase_id, seed_phase_names,
                 stages=KNOB_HOLD_OPEN_STAGES) -> np.ndarray:
    """Return a stage index per generated frame; planned frames are approach."""
    names = list(seed_phase_names)
    by_phase = {}
    for index, stage in enumerate(stages):
        for phase in stage.seed_phases:
            by_phase[phase] = index
    missing = set(names) - set(by_phase)
    if missing:
        raise ValueError(f'seed phases without a stage: {sorted(missing)}')
    seed_stage = np.asarray([by_phase[names[phase]] for phase in np.asarray(seed_phase_id)])
    labels = np.zeros(len(alignment.seed_frame), dtype=np.int64)
    replay = alignment.source == SEED_REPLAY
    labels[replay] = seed_stage[alignment.seed_frame[replay]]
    if np.any(np.diff(labels) < 0):
        raise ValueError('stage sequence must be monotonic')
    return labels


def transition_metrics(arm_state, end: int) -> dict:
    """Describe the planned free-space segment ``[0, end)`` of six arm joints.

    ``path_ratio`` is the joint-space L1 path length over the endpoint L1
    distance (1 for a straight joint path). ``overshoot_rad`` is the largest
    excursion of any joint outside its start/end interval; a planner branch
    that first swings the wrong way shows up here.
    """
    joints = np.asarray(arm_state, dtype=np.float64)[:end + 1, :6]
    if len(joints) < 2 or not np.isfinite(joints).all():
        raise ValueError('transition requires at least two finite six-joint states')
    start, stop = joints[0], joints[-1]
    low, high = np.minimum(start, stop), np.maximum(start, stop)
    overshoot = np.maximum(joints.max(axis=0) - high, low - joints.min(axis=0))
    distance = float(np.abs(stop - start).sum())
    path = float(np.abs(np.diff(joints, axis=0)).sum())
    return {'frames': int(end), 'path_l1_rad': path, 'endpoint_l1_rad': distance,
            'path_ratio': path / distance if distance > 1e-9 else float('inf'),
            'overshoot_rad': overshoot.tolist(),
            'max_overshoot_rad': float(overshoot.max())}


@dataclass(frozen=True)
class SeedSkill:
    """The seed recording fields needed to align and label generated episodes."""

    target: np.ndarray
    knob: np.ndarray
    start: int
    end: int
    phase_id: np.ndarray
    phase_names: tuple[str, ...]


def load_seed_skill(path, skill='knob_interaction') -> SeedSkill:
    """Read one SkillGen seed: knob-skill boundaries, targets and reviewed phases."""
    import json

    import h5py

    with h5py.File(path, 'r') as handle:
        demos = list(handle['data'])
        if len(demos) != 1:
            raise ValueError(f'{path}: expected exactly one seed demonstration')
        demo = handle['data'][demos[0]]
        info = demo['obs/datagen_info']
        start = np.flatnonzero(info[f'subtask_start_signals/{skill}'][:, 0])
        end = np.flatnonzero(info[f'subtask_term_signals/{skill}'][:, 0])
        if not len(start) or not len(end):
            raise ValueError(f'{path}: missing {skill} subtask signals')
        diagnostics = demo['obs/diagnostics']
        return SeedSkill(
            target=info['target_eef_pose/link6'][:], knob=info['object_pose/knob'][:],
            start=int(start[0]), end=int(end[0]) + 1,
            phase_id=diagnostics['phase_id'][:],
            phase_names=tuple(json.loads(diagnostics.attrs['phase_names'])))


def annotate_generated(demo, seed: SeedSkill) -> dict:
    """Align an open HDF5 generated demo and return labels plus audit metrics."""
    alignment = align_to_seed(
        demo['obs/datagen_info/target_eef_pose/link6'][:], demo['obs/datagen_info/object_pose/knob'][:],
        seed.target, seed.knob, seed_start=seed.start, seed_end=seed.end,
        transition_index=demo['obs/diagnostics/transition_index'][:])
    labels = stage_labels(alignment, seed.phase_id, seed.phase_names)
    state = demo['obs/joint_pos'][:]
    first_plan_end = alignment.recovery_start or alignment.replay_start
    recovery = None
    if alignment.recovery_start is not None:
        recovery = transition_metrics(state[alignment.recovery_start:],
                                      alignment.replay_start - alignment.recovery_start)
    return {'alignment': alignment, 'stage': labels,
            'transition': transition_metrics(state, first_plan_end), 'recovery': recovery}
