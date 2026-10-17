# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Off-nominal pre-grasp entrances that teach recovery toward the knob.

Rigid SkillGen replay gives every episode the same knob-relative approach, so
a policy never sees how to return once it is off that line. A recovery trial
first moves to a perturbed, still contact-free entrance (optionally with the
gripper partly closed, the failure seen in closed-loop rollouts), then plans a
collision-checked correction that rejoins the seed approach before contact.
Only the correction onward is a recovery demonstration; the deliberate
perturbation is identified per frame and can be cropped at export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

GENERATION_PROFILE_SCHEMA = 'arx-door-generation-profile-v1'


@dataclass(frozen=True)
class RecoveryProfile:
    """Sampling ranges in the knob frame; translation in metres, rotation in degrees."""

    probability: float
    translation_m: tuple[tuple[float, float], ...]
    rotation_deg: tuple[tuple[float, float], ...]
    gripper_m: tuple[float, float] | None
    rejoin_fraction: float
    # Straight correction clearance. Near the knob cuMotion's conservative
    # spheres sit within millimetres of the door, and its optimizer then
    # detours; 0 still rejects any sphere penetration.
    direct_clearance_m: float = 0.0

    @classmethod
    def from_json(cls, value: dict) -> RecoveryProfile:
        def ranges(key):
            pairs = tuple(tuple(float(bound) for bound in pair) for pair in value[key])
            if len(pairs) != 3 or any(len(pair) != 2 or pair[0] > pair[1] for pair in pairs):
                raise ValueError(f'recovery {key} requires three ordered [low, high] ranges')
            return pairs

        gripper = value.get('gripper_m')
        profile = cls(
            probability=float(value['probability']),
            translation_m=ranges('translation_m'),
            rotation_deg=ranges('rotation_deg'),
            gripper_m=None if gripper is None else (float(gripper[0]), float(gripper[1])),
            rejoin_fraction=float(value['rejoin_fraction']),
            direct_clearance_m=float(value.get('direct_clearance_m', 0.0)),
        )
        if not 0 <= profile.probability <= 1 or not 0 < profile.rejoin_fraction < 1:
            raise ValueError('recovery probability must be in [0, 1], rejoin_fraction in (0, 1)')
        if profile.gripper_m is not None and not 0 <= profile.gripper_m[0] <= profile.gripper_m[1]:
            raise ValueError('recovery gripper_m must be an ordered nonnegative range')
        return profile


def load_generation_profile(path: Path | None) -> dict:
    """Read an optional generation profile; ``None`` keeps the historical behaviour."""
    if path is None:
        return {}
    profile = json.loads(Path(path).read_text())
    if profile.get('schema') != GENERATION_PROFILE_SCHEMA:
        raise ValueError(f'{path}: generation profile schema must be {GENERATION_PROFILE_SCHEMA}')
    if 'recovery' in profile:
        RecoveryProfile.from_json(profile['recovery'])
    return profile


def trial_key(trial_seed: int, angle_deg: float, visual_seed: int) -> tuple[int, int, int]:
    """Identify one trial for sampling; a batch shares its trial seed across angles."""
    return int(trial_seed), round(float(angle_deg) * 1000) & 0xFFFFFFFF, int(visual_seed)


def sample_perturbation(profile: RecoveryProfile, key) -> dict | None:
    """Draw one reproducible perturbation per trial, or ``None`` (nominal).

    ``key`` is a non-negative int or a :func:`trial_key` tuple.
    """
    key = [int(key)] if np.isscalar(key) else [int(value) for value in key]
    rng = np.random.default_rng([*key, 0x5EED])
    if rng.random() >= profile.probability:
        return None
    translation = [rng.uniform(low, high) for low, high in profile.translation_m]
    rotation = [rng.uniform(low, high) for low, high in profile.rotation_deg]
    gripper = None if profile.gripper_m is None else rng.uniform(*profile.gripper_m)
    return {'translation_knob_m': translation, 'rotation_knob_deg': rotation,
            'transition_gripper_m': gripper, 'rejoin_fraction': profile.rejoin_fraction,
            'direct_clearance_m': profile.direct_clearance_m}


def perturbed_entrance(entrance_world, knob_world, perturbation: dict) -> np.ndarray:
    """Offset and rotate the entrance about link6, both along knob-frame axes."""
    entrance = np.asarray(entrance_world, dtype=np.float64)
    knob = np.asarray(knob_world, dtype=np.float64)
    if entrance.shape != (4, 4) or knob.shape != (4, 4):
        raise ValueError('entrance and knob poses must be 4x4 matrices')
    axes = knob[:3, :3]
    delta = Rotation.from_euler('xyz', perturbation['rotation_knob_deg'], degrees=True).as_matrix()
    result = entrance.copy()
    result[:3, :3] = axes @ delta @ axes.T @ entrance[:3, :3]
    result[:3, 3] = entrance[:3, 3] + axes @ np.asarray(perturbation['translation_knob_m'])
    return result


def knob_approach_mask(seed_path: Path, *, open_gripper_min=0.0275) -> np.ndarray:
    """Seed frames proven contact-free in the open-gripper knob approach."""
    import h5py

    with h5py.File(seed_path, 'r') as handle:
        demos = list(handle['data'])
        if len(demos) != 1:
            raise ValueError('recovery requires exactly one seed demonstration')
        demo = handle['data'][demos[0]]
        diagnostics = demo['obs/diagnostics']
        names = json.loads(diagnostics.attrs['phase_names'])
        phase = np.asarray(diagnostics['phase_id'][:])
        fingers = np.asarray(diagnostics['finger_contacts'][:])
        arms = np.asarray(diagnostics['arm_contacts'][:])
        actions = np.asarray(demo['actions'][:])
    return ((phase == names.index('approach_knob'))
            & (actions[:, 6] >= open_gripper_min)
            & (np.linalg.norm(fingers, axis=-1).max(axis=(1, 2)) <= 1e-3)
            & (np.linalg.norm(arms, axis=-1).max(axis=1) <= 1e-3))


def rejoin_index(noncontact_mask, fraction: float) -> int:
    """Pick a rejoin waypoint inside the contact-free prefix of a skill trajectory."""
    mask = np.asarray(noncontact_mask, dtype=bool)
    prefix = int(np.argmin(mask)) if not mask.all() else len(mask)
    if prefix < 3:
        raise ValueError('skill trajectory has no contact-free approach to rejoin')
    return min(prefix - 1, max(1, round(fraction * prefix)))
