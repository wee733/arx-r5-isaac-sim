# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Explicit agent review of source recordings; human approval stays pending."""

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import numpy as np

from ..files import write_json
from ..quality import knob_hold_contact_summary
from ..runtime import video_tool
from ..contract import (
    EDGE_REGRASP, KNOB_HOLD_OPEN, behavior_family_from_metadata, final_contact_object,
)
from .seed import DoorSeed, _validate_phase_order, load_attempt, trajectory_sha256


REVIEW_SCHEMA = 'arx-door-agent-seed-review-v1'
VISUAL_SCOPE = 'knob_grasp_unlock_gap_edge_grasp_final_hold'
KNOB_VISUAL_SCOPE = 'knob_grasp_unlock_gap_knob_pull_final_hold'


def _visual_scope(seed: DoorSeed) -> str:
    return (KNOB_VISUAL_SCOPE if behavior_family_from_metadata(seed.metadata) == KNOB_HOLD_OPEN
            else VISUAL_SCOPE)


def physical_seed_evidence(seed: DoorSeed) -> dict:
    """Recompute success from the full NPZ, never from its success label alone."""
    metadata = seed.metadata
    family = behavior_family_from_metadata(metadata)
    if metadata.get('success') is not True or metadata.get('safety_events', []) != []:
        raise ValueError('agent review requires physical success without safety events')
    count = len(seed.state)
    if metadata.get('fps') != 30 or not np.allclose(np.diff(seed.timestamp), 1 / 30,
                                                  rtol=0, atol=1e-7):
        raise ValueError('agent-reviewed seeds must be uniformly sampled at 30 Hz')
    if not np.isclose(seed.timestamp[0], 0, atol=1e-7):
        raise ValueError('agent-reviewed timestamps must start at zero')
    phases = seed.phase.astype(str)
    if phases.shape != (count,):
        raise ValueError('phase frame count must match the trajectory')
    blocks = tuple(phases[np.r_[True, phases[1:] != phases[:-1]]])
    if blocks != _validate_phase_order(phases, family):
        raise ValueError('agent review requires complete, ordered, contiguous phase blocks')
    limits = {
        'fps': 30, 'final_door_angle_deg_exclusive': 29.0,
        'finger_force_n': 3.0, 'unlock_handle_angle_deg': 43.5,
        'unlock_latch_m': .0154, 'knob_gap_angle_deg': 10.0,
        'min_camera_clearance_m': .01,
    }
    arm_limit = metadata.get('arm_contact_limit_N')
    if (not isinstance(arm_limit, (int, float)) or not np.isfinite(arm_limit)
            or arm_limit <= 0):
        raise ValueError('episode must declare a finite positive arm contact limit')
    limits['max_arm_contact_n'] = min(float(arm_limit), 40.0)
    clearance_limit = metadata.get('viewpoint_guard', {}).get('minimum_clearance_m', .01)
    if not isinstance(clearance_limit, (int, float)) or not np.isfinite(clearance_limit):
        raise ValueError('invalid episode camera clearance limit')
    limits['min_camera_clearance_m'] = max(.01, float(clearance_limit))
    angle = np.rad2deg(seed.door[:, 0])
    handle = np.rad2deg(seed.door[:, 1])
    knob_forces = np.linalg.norm(seed.contacts[:, :, 0], axis=-1)
    knob = np.min(knob_forces, axis=1)

    def event(flags, label):
        indices = np.flatnonzero(flags)
        if not len(indices):
            raise ValueError(f'agent review is missing physical {label} evidence')
        return int(indices[0])

    knob_index = event((phases == 'grasp_knob') & (knob >= 3), 'knob grasp')
    unlock_index = event((phases == 'unlock_round_knob') & (handle >= 43.5)
                         & (seed.door[:, 2] >= .0154), 'knob unlock and latch retraction')
    gap_index = event((phases == 'pull_knob_for_gap') & (angle >= 10) & (knob >= 3), 'door gap')
    if not knob_index < unlock_index < gap_index:
        raise ValueError('physical knob/unlock/gap events are out of order')
    hold_evidence = None
    if family == KNOB_HOLD_OPEN:
        # The physical opening fixes the anchor even if a finger has already
        # lost contact. A later re-touch must never postpone this check.
        first_opening_gap = event(angle >= 10, 'opening gap')
        hold_evidence = knob_hold_contact_summary(knob_forces, first_opening_gap)
        if hold_evidence['knob_hold_contact_fraction'] != 1.0:
            raise ValueError(
                'knob contact not continuously held from gap to final: '
                f"first_loss_frame={hold_evidence['knob_hold_first_loss_frame']}, "
                f"contact_fraction={hold_evidence['knob_hold_contact_fraction']:.9g}; "
                'both finger forces must remain at least 3 N')
    # This is the same closed panel/hinge frame used by the seed pose converter:
    # panel rotation is Rz(-hinge), so its outward Y normal is [sin, cos, 0].
    theta = float(seed.door[-1, 0])
    normal = np.array([np.sin(theta), np.cos(theta), 0.0])
    final_forces = seed.contacts[-1, :, 0 if family == KNOB_HOLD_OPEN else 1]
    components = final_forces @ normal
    final_force = np.linalg.norm(final_forces, axis=1)
    camera_min = float(seed.camera_clearance.min())
    arm_max = float(np.linalg.norm(seed.arm_contacts, axis=-1).max())
    if seed.door[-1, 0] <= np.deg2rad(29):
        raise ValueError('final physical door angle must be above 29 degrees')
    if family == EDGE_REGRASP:
        if (np.min(final_force) < 3 or np.min(np.abs(components)) < 3
                or components[0] * components[1] >= 0):
            raise ValueError('final edge grip must have opposing panel-normal forces of at least 3 N')
    elif np.min(final_force) < 3 or np.dot(final_forces[0], final_forces[1]) >= 0:
        raise ValueError('final knob grip requires both finger forces of at least 3 N and opposing vectors')
    if camera_min < limits['min_camera_clearance_m']:
        raise ValueError('camera clearance is below the physical review threshold')
    if arm_max > limits['max_arm_contact_n']:
        raise ValueError('non-finger arm contact exceeds the episode/40 N review limit')
    evidence = {
        'passed': True, 'frames': count, 'fps': 30, 'phase_order': list(blocks),
        'behavior_family': family, 'final_contact_object': final_contact_object(family),
        'thresholds': limits,
        'knob_grasp_frame': knob_index, 'unlock_frame': unlock_index, 'gap_frame': gap_index,
        'final_door_angle_deg': float(angle[-1]),
        'final_finger_force_n': final_force.tolist(),
        'min_camera_clearance_m': camera_min, 'max_arm_contact_n': arm_max,
    }
    if family == EDGE_REGRASP:
        evidence['final_panel_normal_force_n'] = components.tolist()
    else:
        evidence['final_knob_force_dot_n2'] = float(np.dot(final_forces[0], final_forces[1]))
        evidence.update(hold_evidence)
    return evidence


def _artifact_path(seed: DoorSeed, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else seed.path / path).resolve()


def _record_path(seed: DoorSeed, path: Path) -> str:
    try:
        return path.relative_to(seed.path).as_posix()
    except ValueError:
        return str(path)


def decoded_seed_videos(seed: DoorSeed) -> dict:
    """Ask FFProbe to decode/count every video frame rather than read nb_frames."""
    tool, environment = video_tool('ffprobe')
    videos = {}
    for camera in ('wrist', 'overview'):
        path = seed.path / f'{camera}.mp4'
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'missing {camera} source video')
        result = subprocess.run([
            tool, '-v', 'error', '-err_detect', 'explode', '-count_frames',
            '-select_streams', 'v:0', '-show_entries',
            'stream=nb_read_frames,avg_frame_rate,width,height', '-of', 'json', str(path),
        ], env=environment, text=True, capture_output=True, timeout=180)
        if result.returncode or result.stderr.strip():
            raise ValueError(f'{camera} video decoding failed: {result.stderr[:400]}')
        streams = json.loads(result.stdout).get('streams', [])
        if len(streams) != 1:
            raise ValueError(f'{camera} video must contain a decodable video stream')
        stream = streams[0]
        try:
            count = int(stream['nb_read_frames'])
            fps = float(Fraction(stream['avg_frame_rate']))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise ValueError(f'{camera} video has no decoded frame count/FPS') from error
        if count != len(seed.state) or not np.isclose(fps, 30, rtol=0, atol=1e-6):
            raise ValueError(f'{camera} decoded frames/FPS do not match the 30 Hz NPZ')
        videos[camera] = {'path': f'{camera}.mp4', 'decoded_frames': count, 'fps': fps,
                          'width': int(stream['width']), 'height': int(stream['height']),
                          'size_bytes': path.stat().st_size}
    return videos


def visual_seed_evidence(seed: DoorSeed, paths) -> list[dict]:
    """Verify supplied review images exist and decode; never infer visual approval."""
    from PIL import Image

    evidence = []
    seen = set()
    for value in paths:
        path = _artifact_path(seed, str(value))
        if path in seen:
            raise ValueError('visual review evidence paths must be distinct')
        seen.add(path)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'visual review image is missing: {path}')
        try:
            with Image.open(path) as image:
                width, height = image.size
                format_name = image.format
                image.verify()
        except (OSError, ValueError) as error:
            raise ValueError(f'visual review image cannot be decoded: {path}') from error
        if format_name not in ('PNG', 'JPEG', 'WEBP') or width <= 0 or height <= 0:
            raise ValueError(f'visual evidence must be an actual raster review image: {path}')
        item = {'path': _record_path(seed, path), 'width': width, 'height': height,
                'format': format_name, 'size_bytes': path.stat().st_size}
        if path.parent.name in ('wrist', 'overview') and path.stem.isdigit():
            index = int(path.stem)
            if not 0 <= index < len(seed.state):
                raise ValueError('visual evidence frame is outside the source trajectory')
            item.update(camera=path.parent.name, frame_index=index, phase=str(seed.phase[index]))
        evidence.append(item)
    if not evidence:
        raise ValueError('agent review requires actual visual evidence images')
    return evidence


def _trajectory_evidence(seed: DoorSeed) -> dict:
    # One integrity binding at the explicit review boundary; videos/images are
    # checked by real decoding and recorded dimensions/counts, without hashes.
    return {'path': 'trajectory.npz', 'frames': len(seed.state), 'fps': 30,
            'sha256': trajectory_sha256(seed)}


def validate_agent_reviewed_seed(seed: DoorSeed) -> dict:
    """Recheck the signed-off evidence while keeping human approval pending."""
    metadata = seed.metadata
    family = behavior_family_from_metadata(metadata)
    if (metadata.get('quality_status') != 'agent_reviewed_seed'
            or metadata.get('review_required') is not True
            or metadata.get('human_review_status') != 'pending'
            or metadata.get('review_status') != 'pending'):
        raise ValueError('agent-reviewed seed must preserve pending human review')
    path = seed.path / 'agent_review.json'
    if not path.is_file():
        raise ValueError('agent-reviewed seed is missing agent_review.json')
    review = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(review, dict) or review.get('schema') != REVIEW_SCHEMA or review.get('reviewer_kind') != 'agent'
            or review.get('decision') != 'agent_reviewed_seed'
            or review.get('human_review_status') != 'pending'):
        raise ValueError('unsupported or inconsistent agent review declaration')
    visual = review.get('visual_review', {})
    if not isinstance(visual, dict):
        raise ValueError('agent visual review declaration must be an object')
    notes = visual.get('notes')
    if (visual.get('confirmed') is not True or visual.get('scope') != _visual_scope(seed)
            or not isinstance(notes, str) or len(notes.strip()) < 10):
        raise ValueError('agent review requires an explicit substantive visual review statement')
    physical = physical_seed_evidence(seed)
    saved_physical = review.get('physical_evidence')
    # Pre-family edge reviews remain valid against precisely their original
    # physical evidence. This exception never applies to the new knob family.
    if family == EDGE_REGRASP and isinstance(saved_physical, dict) and 'behavior_family' not in saved_physical:
        physical = {key: value for key, value in physical.items()
                    if key not in ('behavior_family', 'final_contact_object')}
    if family == KNOB_HOLD_OPEN and isinstance(saved_physical, dict):
        # Existing knob reviews retain their exact saved values. The full
        # gap-to-final continuity gate above is always recomputed first;
        # only newly derived diagnostics may be absent from the old review.
        added_fields = ('knob_hold_start_frame', 'knob_hold_checked_frames',
                        'knob_hold_first_loss_frame', 'knob_hold_min_finger_force_n')
        physical = {key: value for key, value in physical.items()
                    if key not in added_fields or key in saved_physical}
    if (review.get('trajectory') != _trajectory_evidence(seed)
            or saved_physical != physical
            or review.get('video_evidence') != decoded_seed_videos(seed)):
        raise ValueError('agent review physical/trajectory/video evidence no longer matches the recording')
    for key in ('runtime_profile', 'runtime', 'urdf_path', 'expert_parameters'):
        if review.get(key) != seed.metadata.get(key):
            raise ValueError(f'agent review {key} differs from the source episode')
    if (review.get('behavior_family', EDGE_REGRASP) != family
            or review.get('final_contact_object', 'panel') != final_contact_object(family)):
        raise ValueError('agent review behavior family/contact object differs from the source episode')
    saved_images = visual.get('evidence', [])
    if (not isinstance(saved_images, list)
            or any(not isinstance(item, dict) or not item.get('path') for item in saved_images)
            or saved_images != visual_seed_evidence(seed, [item['path'] for item in saved_images])):
        raise ValueError('agent review visual evidence no longer matches its declaration')
    return review


def mark_agent_reviewed_seed(path, *, confirm_visual_review: bool,
                             visual_evidence, visual_review_notes: str) -> dict:
    """Write agent review only after an explicit visual declaration and hard evidence."""
    if not confirm_visual_review or len(visual_review_notes.strip()) < 10:
        raise ValueError('explicit agent visual confirmation and substantive review notes are required')
    seed = load_attempt(path)
    if seed.metadata.get('quality_status') in ('user_approved_seed', 'agent_reviewed_seed'):
        raise ValueError('review does not overwrite an already approved/reviewed source')
    destination = seed.path / 'agent_review.json'
    if destination.exists():
        raise ValueError('existing agent review is retained; choose an unreviewed source')
    physical = physical_seed_evidence(seed)
    videos = decoded_seed_videos(seed)
    images = visual_seed_evidence(seed, visual_evidence)
    family = behavior_family_from_metadata(seed.metadata)
    review = {
        'schema': REVIEW_SCHEMA, 'reviewer_kind': 'agent', 'decision': 'agent_reviewed_seed',
        'created_at': datetime.now(timezone.utc).isoformat(), 'human_review_status': 'pending',
        'trajectory': _trajectory_evidence(seed), 'physical_evidence': physical,
        'video_evidence': videos,
        'behavior_family': family, 'final_contact_object': final_contact_object(family),
        'visual_review': {'confirmed': True, 'scope': _visual_scope(seed),
                          'notes': visual_review_notes.strip(), 'evidence': images},
        'runtime_profile': seed.metadata.get('runtime_profile'),
        'runtime': seed.metadata.get('runtime'),
        'urdf_path': seed.metadata.get('urdf_path'),
        'expert_parameters': seed.metadata.get('expert_parameters'),
    }
    metadata = {**seed.metadata, 'quality_status': 'agent_reviewed_seed',
                'review_required': True, 'review_status': 'pending',
                'human_review_status': 'pending', 'agent_review_path': 'agent_review.json'}
    # Publish evidence first. An interruption can leave an unapproved recording,
    # but cannot make a source eligible without its evidence file.
    for target, value in ((destination, review), (seed.path / 'episode.json', metadata)):
        write_json(target, value, allow_nan=True)
    return review
