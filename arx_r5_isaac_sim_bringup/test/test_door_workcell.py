# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Verify measured geometry and reproducible episode placement."""

import math

import pytest

from arx_r5_isaac_sim_bringup.door_workcell import sample_base_pose


@pytest.mark.parametrize('angle', [-30, 0, 30])
def test_measured_arc(angle):
    """Endpoints retain the horizontal radius and base installation height."""
    pose = sample_base_pose((0, 0, 0.928), angle_deg=angle)
    x, y, z = pose.position
    assert math.hypot(x, y) == pytest.approx(0.59)
    assert z == 0.63
    assert x == pytest.approx(0.59 * math.sin(math.radians(angle)))
    assert y == pytest.approx(-0.59 * math.cos(math.radians(angle)))
    yaw = 2 * math.atan2(pose.quaternion_xyzw[2], pose.quaternion_xyzw[3])
    assert x * math.cos(yaw) + y * math.sin(yaw) == pytest.approx(-0.59)


def test_reproducible_episodes():
    """Random access to an episode does not depend on previous sample calls."""
    first = sample_base_pose((1, 2, 1), seed=42, episode_index=10)
    sample_base_pose((1, 2, 1), seed=42, episode_index=99)
    assert first == sample_base_pose((1, 2, 1), seed=42, episode_index=10)
    assert first != sample_base_pose((1, 2, 1), seed=42, episode_index=11)


def test_rotated_normal_and_fixed_yaw():
    """The sampler also handles doors rotated in the world."""
    pose = sample_base_pose((1, 2, 1), front_normal_xy=(1, 0),
                            angle_deg=30, yaw_mode='fixed')
    assert pose.position[:2] == pytest.approx((1 + 0.59 * math.cos(math.pi / 6), 2.295))
    assert pose.quaternion_xyzw == pytest.approx((0, 0, -1, 0), abs=1e-10)


@pytest.mark.parametrize('kwargs', [
    {'radius_m': -1}, {'angle_deg': 31}, {'base_height_m': float('nan')},
    {'front_normal_xy': (0, 0)}, {'episode_index': -1}, {'yaw_mode': 'unknown'},
])
def test_reject_invalid_geometry(kwargs):
    """Malformed inputs fail before an invalid scene is authored."""
    with pytest.raises(ValueError):
        sample_base_pose((0, 0, 1), **kwargs)
