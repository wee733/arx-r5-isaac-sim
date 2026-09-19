# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for destination-tag sequence and freshness policy."""

from pathlib import Path

from arx_r5_isaac_sim_bringup.tag_goal_policy import (
    TargetSequenceGate,
    TranslationMedianGate,
)

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_target_loss_invalidates_pose_and_rejects_old_async_result():
    """A completed TF future may never cross a visibility discontinuity."""
    gate = TargetSequenceGate()
    first_generation = gate.observe_target()
    assert gate.accept_pose(first_generation, 1_000_000_000)

    gate.miss_target()
    assert gate.generation > first_generation
    assert not gate.accept_pose(first_generation, 1_100_000_000)
    assert not gate.ready(
        required_frames=1,
        now_ns=1_100_000_000,
        ttl_ns=500_000_000,
    )

    second_generation = gate.observe_target()
    assert second_generation > first_generation
    assert not gate.accept_pose(first_generation, 1_200_000_000)


def test_ready_requires_current_stable_sequence_and_fresh_sim_stamp():
    """A current-generation pose must also meet stability and sim-time TTL."""
    gate = TargetSequenceGate()
    generation = gate.observe_target()
    for _ in range(4):
        assert gate.observe_target() == generation
    assert gate.accept_pose(generation, 2_000_000_000)

    assert gate.ready(
        required_frames=5,
        now_ns=2_500_000_000,
        ttl_ns=500_000_000,
    )
    assert not gate.ready(
        required_frames=6,
        now_ns=2_500_000_000,
        ttl_ns=500_000_000,
    )
    assert not gate.ready(
        required_frames=5,
        now_ns=2_500_000_001,
        ttl_ns=500_000_000,
    )
    assert not gate.ready(
        required_frames=5,
        now_ns=1_999_999_999,
        ttl_ns=500_000_000,
    )


def test_translation_gate_returns_component_median_for_stable_window():
    """Small PnP noise is reduced without averaging toward one outlier."""
    gate = TranslationMedianGate(window_size=5, max_spread_m=0.01)
    samples = (
        (0.100, 0.200, 0.300),
        (0.101, 0.198, 0.303),
        (0.099, 0.201, 0.299),
        (0.102, 0.199, 0.301),
        (0.098, 0.202, 0.302),
    )

    for index, sample in enumerate(samples[:-1]):
        assert gate.observe(
            generation=1,
            stamp_ns=1_000_000_000 + index,
            translation=sample,
        ) is None
    estimate = gate.observe(
        generation=1,
        stamp_ns=1_000_000_004,
        translation=samples[-1],
    )

    assert estimate == pytest.approx((0.100, 0.200, 0.301))
    assert gate.spread_m is not None
    assert gate.spread_m < 0.01


def test_translation_gate_rejects_twenty_millimetre_depth_swing():
    """A 20 mm PnP depth swing must not become an actionable drop pose."""
    gate = TranslationMedianGate(window_size=5, max_spread_m=0.01)
    depths = (0.300, 0.301, 0.320, 0.299, 0.302)

    estimate = None
    for index, depth in enumerate(depths):
        estimate = gate.observe(
            generation=7,
            stamp_ns=2_000_000_000 + index,
            translation=(0.1, 0.2, depth),
        )

    assert estimate is None
    assert gate.spread_m == pytest.approx(0.021)


def test_translation_gate_recovers_only_after_outlier_leaves_window():
    """A rolling window cannot reuse an old median while it is unstable."""
    gate = TranslationMedianGate(window_size=3, max_spread_m=0.01)
    observations = (
        (0, (0.0, 0.0, 0.300)),
        (1, (0.0, 0.0, 0.320)),
        (2, (0.0, 0.0, 0.301)),
        (3, (0.0, 0.0, 0.302)),
        (4, (0.0, 0.0, 0.303)),
    )

    estimates = [
        gate.observe(
            generation=1,
            stamp_ns=3_000_000_000 + offset,
            translation=translation,
        )
        for offset, translation in observations
    ]

    assert estimates[:4] == [None, None, None, None]
    assert estimates[4] == pytest.approx((0.0, 0.0, 0.302))


def test_translation_gate_resets_on_visibility_generation_and_clock_rewind():
    """Neither a tag loss nor a simulator restart may share old samples."""
    gate = TranslationMedianGate(window_size=3, max_spread_m=0.01)
    for stamp_ns in (10, 11, 12):
        estimate = gate.observe(
            generation=1,
            stamp_ns=stamp_ns,
            translation=(0.1, 0.2, 0.3),
        )
    assert estimate == pytest.approx((0.1, 0.2, 0.3))

    assert gate.observe(
        generation=2,
        stamp_ns=13,
        translation=(0.1, 0.2, 0.3),
    ) is None
    assert gate.sample_count == 1
    assert gate.observe(
        generation=2,
        stamp_ns=2,
        translation=(0.1, 0.2, 0.3),
    ) is None
    assert gate.sample_count == 1


@pytest.mark.parametrize(
    ('window_size', 'max_spread_m'),
    ((0, 0.01), (5, 0.0), (5, float('inf'))),
)
def test_translation_gate_rejects_invalid_policy(
    window_size,
    max_spread_m,
):
    """Bad settings fail at startup rather than silently bypassing."""
    with pytest.raises(ValueError):
        TranslationMedianGate(
            window_size=window_size,
            max_spread_m=max_spread_m,
        )


def test_goal_client_wires_node_clock_generation_and_ttl():
    """Use the tested gate and a sim-clock-aware TF buffer."""
    source = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'tag_goal_client.py'
    ).read_text(encoding='utf-8')
    launch_source = (
        PACKAGE_ROOT / 'launch' / 'arx_r5a_authored_usd_demo.launch.py'
    ).read_text(encoding='utf-8')

    assert 'Buffer(node=self)' in source
    assert 'self._invalidate_target_sequence()' in source
    assert 'generation, header, target_pose' in source
    assert 'self._target_sequence.ready(' in source
    assert "declare_parameter('drop_pose_ttl_sec', 0.5)" in source
    assert "'max_translation_spread_m',\n                0.01," in source
    assert 'translation=drop_translation' in source
    assert "'drop_pose_ttl_sec': float(" in launch_source
    declaration = (
        "DeclareLaunchArgument('drop_pose_ttl_sec', default_value='0.5')"
    )
    assert declaration in launch_source
