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

"""Behavior tests for ROS-free episode timestamp alignment."""

from dataclasses import dataclass

from arx_r5_isaac_sim_bringup.vla.episode_events import (
    EpisodePhase,
    EpisodeStatus,
)
from arx_r5_isaac_sim_bringup.vla.timeline import (
    EpisodeTimeline,
    EventSample,
    TimelineNotFinalizedError,
)

import pytest


def _event(stamp_ns, phase, status=EpisodeStatus.RUNNING, value=None):
    return EventSample(
        stamp_ns=stamp_ns,
        phase=phase,
        status=status,
        value=value,
    )


def _start(timeline, stamp_ns=100):
    timeline.add_event(_event(
        stamp_ns,
        EpisodePhase.RESET,
        EpisodeStatus.STARTED,
        value='started',
    ))


def _succeed(timeline, stamp_ns=200):
    timeline.add_event(_event(
        stamp_ns,
        EpisodePhase.VERIFY_PLACE,
        EpisodeStatus.SUCCEEDED,
        value='succeeded',
    ))


def test_joint_state_is_latest_sample_not_newer_than_camera_stamp():
    """A future feedback sample must never leak into the current observation."""
    timeline = EpisodeTimeline()
    _start(timeline)
    timeline.add_joint_state(140, 'future')
    timeline.add_joint_state(90, 'old')
    timeline.add_joint_state(120, 'causal')
    timeline.add_frame(130, 'camera-130')
    _succeed(timeline)

    aligned = timeline.finalize()

    assert len(aligned) == 1
    assert aligned[0].frame == 'camera-130'
    assert aligned[0].joint_state is not None
    assert aligned[0].joint_state.stamp_ns == 120
    assert aligned[0].joint_state.value == 'causal'


def test_late_phase_event_relabels_an_already_buffered_frame():
    """Callback arrival order cannot freeze a frame to the previous phase."""
    timeline = EpisodeTimeline()
    _start(timeline)
    timeline.add_frame(160, 'camera-arrived-first')
    timeline.add_event(_event(
        150,
        EpisodePhase.APPROACH,
        value='approach-arrived-late',
    ))
    _succeed(timeline)

    aligned = timeline.finalize()

    assert aligned[0].event is not None
    assert aligned[0].event.phase == EpisodePhase.APPROACH
    assert aligned[0].event.value == 'approach-arrived-late'


def test_terminal_arrival_does_not_close_before_late_tail_frame():
    """A pre-terminal camera sample may arrive after the terminal callback."""
    timeline = EpisodeTimeline()
    _start(timeline)
    _succeed(timeline, stamp_ns=200)

    assert timeline.terminal_event is not None
    assert not timeline.finalized

    timeline.add_joint_state(190, 'tail-state')
    timeline.add_frame(200, 'tail-at-terminal-stamp')
    timeline.add_frame(201, 'genuinely-post-terminal')
    assert timeline.pending_frame_count == 2

    aligned = timeline.finalize()

    assert [frame.frame for frame in aligned] == ['tail-at-terminal-stamp']
    assert aligned[0].joint_state is not None
    assert aligned[0].joint_state.value == 'tail-state'


def test_started_boundary_is_strict_while_terminal_boundary_is_inclusive():
    """Setup-tick frames are excluded; terminal-tick frames are retained."""
    timeline = EpisodeTimeline()
    _start(timeline, stamp_ns=100)
    timeline.add_frame(100, 'setup-tick')
    timeline.add_frame(200, 'terminal-tick')
    _succeed(timeline, stamp_ns=200)

    assert [frame.frame for frame in timeline.finalize()] == ['terminal-tick']


def test_equal_stamp_events_use_episode_phase_order_not_arrival_order():
    """The furthest phase at one clock tick wins deterministically."""
    timeline = EpisodeTimeline()
    _start(timeline)
    timeline.add_event(_event(
        150,
        EpisodePhase.GRASP,
        value='grasp-arrived-first',
    ))
    timeline.add_event(_event(
        150,
        EpisodePhase.APPROACH,
        value='approach-arrived-later',
    ))
    timeline.add_frame(160, 'camera')
    _succeed(timeline)

    aligned = timeline.finalize()

    assert aligned[0].event is not None
    assert aligned[0].event.phase == EpisodePhase.GRASP
    assert aligned[0].event.value == 'grasp-arrived-first'


def test_frames_are_emitted_in_timestamp_order_after_out_of_order_arrival():
    """A dataset writer receives monotonically ordered camera payloads."""
    timeline = EpisodeTimeline()
    _start(timeline)
    timeline.add_frame(170, 'later')
    timeline.add_frame(130, 'earlier')
    _succeed(timeline)

    assert [frame.frame for frame in timeline.finalize()] == [
        'earlier',
        'later',
    ]


def test_finalize_requires_an_explicit_terminal_event():
    """Receiving a few frames is not enough to commit an open episode."""
    timeline = EpisodeTimeline()
    _start(timeline)
    timeline.add_frame(120, 'camera')

    with pytest.raises(TimelineNotFinalizedError):
        timeline.finalize()


@dataclass
class _EpisodeEventLike:
    phase: str
    status: str
    stamp_sec: float


def test_episode_event_wrapper_preserves_original_payload():
    """Recorder integration can retain every field of its EpisodeEvent."""
    original = _EpisodeEventLike(
        phase=EpisodePhase.OBSERVE,
        status=EpisodeStatus.RUNNING,
        stamp_sec=1.25,
    )

    sample = EventSample.from_episode_event(original)

    assert sample.stamp_ns == 1_250_000_000
    assert sample.phase == EpisodePhase.OBSERVE
    assert sample.status == EpisodeStatus.RUNNING
    assert sample.value is original
