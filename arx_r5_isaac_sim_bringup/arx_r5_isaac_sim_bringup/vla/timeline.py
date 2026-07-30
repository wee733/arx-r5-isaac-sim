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

"""
ROS-free timestamp alignment for one recorded episode.

The ROS callbacks that deliver camera images, joint states and episode events
run on different subscriptions.  Arrival order is therefore not a temporal
contract.  :class:`EpisodeTimeline` keeps the raw samples until the caller has
received a terminal event and explicitly calls :meth:`finalize`; alignment is
then performed from simulation timestamps rather than callback order.

This module intentionally contains no ROS imports.  A ROS recorder can wrap an
``EpisodeEvent`` in :class:`EventSample`, add its ``Frame`` payload and call
``finalize`` after a short post-terminal drain period.
"""

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence, Tuple

from arx_r5_isaac_sim_bringup.vla.episode_events import (
    EpisodePhase,
    EpisodeStatus,
)


class TimelineError(ValueError):
    """Raised when a timeline sample violates the episode contract."""


class TimelineNotFinalizedError(RuntimeError):
    """Raised when finalization is requested before a terminal event."""


@dataclass(frozen=True)
class TimestampedValue:
    """An opaque value captured at one simulation timestamp."""

    stamp_ns: int
    value: Any
    sequence: int = 0


@dataclass(frozen=True)
class EventSample:
    """
    A phase transition with an integer simulation timestamp.

    ``value`` may hold the original ``EpisodeEvent`` so an integration layer
    can retain all of its instruction/detail/extra fields without making this
    component depend on ROS message types.
    """

    stamp_ns: int
    phase: str
    status: str
    value: Any = None
    sequence: int = 0

    @classmethod
    def from_episode_event(
        cls,
        event: Any,
        *,
        stamp_ns: Optional[int] = None,
    ) -> 'EventSample':
        """
        Wrap an ``EpisodeEvent``-like object without importing ROS.

        The preferred path supplies ``stamp_ns`` obtained from an exact ROS
        header.  For the current JSON event contract, ``stamp_sec`` is accepted
        as a compatibility fallback; conversion is done once at the boundary.
        """
        if stamp_ns is None:
            try:
                stamp_sec = float(event.stamp_sec)
            except (AttributeError, TypeError, ValueError) as error:
                raise TimelineError(
                    'event requires stamp_ns or a finite stamp_sec'
                ) from error
            if not math.isfinite(stamp_sec):
                raise TimelineError('event stamp_sec must be finite')
            stamp_ns = int(round(stamp_sec * 1e9))
        return cls(
            stamp_ns=int(stamp_ns),
            phase=str(event.phase),
            status=str(event.status),
            value=event,
        )


@dataclass(frozen=True)
class AlignedFrame:
    """A camera payload resolved against causal state and phase histories."""

    stamp_ns: int
    frame: Any
    joint_state: Optional[TimestampedValue]
    event: Optional[EventSample]


class EpisodeTimeline:
    """
    Collect and causally align samples for one episode.

    Samples may arrive in any order.  ``add_frame`` and ``add_joint_state``
    remain legal after a terminal event, because DDS can deliver a camera frame
    captured before that event after the terminal callback.  Only ``finalize``
    closes the timeline and applies the start/terminal timestamp boundaries.
    """

    def __init__(
        self,
        *,
        phase_order: Sequence[str] = EpisodePhase.ORDER,
    ) -> None:
        """Create an empty timeline with an explicit same-stamp phase order."""
        phase_order = tuple(phase_order)
        self._phase_rank = {
            str(phase): index for index, phase in enumerate(phase_order)
        }
        if len(self._phase_rank) != len(phase_order):
            raise TimelineError('phase_order must contain unique phases')
        self._frames: list[TimestampedValue] = []
        self._joint_states: list[TimestampedValue] = []
        self._events: list[EventSample] = []
        self._start_event: Optional[EventSample] = None
        self._terminal_event: Optional[EventSample] = None
        self._sequence = 0
        self._finalized = False
        self._result: Optional[Tuple[AlignedFrame, ...]] = None

    # ---------------------------------------------------------- validation --

    @staticmethod
    def _stamp(value: int) -> int:
        """Validate and normalize a non-negative integer nanosecond stamp."""
        if isinstance(value, bool):
            raise TimelineError('timestamps must be integer nanoseconds')
        try:
            stamp = int(value)
        except (TypeError, ValueError) as error:
            raise TimelineError(
                'timestamps must be integer nanoseconds'
            ) from error
        if stamp < 0:
            raise TimelineError('timestamps must not be negative')
        return stamp

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError('episode timeline has already been finalized')

    def _validate_phase(self, phase: str) -> str:
        normalized = str(phase)
        if normalized not in self._phase_rank:
            raise TimelineError(f'unknown episode phase: {normalized!r}')
        return normalized

    # -------------------------------------------------------------- ingest --

    def add_frame(self, stamp_ns: int, frame: Any) -> None:
        """Retain one camera-complete frame; arrival order is immaterial."""
        self._ensure_open()
        sample = TimestampedValue(
            stamp_ns=self._stamp(stamp_ns),
            value=frame,
            sequence=self._next_sequence(),
        )
        self._frames.append(sample)

    def add_joint_state(self, stamp_ns: int, joint_state: Any) -> None:
        """Retain a state sample for causal lookup at frame finalization."""
        self._ensure_open()
        sample = TimestampedValue(
            stamp_ns=self._stamp(stamp_ns),
            value=joint_state,
            sequence=self._next_sequence(),
        )
        self._joint_states.append(sample)

    def add_event(self, event: EventSample) -> None:
        """Retain a phase event, including one received after its frame."""
        self._ensure_open()
        if not isinstance(event, EventSample):
            raise TimelineError('event must be an EventSample')
        phase = self._validate_phase(event.phase)
        status = str(event.status)
        if not status:
            raise TimelineError('event status must be non-empty')
        normalized = EventSample(
            stamp_ns=self._stamp(event.stamp_ns),
            phase=phase,
            status=status,
            value=event.value,
            sequence=self._next_sequence(),
        )
        if status == EpisodeStatus.STARTED:
            if self._start_event is not None:
                raise TimelineError('episode has more than one STARTED event')
            self._start_event = normalized
        if status in EpisodeStatus.TERMINAL:
            if self._terminal_event is not None:
                raise TimelineError('episode has more than one terminal event')
            self._terminal_event = normalized
        self._events.append(normalized)

    # ------------------------------------------------------------- queries --

    @property
    def terminal_event(self) -> Optional[EventSample]:
        """Return the terminal event, if one has arrived."""
        return self._terminal_event

    @property
    def pending_frame_count(self) -> int:
        """Return retained frames, including post-terminal late arrivals."""
        return len(self._frames)

    @property
    def finalized(self) -> bool:
        """Return whether :meth:`finalize` has closed this timeline."""
        return self._finalized

    def _latest_joint_state(
        self,
        frame_stamp_ns: int,
    ) -> Optional[TimestampedValue]:
        """Select the latest state whose stamp is not newer than the frame."""
        eligible = [
            sample for sample in self._joint_states
            if sample.stamp_ns <= frame_stamp_ns
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda sample: (sample.stamp_ns, sample.sequence),
        )

    def _event_at(self, frame_stamp_ns: int) -> Optional[EventSample]:
        """Select the latest event; same-stamp phases use ``phase_order``."""
        eligible = [
            event for event in self._events
            if event.stamp_ns <= frame_stamp_ns
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda event: (
                event.stamp_ns,
                self._phase_rank[event.phase],
                event.sequence,
            ),
        )

    def finalize(self) -> Tuple[AlignedFrame, ...]:
        """
        Resolve and close the episode after a terminal event is known.

        Frames are sorted by simulation timestamp.  The strict STARTED and
        inclusive terminal boundaries intentionally retain a frame captured at
        the terminal stamp even when DDS delivered it after the terminal
        event, while excluding setup frames stamped on the STARTED event.
        Frames outside ``(STARTED, terminal]`` are excluded from the returned result but
        remain inspectable through the pre-finalize count for diagnostics.
        """
        if self._finalized:
            assert self._result is not None
            return self._result
        if self._terminal_event is None:
            raise TimelineNotFinalizedError(
                'cannot finalize before a terminal event arrives'
            )

        start_stamp = (
            self._start_event.stamp_ns
            if self._start_event is not None
            else 0
        )
        terminal_stamp = self._terminal_event.stamp_ns
        ordered_frames = sorted(
            self._frames,
            key=lambda sample: (sample.stamp_ns, sample.sequence),
        )
        aligned = []
        for sample in ordered_frames:
            if not start_stamp < sample.stamp_ns <= terminal_stamp:
                continue
            aligned.append(AlignedFrame(
                stamp_ns=sample.stamp_ns,
                frame=sample.value,
                joint_state=self._latest_joint_state(sample.stamp_ns),
                event=self._event_at(sample.stamp_ns),
            ))

        self._result = tuple(aligned)
        self._finalized = True
        return self._result


__all__ = [
    'AlignedFrame',
    'EpisodeTimeline',
    'EventSample',
    'TimelineError',
    'TimelineNotFinalizedError',
    'TimestampedValue',
]
