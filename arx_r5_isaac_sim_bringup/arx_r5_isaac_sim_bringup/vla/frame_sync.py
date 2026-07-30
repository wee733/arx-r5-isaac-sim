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
Group per-camera images into synchronized frames.

Both Isaac Sim cameras are triggered by the same ``OnPhysicsStep`` node and
publish with ``useSystemTime=False``, so their images carry identical
simulation timestamps. Frames are therefore paired by exact stamp rather than
an approximate-time policy.

Kept free of ROS message types so the pairing rule is unit-testable without a
sourced environment; it only reads ``header.stamp``.
"""

from bisect import bisect_left, bisect_right, insort_right
from collections import deque
from dataclasses import dataclass, field
import math
from statistics import median
from typing import Any, Dict, Optional, Sequence, Tuple

from arx_r5_isaac_sim_bringup.contracts import INDEPENDENT_JOINTS


# Isaac Sim cameras are triggered by the same physics step and use simulation
# time, so collection must reject even sub-millisecond timestamp differences.
DEFAULT_STAMP_TOLERANCE_NS = 0
DEFAULT_MAX_STAMP_HISTORY = 10_000
COLOR_MODALITY = 'color'
DEPTH_MODALITY = 'depth'
LATEST_JOINT_STATE_POLICY = 'latest_causal_received'
LATEST_JOINT_COMMAND_POLICY = 'latest_causal_received'
UNAVAILABLE_JOINT_STATE_POLICY = 'unavailable'
UNAVAILABLE_JOINT_COMMAND_POLICY = 'unavailable'


def stamp_ns(message: Any) -> int:
    """Return a stamped message's timestamp in nanoseconds."""
    stamp = message.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def canonical_joint_positions(
    message: Any,
    joint_names: Sequence[str] = INDEPENDENT_JOINTS,
) -> Tuple[float, ...]:
    """Return finite positions in the model's canonical 6+1 order."""
    names = tuple(str(name) for name in message.name)
    positions = tuple(float(value) for value in message.position)
    if len(names) != len(positions):
        raise ValueError('joint sample names and positions differ in length')
    if len(set(names)) != len(names):
        raise ValueError('joint sample contains duplicate names')
    missing = tuple(name for name in joint_names if name not in names)
    if missing:
        raise ValueError(f'joint sample is missing required joints: {missing}')
    ordered = tuple(positions[names.index(name)] for name in joint_names)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError('joint sample contains NaN or Inf')
    return ordered


class CausalSampleBuffer:
    """Bound stamped feedback/command history for camera-time lookup."""

    def __init__(self, max_samples: int = 2000) -> None:
        """Retain enough high-rate samples for delayed camera callbacks."""
        self._max_samples = max(1, int(max_samples))
        self._samples = []
        self._keys = []
        self._sequence = 0

    def clear(self) -> None:
        """Discard every retained sample."""
        self._samples.clear()
        self._keys.clear()

    def add(self, message: Any) -> None:
        """Add a stamped message; arrival order need not match stamp order."""
        key = (stamp_ns(message), self._sequence)
        sample = (key[0], key[1], message)
        self._sequence += 1
        if not self._keys or key >= self._keys[-1]:
            self._keys.append(key)
            self._samples.append(sample)
        else:
            index = bisect_right(self._keys, key)
            self._keys.insert(index, key)
            self._samples.insert(index, sample)
        if len(self._samples) > self._max_samples:
            excess = len(self._samples) - self._max_samples
            del self._keys[:excess]
            del self._samples[:excess]

    def latest_at(self, timestamp_ns: int) -> Optional[Any]:
        """Return the latest retained sample not newer than the query."""
        index = bisect_right(
            self._keys,
            (int(timestamp_ns), self._sequence),
        ) - 1
        return None if index < 0 else self._samples[index][2]


@dataclass
class Frame:
    """One synchronized observation across every recorded camera."""

    stamp_ns: int
    images: Dict[str, Any]
    joint_state: Optional[Any] = None
    joint_command: Optional[Any] = None
    object_pose: Optional[Any] = None
    event: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    joint_state_sync_policy: str = UNAVAILABLE_JOINT_STATE_POLICY
    joint_state_stamp_ns: Optional[int] = None
    joint_state_time_offset_ns: Optional[int] = None
    joint_command_sync_policy: str = UNAVAILABLE_JOINT_COMMAND_POLICY
    joint_command_stamp_ns: Optional[int] = None
    joint_command_time_offset_ns: Optional[int] = None

    @property
    def depth_images(self) -> Dict[str, Any]:
        """Return depth images synchronized with this frame's RGB images."""
        return self.extra.get('depth', {})

    def attach_latest_joint_state(self, message: Optional[Any]) -> None:
        """
        Attach the latest received state and expose its time relationship.

        The recorder selects from a causal history, not merely the most recent
        callback.  The signed offset is ``joint stamp - camera stamp`` and is
        therefore always non-positive for an accepted sample.
        """
        self.joint_state = message
        if message is None:
            self.joint_state_sync_policy = UNAVAILABLE_JOINT_STATE_POLICY
            self.joint_state_stamp_ns = None
            self.joint_state_time_offset_ns = None
            return
        self.joint_state_sync_policy = LATEST_JOINT_STATE_POLICY
        self.joint_state_stamp_ns = stamp_ns(message)
        if self.joint_state_stamp_ns > self.stamp_ns:
            raise ValueError('joint state is newer than the camera frame')
        self.joint_state_time_offset_ns = (
            self.joint_state_stamp_ns - self.stamp_ns
        )

    def attach_latest_joint_command(self, message: Optional[Any]) -> None:
        """
        Attach the latest absolute ros2_control position command.

        ``/isaac_joint_commands`` is the post-controller target actually sent
        to Isaac Sim.  This is the action signal required by GR00T; a MoveIt
        waypoint or measured joint state is not an equivalent substitute.
        The offset remains explicit for dataset synchronization audits.
        """
        self.joint_command = message
        if message is None:
            self.joint_command_sync_policy = UNAVAILABLE_JOINT_COMMAND_POLICY
            self.joint_command_stamp_ns = None
            self.joint_command_time_offset_ns = None
            return
        self.joint_command_sync_policy = LATEST_JOINT_COMMAND_POLICY
        self.joint_command_stamp_ns = stamp_ns(message)
        if self.joint_command_stamp_ns > self.stamp_ns:
            raise ValueError('joint command is newer than the camera frame')
        self.joint_command_time_offset_ns = (
            self.joint_command_stamp_ns - self.stamp_ns
        )


class StampPairer:
    """
    Collect streams until every camera has reported one timestamp.

    A tolerance is accepted because a future real-robot recorder will not have
    Isaac Sim's exact-stamp guarantee; in simulation it resolves to an exact
    match.  When depth is requested, every RGB and depth image participates in
    the same barrier; no latest-depth cache is used.
    """

    def __init__(
        self,
        camera_names: Sequence[str],
        tolerance_ns: int = DEFAULT_STAMP_TOLERANCE_NS,
        max_pending: int = 30,
        modalities: Sequence[str] = (COLOR_MODALITY,),
        max_stamp_history: int = DEFAULT_MAX_STAMP_HISTORY,
    ) -> None:
        """Track every required camera/modality stream for one observation."""
        self._camera_names = tuple(camera_names)
        if not self._camera_names:
            raise ValueError('at least one camera is required')
        self._modalities = tuple(modalities)
        if not self._modalities:
            raise ValueError('at least one modality is required')
        if len(set(self._modalities)) != len(self._modalities):
            raise ValueError('modalities must be unique')
        unsupported_modalities = set(self._modalities) - {
            COLOR_MODALITY,
            DEPTH_MODALITY,
        }
        if unsupported_modalities:
            raise ValueError(
                f'unsupported modalities: {sorted(unsupported_modalities)}'
            )
        if COLOR_MODALITY not in self._modalities:
            raise ValueError('color modality is required')
        self._tolerance_ns = int(tolerance_ns)
        if self._tolerance_ns < 0:
            raise ValueError('tolerance_ns must not be negative')
        if DEPTH_MODALITY in self._modalities and self._tolerance_ns != 0:
            raise ValueError(
                'RGB-D recording requires exact timestamps '
                '(tolerance_ns must be 0)'
            )
        self._max_pending = max(1, int(max_pending))
        self._max_stamp_history = max(1, int(max_stamp_history))
        self._required_streams = tuple(
            (camera_name, modality)
            for camera_name in self._camera_names
            for modality in self._modalities
        )
        self._pending: Dict[int, Dict[Tuple[str, str], Any]] = {}
        self._completed_stamps = set()
        self._pending_stamp_index = []
        self._completed_stamp_index = []
        self.reset()

    def reset(self) -> None:
        """Discard unfinished buckets and reset one episode's statistics."""
        self._pending.clear()
        self._completed_stamps.clear()
        self._pending_stamp_index.clear()
        self._completed_stamp_index.clear()
        self.paired_frames = 0
        self.overflow_buckets = 0
        self.out_of_order_frames = 0
        self.max_completed_frame_skew_ns = 0
        self._last_completed_stamp_ns: Optional[int] = None
        self.received_by_camera = {name: 0 for name in self._camera_names}
        self.duplicate_by_camera = {name: 0 for name in self._camera_names}
        self.overflow_by_camera = {name: 0 for name in self._camera_names}
        self.received_by_stream = {
            stream: 0 for stream in self._required_streams
        }
        self.duplicate_by_stream = {
            stream: 0 for stream in self._required_streams
        }
        self.overflow_by_stream = {
            stream: 0 for stream in self._required_streams
        }
        self.non_monotonic_by_stream = {
            stream: 0 for stream in self._required_streams
        }
        self.first_stamp_by_stream = {
            stream: None for stream in self._required_streams
        }
        self.last_stamp_by_stream = {
            stream: None for stream in self._required_streams
        }
        self._stamp_history_by_stream = {
            stream: deque(maxlen=self._max_stamp_history)
            for stream in self._required_streams
        }
        self.stamp_history_truncations_by_stream = {
            stream: 0 for stream in self._required_streams
        }

    @property
    def pending_buckets(self) -> int:
        """Return the number of incomplete timestamp buckets still retained."""
        return len(self._pending)

    @property
    def pending_images(self) -> int:
        """Return images held in incomplete timestamp buckets."""
        return sum(len(images) for images in self._pending.values())

    @property
    def dropped(self) -> int:
        """Compatibility alias for overflowed incomplete timestamp buckets."""
        return self.overflow_buckets

    def _bucket_for(self, timestamp_ns: int) -> int:
        if self._tolerance_ns == 0:
            return timestamp_ns
        existing = self._nearest_indexed_stamp(
            self._pending_stamp_index,
            timestamp_ns,
        )
        if existing is None:
            return timestamp_ns
        return existing

    def _completed_bucket_for(self, timestamp_ns: int) -> Optional[int]:
        """Return an already-emitted bucket matching this timestamp."""
        if self._tolerance_ns == 0:
            return (
                timestamp_ns
                if timestamp_ns in self._completed_stamps else None
            )
        return self._nearest_indexed_stamp(
            self._completed_stamp_index,
            timestamp_ns,
        )

    def _nearest_indexed_stamp(
        self,
        stamps: Sequence[int],
        timestamp_ns: int,
    ) -> Optional[int]:
        """Find the nearest tolerance-matching stamp with two bisect probes."""
        index = bisect_left(stamps, timestamp_ns)
        candidates = []
        if index > 0:
            candidates.append(stamps[index - 1])
        if index < len(stamps):
            candidates.append(stamps[index])
        matching = tuple(
            stamp for stamp in candidates
            if abs(stamp - timestamp_ns) <= self._tolerance_ns
        )
        if not matching:
            return None
        return min(
            matching,
            key=lambda stamp: (abs(stamp - timestamp_ns), stamp),
        )

    @staticmethod
    def _remove_indexed_stamp(stamps: list, timestamp_ns: int) -> None:
        """Remove one known stamp from a sorted index."""
        index = bisect_left(stamps, timestamp_ns)
        if index < len(stamps) and stamps[index] == timestamp_ns:
            stamps.pop(index)

    def stream_rate_hz(
        self,
        camera_name: str,
        modality: str = COLOR_MODALITY,
    ) -> Optional[float]:
        """Estimate one stream's unique-stamp rate over the active episode."""
        stream = (camera_name, modality)
        if stream not in self._stamp_history_by_stream:
            raise KeyError(f'unknown stream: {stream}')
        stamps = sorted(set(self._stamp_history_by_stream[stream]))
        if len(stamps) < 2 or stamps[-1] <= stamps[0]:
            return None
        return (len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0])

    def nearest_camera_skew_summary(
        self,
        modality: str = COLOR_MODALITY,
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Summarize nearest-stamp deltas between every camera pair.

        This is diagnostic rather than a pairing policy: exact pairing remains
        the default, but a fixed render/bridge skew becomes visible instead of
        being reported only as a large number of incomplete buckets.
        """
        if modality not in self._modalities:
            raise KeyError(f'unknown modality: {modality}')
        summaries = {}
        for left_index, left_camera in enumerate(self._camera_names):
            left_stamps = sorted(set(
                self._stamp_history_by_stream[(left_camera, modality)]
            ))
            for right_camera in self._camera_names[left_index + 1:]:
                right_stamps = sorted(set(
                    self._stamp_history_by_stream[(right_camera, modality)]
                ))
                deltas = []
                if right_stamps:
                    for left_stamp in left_stamps:
                        index = bisect_left(right_stamps, left_stamp)
                        candidates = right_stamps[max(0, index - 1):index + 1]
                        if candidates:
                            deltas.append(min(
                                abs(right_stamp - left_stamp)
                                for right_stamp in candidates
                            ))
                if deltas:
                    summaries[(left_camera, right_camera)] = {
                        'samples': float(len(deltas)),
                        'median_ns': float(median(deltas)),
                        'max_ns': float(max(deltas)),
                    }
        return summaries

    def add(
        self,
        camera_name: str,
        message: Any,
        modality: str = COLOR_MODALITY,
    ) -> Optional[Frame]:
        """Add one stream sample; emit only after the full barrier arrives."""
        if camera_name not in self._camera_names:
            raise KeyError(f'unknown camera: {camera_name}')
        if modality not in self._modalities:
            raise KeyError(f'unknown modality: {modality}')
        stream = (camera_name, modality)
        timestamp_ns = stamp_ns(message)
        self.received_by_camera[camera_name] += 1
        self.received_by_stream[stream] += 1
        previous_stamp = self.last_stamp_by_stream[stream]
        if previous_stamp is not None and timestamp_ns <= previous_stamp:
            self.non_monotonic_by_stream[stream] += 1
        first_stamp = self.first_stamp_by_stream[stream]
        self.first_stamp_by_stream[stream] = (
            timestamp_ns
            if first_stamp is None else min(first_stamp, timestamp_ns)
        )
        self.last_stamp_by_stream[stream] = (
            timestamp_ns
            if previous_stamp is None else max(previous_stamp, timestamp_ns)
        )
        history = self._stamp_history_by_stream[stream]
        if len(history) == history.maxlen:
            self.stamp_history_truncations_by_stream[stream] += 1
        history.append(timestamp_ns)

        if self._completed_bucket_for(timestamp_ns) is not None:
            self.duplicate_by_camera[camera_name] += 1
            self.duplicate_by_stream[stream] += 1
            return None

        bucket = self._bucket_for(timestamp_ns)
        samples = self._pending.get(bucket)
        if samples is None:
            samples = {}
            self._pending[bucket] = samples
            if self._tolerance_ns != 0:
                insort_right(self._pending_stamp_index, bucket)
        if stream in samples:
            self.duplicate_by_camera[camera_name] += 1
            self.duplicate_by_stream[stream] += 1
        samples[stream] = message
        if len(samples) == len(self._required_streams):
            del self._pending[bucket]
            if self._tolerance_ns != 0:
                self._remove_indexed_stamp(
                    self._pending_stamp_index,
                    bucket,
                )
            self._completed_stamps.add(bucket)
            if self._tolerance_ns != 0:
                insort_right(self._completed_stamp_index, bucket)
            self.paired_frames += 1
            sample_stamps = tuple(
                stamp_ns(sample) for sample in samples.values()
            )
            frame_skew = max(sample_stamps) - min(sample_stamps)
            self.max_completed_frame_skew_ns = max(
                self.max_completed_frame_skew_ns,
                frame_skew,
            )
            if (
                self._last_completed_stamp_ns is not None
                and bucket < self._last_completed_stamp_ns
            ):
                self.out_of_order_frames += 1
            self._last_completed_stamp_ns = max(
                bucket,
                self._last_completed_stamp_ns
                if self._last_completed_stamp_ns is not None else bucket,
            )
            images = {
                camera_name: samples[(camera_name, COLOR_MODALITY)]
                for camera_name in self._camera_names
            }
            extra = {}
            extra['stream_stamp_offsets_ns'] = {
                f'{camera_name}/{modality}': stamp_ns(
                    samples[(camera_name, modality)]
                ) - bucket
                for camera_name, modality in self._required_streams
            }
            if DEPTH_MODALITY in self._modalities:
                extra['depth'] = {
                    camera_name: samples[(camera_name, DEPTH_MODALITY)]
                    for camera_name in self._camera_names
                }
            return Frame(stamp_ns=bucket, images=images, extra=extra)
        while len(self._pending) > self._max_pending:
            if self._tolerance_ns == 0:
                stale_stamp = min(self._pending)
            else:
                stale_stamp = self._pending_stamp_index.pop(0)
            stale_streams = self._pending.pop(stale_stamp)
            self.overflow_buckets += 1
            for stale_camera, stale_modality in stale_streams:
                self.overflow_by_camera[stale_camera] += 1
                self.overflow_by_stream[(
                    stale_camera,
                    stale_modality,
                )] += 1
        return None
