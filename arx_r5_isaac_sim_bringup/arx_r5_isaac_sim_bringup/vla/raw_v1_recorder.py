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

"""ROS adapter that persists RecorderBase frames as ARX raw-v1."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from arx_r5_isaac_sim_bringup.vla.episode_events import (
    EpisodeEvent,
    EpisodeStatus,
)
from arx_r5_isaac_sim_bringup.vla.frame_sync import (
    canonical_joint_positions,
    Frame,
    stamp_ns,
)
from arx_r5_isaac_sim_bringup.vla.raw_v1_writer import ArxRawRecorder
from arx_r5_isaac_sim_bringup.vla.recorder_base import (
    RecorderBase,
    run_recorder,
)

import numpy as np


_IMAGE_CHANNELS = {
    'rgb8': 3,
    'bgr8': 3,
    'rgba8': 4,
    'bgra8': 4,
}


def image_to_rgb(message: Any) -> np.ndarray:
    """Decode a ROS 8-bit color image into contiguous HWC RGB pixels."""
    encoding = str(message.encoding).strip().lower()
    if encoding not in _IMAGE_CHANNELS:
        raise ValueError(
            f'unsupported RGB image encoding {message.encoding!r}; '
            f'expected one of {tuple(_IMAGE_CHANNELS)}'
        )
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    channels = _IMAGE_CHANNELS[encoding]
    if width <= 0 or height <= 0:
        raise ValueError('image width and height must be positive')
    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError(
            f'image step {step} is smaller than {row_bytes} bytes of pixels'
        )
    try:
        raw = np.frombuffer(message.data, dtype=np.uint8)
    except TypeError:
        raw = np.asarray(message.data, dtype=np.uint8).reshape(-1)
    required_bytes = step * height
    if raw.size < required_bytes:
        raise ValueError(
            f'image data has {raw.size} bytes; expected at least '
            f'{required_bytes}'
        )
    rows = raw[:required_bytes].reshape(height, step)
    packed = np.ascontiguousarray(rows[:, :row_bytes])
    pixels = packed.reshape(height, width, channels)
    if encoding == 'rgb8':
        return pixels
    if encoding == 'bgr8':
        return np.ascontiguousarray(pixels[..., ::-1])
    if encoding == 'rgba8':
        return np.ascontiguousarray(pixels[..., :3])
    return np.ascontiguousarray(pixels[..., [2, 1, 0]])


class RawV1Recorder(RecorderBase):
    """Persist successful synchronized episodes in the raw-v1 contract."""

    def __init__(self) -> None:
        """Load recorder parameters before accepting the first episode."""
        super().__init__('vla_recorder')
        if self._record_depth:
            raise ValueError(
                'raw-v1 stores RGB only; set record_depth:=False or use a '
                'different writer'
            )
        output_dir = self.declare_parameter(
            'output_dir',
            '~/arx_raw',
        ).get_parameter_value().string_value
        fps = self.declare_parameter(
            'fps',
            30,
        ).get_parameter_value().integer_value
        encoder = self.declare_parameter(
            'encoder',
            'auto',
        ).get_parameter_value().string_value
        ffmpeg = self.declare_parameter(
            'ffmpeg',
            'ffmpeg',
        ).get_parameter_value().string_value
        front_camera = self.declare_parameter(
            'front_camera',
            'zedx',
        ).get_parameter_value().string_value.strip()
        wrist_camera = self.declare_parameter(
            'wrist_camera',
            'd455',
        ).get_parameter_value().string_value.strip()
        if not front_camera or not wrist_camera:
            raise ValueError('front_camera and wrist_camera are required')
        if front_camera == wrist_camera:
            raise ValueError('front_camera and wrist_camera must differ')
        missing = sorted(
            {front_camera, wrist_camera} - set(self._cameras)
        )
        if missing:
            raise ValueError(
                f'raw-v1 camera roles {missing} are absent from scene cameras '
                f'{tuple(self._cameras)}'
            )
        self._front_camera = front_camera
        self._wrist_camera = wrist_camera
        self._raw = ArxRawRecorder(
            output_dir,
            fps=fps,
            action_source='isaac_joint_commands',
            ffmpeg=ffmpeg,
            encoder=encoder,
        )
        self._raw_episode_index: Optional[int] = None
        self.get_logger().info(
            f'raw-v1 recorder writing to {self._raw.root}; '
            f'front={self._front_camera}, wrist={self._wrist_camera}, '
            f'fps={self._raw.fps}, encoder={self._raw.encoder}'
        )

    @staticmethod
    def _stamp_sec(message: Any) -> float:
        """Return one ROS image stamp as finite seconds."""
        value = stamp_ns(message) / 1e9
        if not math.isfinite(value) or value < 0.0:
            raise ValueError('image timestamp must be finite and non-negative')
        return value

    @staticmethod
    def _positions(message: Any, label: str) -> np.ndarray:
        """Select the canonical 6+1 vector by joint name, never by index."""
        if message is None:
            raise ValueError(f'{label} is unavailable at the camera timestamp')
        try:
            values = canonical_joint_positions(message)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f'invalid {label}: {error}') from error
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (7,) or not np.isfinite(result).all():
            raise ValueError(f'{label} did not produce a finite 7-D vector')
        return result

    def on_episode_start(self, event: EpisodeEvent) -> None:
        """Open an atomic temporary raw-v1 directory for this identity."""
        metadata = {
            'session_id': event.session_id,
            'run_id': event.run_id,
            'source_episode_id': event.episode_id,
            'seed': event.seed,
            'start_stamp_sec': float(event.stamp_sec),
            'start_phase': event.phase,
            'event_extra': dict(event.extra),
        }
        self._raw_episode_index = self._raw.start_episode(
            event.instruction,
            metadata=metadata,
        )
        episode_path = (
            self._raw.root
            / 'episodes'
            / f'episode_{self._raw_episode_index:06d}'
        )
        self.get_logger().info(
            f'raw-v1 episode {event.episode_id} -> {episode_path}'
        )

    def on_frame(self, frame: Frame) -> None:
        """Convert one synchronized ROS frame and append it to raw-v1."""
        front_message = frame.images.get(self._front_camera)
        wrist_message = frame.images.get(self._wrist_camera)
        if front_message is None or wrist_message is None:
            raise ValueError(
                'frame is missing one of the configured RGB cameras: '
                f'{self._front_camera}, {self._wrist_camera}'
            )
        state = self._positions(frame.joint_state, 'joint state')
        action = self._positions(frame.joint_command, 'joint command')
        self._raw.append(
            front_rgb=image_to_rgb(front_message),
            wrist_rgb=image_to_rgb(wrist_message),
            state=state,
            action=action,
            timestamp=frame.stamp_ns / 1e9,
            front_timestamp=self._stamp_sec(front_message),
            wrist_timestamp=self._stamp_sec(wrist_message),
        )

    def on_episode_end(self, event: EpisodeEvent, frame_count: int) -> None:
        """Commit only successful episodes and discard failed trajectories."""
        if not self._raw.active:
            if event.status == EpisodeStatus.SUCCEEDED:
                raise RuntimeError(
                    'raw-v1 writer has no active episode at successful end'
                )
            return
        try:
            if event.status == EpisodeStatus.SUCCEEDED:
                if self._raw.active_frame_count != frame_count:
                    raise RuntimeError(
                        'raw-v1 frame count disagrees with RecorderBase: '
                        f'{self._raw.active_frame_count} != {frame_count}'
                    )
                index = self._raw.end_episode(notes=event.detail)
                self.get_logger().info(
                    f'committed raw-v1 episode {event.episode_id} '
                    f'(index={index}, frames={frame_count})'
                )
            else:
                self._raw.abort_episode()
                self.get_logger().warning(
                    f'discarded {event.status} raw-v1 episode '
                    f'{event.episode_id} after {frame_count} frames'
                )
        finally:
            self._raw_episode_index = None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Spin the persistent raw-v1 recorder."""
    return run_recorder(RawV1Recorder, argv)


if __name__ == '__main__':
    raise SystemExit(main())
