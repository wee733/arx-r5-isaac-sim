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

"""Dependency-free stability gates for timestamped destination-tag poses."""

from collections import deque
from dataclasses import dataclass
from math import dist, isfinite
from statistics import median
from typing import Optional


@dataclass
class TargetSequenceGate:
    """Track one uninterrupted target-visible sequence and its newest pose."""

    generation: int = 0
    visible_frames: int = 0
    pose_generation: Optional[int] = None
    pose_stamp_ns: Optional[int] = None

    def observe_target(self) -> int:
        """Record one visible frame and return its sequence generation."""
        if self.visible_frames == 0:
            self.generation += 1
            self.pose_generation = None
            self.pose_stamp_ns = None
        self.visible_frames += 1
        return self.generation

    def miss_target(self) -> None:
        """Invalidate the sequence and every pose derived from it."""
        if self.visible_frames > 0 or self.pose_generation is not None:
            self.generation += 1
        self.visible_frames = 0
        self.pose_generation = None
        self.pose_stamp_ns = None

    def accept_pose(self, generation: int, stamp_ns: int) -> bool:
        """Accept an asynchronous result only for the active sequence."""
        if self.visible_frames == 0 or generation != self.generation:
            return False
        self.pose_generation = generation
        self.pose_stamp_ns = int(stamp_ns)
        return True

    def ready(
        self,
        *,
        required_frames: int,
        now_ns: int,
        ttl_ns: int,
    ) -> bool:
        """Return whether the active sequence owns a stable, fresh pose."""
        if (
            self.visible_frames < required_frames or
            self.pose_generation != self.generation or
            self.pose_stamp_ns is None
        ):
            return False
        age_ns = int(now_ns) - self.pose_stamp_ns
        return 0 <= age_ns <= int(ttl_ns)


class TranslationMedianGate:
    """Estimate a translation only from a compact, low-spread sample window."""

    def __init__(self, *, window_size: int, max_spread_m: float) -> None:
        """Validate the window policy and start without an active sequence."""
        if int(window_size) != window_size or int(window_size) < 1:
            raise ValueError(
                'window_size must be a positive integer'
            )
        if not isfinite(max_spread_m) or max_spread_m <= 0.0:
            raise ValueError('max_spread_m must be finite and positive')
        self._window_size = int(window_size)
        self._max_spread_m = float(max_spread_m)
        self._generation = None
        self._samples = deque(maxlen=self._window_size)
        self._last_stamp_ns = None
        self._spread_m = None

    @property
    def sample_count(self) -> int:
        """Return the number of samples in the active rolling window."""
        return len(self._samples)

    @property
    def spread_m(self) -> Optional[float]:
        """Return the latest full-window diameter, if one was computed."""
        return self._spread_m

    def reset(self) -> None:
        """Discard every sample so visibility gaps cannot share a median."""
        self._generation = None
        self._samples.clear()
        self._last_stamp_ns = None
        self._spread_m = None

    def observe(
        self,
        *,
        generation: int,
        stamp_ns: int,
        translation,
    ) -> Optional[tuple]:
        """
        Return the component-wise median when the current window is stable.

        Samples are expected to have already been transformed at their exact
        detection timestamps. A timestamp rewind starts a new window, while a
        duplicate timestamp replaces the most recent value instead of gaining
        extra voting weight.
        """
        sample = tuple(float(value) for value in translation)
        if len(sample) != 3 or not all(isfinite(value) for value in sample):
            raise ValueError('translation must contain three finite values')
        stamp_ns = int(stamp_ns)
        if self._generation != int(generation):
            self.reset()
            self._generation = int(generation)
        elif self._last_stamp_ns is not None:
            if stamp_ns < self._last_stamp_ns:
                self.reset()
                self._generation = int(generation)
            elif stamp_ns == self._last_stamp_ns:
                self._samples[-1] = sample
                return self._evaluate()

        self._samples.append(sample)
        self._last_stamp_ns = stamp_ns
        return self._evaluate()

    def _evaluate(self) -> Optional[tuple]:
        """Evaluate one complete rolling window without mutating it."""
        if len(self._samples) < self._window_size:
            self._spread_m = None
            return None
        samples = tuple(self._samples)
        self._spread_m = max(
            (
                dist(first, second)
                for index, first in enumerate(samples)
                for second in samples[index + 1:]
            ),
            default=0.0,
        )
        if self._spread_m > self._max_spread_m:
            return None
        return tuple(
            median(sample[axis] for sample in samples)
            for axis in range(3)
        )
