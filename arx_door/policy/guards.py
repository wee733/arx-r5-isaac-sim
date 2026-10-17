# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Deployable gripper gating and stage prompting for closed-loop door policies.

Both use only signals available on the robot: the policy's own gripper
command, the measured finger width and the arm joints. Neither reads door
joints, contact sensors or any simulator ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dataset.stages import KNOB_HOLD_OPEN_STAGES, STAGE_NAMES

GRIPPER_OPEN_M = 0.028
GRIPPER_CLOSED_M = 0.0


@dataclass
class GripperGate:
    """Binarize the continuous gripper head with hysteresis and debouncing.

    Demonstrations only command fully open or fully closed, so an intermediate
    prediction is regression noise. Executing it closed the fingers mid-air in
    every evaluated rollout of the first checkpoint.
    """

    close_below_m: float = 0.010
    open_above_m: float = 0.020
    close_steps: int = 3
    open_steps: int = 15
    closed: bool = False
    _count: int = 0

    def reset(self) -> None:
        self.closed, self._count = False, 0

    def force_open(self) -> None:
        self.closed, self._count = False, 0

    def __call__(self, command_m: float) -> float:
        wants_change = (command_m > self.open_above_m) if self.closed else (command_m < self.close_below_m)
        self._count = self._count + 1 if wants_change else 0
        if self._count >= (self.open_steps if self.closed else self.close_steps):
            self.closed, self._count = not self.closed, 0
        return GRIPPER_CLOSED_M if self.closed else GRIPPER_OPEN_M


@dataclass
class RuleStagePrompter:
    """Advance approach -> grasp -> unlock -> open from deployable signals.

    * approach -> grasp: the policy starts closing (raw command below
      ``closing_below_m`` for ``closing_steps``) or the gate has closed.
    * grasp -> unlock: the gate is closed and the measured width has settled
      while still blocked by the knob (not closed on air) for ``settle_steps``.
    * unlock -> open: the wrist (joint6) has turned ``unlock_wrist_rad`` since
      the grasp started; the demonstrations turn the knob 45 degrees by joint6.
    * A width below ``missed_width_m`` during grasp/unlock is a missed grasp:
      return to approach and ask the caller to reopen. Grasp is re-armed only
      after the policy commands an open gripper again.
    """

    closing_below_m: float = 0.0275
    closing_steps: int = 2
    settle_steps: int = 10
    settle_tolerance_m: float = 2e-5
    blocked_width_m: float = 0.018
    missed_width_m: float = 0.012
    unlock_wrist_rad: float = 0.79
    stage: int = 0
    _widths: list = field(default_factory=list)
    _wrist_at_grasp: float | None = None
    _closing: int = 0
    _armed: bool = True

    def reset(self) -> None:
        self.stage, self._widths, self._wrist_at_grasp, self._closing = 0, [], None, 0
        self._armed = True

    @property
    def name(self) -> str:
        return STAGE_NAMES[self.stage]

    @property
    def instruction(self) -> str:
        return KNOB_HOLD_OPEN_STAGES[self.stage].instruction

    def update(self, state, command_m: float, gripper_closed: bool) -> tuple[int, bool]:
        """Consume one measured 7D state and raw gripper command; return (stage, missed_grasp)."""
        state = np.asarray(state, dtype=float)
        width = float(state[6])
        self._closing = self._closing + 1 if command_m < self.closing_below_m else 0
        self._armed |= command_m >= self.closing_below_m
        self._widths = (self._widths + [width])[-self.settle_steps:]
        if self.stage in (1, 2) and width < self.missed_width_m and len(self._widths) == self.settle_steps \
                and max(self._widths) < self.missed_width_m:
            self.reset()
            self._armed = False
            return self.stage, True
        if self.stage == 0 and self._armed and (gripper_closed or self._closing >= self.closing_steps):
            self.stage, self._widths, self._wrist_at_grasp = 1, [width], float(state[5])
        elif self.stage == 1 and gripper_closed and len(self._widths) == self.settle_steps:
            if (np.ptp(self._widths) <= self.settle_tolerance_m and width >= self.blocked_width_m):
                self.stage = 2
        elif self.stage == 2 and abs(float(state[5]) - self._wrist_at_grasp) >= self.unlock_wrist_rad:
            self.stage = 3
        return self.stage, False
