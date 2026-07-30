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
Topic and payload contract between the episode driver and recorders.

The events are JSON inside ``std_msgs/String`` rather than a custom message so
that this ``ament_python`` package does not have to grow an interface package
just to carry episode boundaries. Recorders parse the payload with
:func:`EpisodeEvent.from_json`.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Dict, Optional


# ROS -> Sim. The payload packs the episode seed and a non-zero request token.
# Isaac Sim feeds the seed to the same deterministic sampler the driver uses,
# then echoes the exact packed value on SCENE_RESET_ACK_TOPIC only after the
# reset pose has actually been written.  The token makes a repeat of the same
# seed a new request instead of something the simulator can discard.
#
# The wire value is the seed PLUS ONE. Isaac Sim's ROS2Subscriber graph node
# reports 0 for a std_msgs/Int32 output before any message has arrived, and it
# is indistinguishable from a real message carrying 0. Without the offset the
# simulator applies seed 0's layout the moment it starts and then ignores a
# genuine seed-0 command as a duplicate -- and 0 is the default start seed.
SCENE_COMMAND_TOPIC = '/vla/scene_command'
SCENE_RESET_ACK_TOPIC = '/vla/scene_reset_ack'
NO_SCENE_COMMAND = 0
_SCENE_SEED_BITS = 16
_SCENE_SEED_MASK = (1 << _SCENE_SEED_BITS) - 1
SCENE_MAX_SEED = _SCENE_SEED_MASK - 1
SCENE_MAX_REQUEST_TOKEN = (1 << (31 - _SCENE_SEED_BITS)) - 1


def scene_command_for_request(seed: int, request_token: int) -> int:
    """
    Pack a reset seed and unique request token into one positive Int32.

    The lower 16 bits carry ``seed + 1``, preserving zero as the OmniGraph
    subscriber's startup/no-message sentinel. The upper 15 bits carry the
    token. Both fields are deliberately bounded so the result is always a
    signed ROS ``std_msgs/Int32`` value.
    """
    value = int(seed)
    if value < 0:
        raise ValueError('episode seeds must not be negative')
    if value > SCENE_MAX_SEED:
        raise ValueError(f'episode seeds must not exceed {SCENE_MAX_SEED}')
    token = int(request_token)
    if token <= 0 or token > SCENE_MAX_REQUEST_TOKEN:
        raise ValueError(
            'scene reset request token must be in '
            f'1..{SCENE_MAX_REQUEST_TOKEN}'
        )
    return (token << _SCENE_SEED_BITS) | (value + 1)


def scene_command_for_seed(seed: int) -> int:
    """
    Return a reset command for ``seed`` using the first request token.

    Kept for command-line callers that issue one reset; episode drivers should
    call :func:`scene_command_for_request` with a fresh token every time.
    """
    return scene_command_for_request(seed, 1)


def seed_from_scene_command(command: int) -> Optional[int]:
    """Return the seed a wire value requests, or None if none was sent."""
    request = scene_request_from_command(command)
    return None if request is None else request[0]


def scene_request_from_command(command: int) -> Optional[tuple[int, int]]:
    """Decode a packed command as ``(seed, request_token)``."""
    value = int(command)
    if value <= NO_SCENE_COMMAND:
        return None
    token = value >> _SCENE_SEED_BITS
    encoded_seed = value & _SCENE_SEED_MASK
    if token <= 0 or encoded_seed == NO_SCENE_COMMAND:
        return None
    return encoded_seed - 1, token


# Driver -> recorders. One message at every phase transition.
EPISODE_EVENT_TOPIC = '/vla/episode_event'
# Driver -> recorders.  A steady-clock timer publishes the currently active
# identity so a recorder can close an episode even when the driver process
# disappears without a terminal event.
EPISODE_HEARTBEAT_TOPIC = '/vla/episode_heartbeat'
# Recorder -> driver.  Transient-local durability makes the most recent writer
# failure visible even if the driver reconnects after the fault occurred.
RECORDER_FAULT_TOPIC = '/vla/recorder_fault'

# Isaac Sim -> the scripted episode driver. ``True`` means the authored
# PhysX FixedJoint has been created after bilateral finger contact; ``False``
# means the object is free. This is deliberately separate from the
# GripperCommand action result: ros2_control's stall detector only observes a
# joint velocity and cannot know whether a simulated object was actually
# grasped.
ATTACHMENT_STATE_TOPIC = '/vla/object_attached'
# Driver -> Isaac Sim.  This explicit intent is independent of the measured
# finger aperture, which may remain partially open around a grasped object.
GRIPPER_INTENT_TOPIC = '/vla/gripper_close_intent'


def tf_frame_from_prim_path(prim_path: str) -> str:
    """
    Return the ROS TF frame Isaac Sim derives from a USD prim path.

    The TF publisher uses the leaf prim name, so keeping a separate hard-coded
    ``Block`` constant would silently break ground-truth recording whenever a
    task selects a differently named object prim.
    """
    path = str(prim_path).strip().rstrip('/')
    frame = path.rsplit('/', 1)[-1]
    if not path.startswith('/') or not frame:
        raise ValueError(
            'object prim_path must be an absolute USD path with a leaf name'
        )
    return frame


class EpisodePhase:
    """Phases an episode moves through, in order."""

    RESET = 'reset'
    OBSERVE = 'observe'
    APPROACH = 'approach'
    GRASP = 'grasp'
    CLOSE_GRIPPER = 'close_gripper'
    LIFT = 'lift'
    TRANSFER = 'transfer'
    PLACE = 'place'
    OPEN_GRIPPER = 'open_gripper'
    RETREAT = 'retreat'
    VERIFY_PLACE = 'verify_place'
    HOME = 'home'

    ORDER = (
        RESET,
        OBSERVE,
        APPROACH,
        GRASP,
        CLOSE_GRIPPER,
        LIFT,
        TRANSFER,
        PLACE,
        OPEN_GRIPPER,
        RETREAT,
        VERIFY_PLACE,
        HOME,
    )


class EpisodeStatus:
    """Lifecycle status carried by every event."""

    STARTED = 'started'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    ABORTED = 'aborted'

    TERMINAL = (SUCCEEDED, FAILED, ABORTED)


def _required_string(name: str, value: Any) -> str:
    """Return a non-empty string used by a boundary protocol payload."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value


def _identity_integer(name: str, value: Any) -> int:
    """Return a non-negative, non-bool integer identity component."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    if value < 0:
        raise ValueError(f'{name} must not be negative')
    return value


def _finite_stamp(value: Any) -> float:
    """Return a finite, non-negative protocol timestamp."""
    if isinstance(value, bool):
        raise ValueError('stamp_sec must be a finite non-negative number')
    try:
        stamp_sec = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            'stamp_sec must be a finite non-negative number'
        ) from error
    if not math.isfinite(stamp_sec) or stamp_sec < 0.0:
        raise ValueError('stamp_sec must be a finite non-negative number')
    return stamp_sec


def _parse_json_object(payload: str, label: str) -> Dict[str, Any]:
    """Parse one JSON object used by the VLA boundary protocol."""
    try:
        raw = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid {label} JSON: {error}') from error
    if not isinstance(raw, dict):
        raise ValueError(f'{label} payload must be a JSON object')
    return raw


@dataclass(frozen=True)
class EpisodeHeartbeat:
    """Lease renewal for one active episode identity."""

    session_id: str
    run_id: str
    episode_id: int
    seed: int
    stamp_sec: float

    def __post_init__(self) -> None:
        """Reject heartbeat payloads that cannot identify an episode."""
        _required_string('session_id', self.session_id)
        _required_string('run_id', self.run_id)
        _identity_integer('episode_id', self.episode_id)
        _identity_integer('seed', self.seed)
        _finite_stamp(self.stamp_sec)

    @property
    def identity(self) -> tuple[str, str, int, int]:
        """Return the identity renewed by this heartbeat."""
        return self.session_id, self.run_id, self.episode_id, self.seed

    def to_json(self) -> str:
        """Serialize this heartbeat for a ROS String payload."""
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, payload: str) -> 'EpisodeHeartbeat':
        """Parse and validate a heartbeat JSON payload."""
        raw = _parse_json_object(payload, 'episode heartbeat')
        missing = sorted(
            {'session_id', 'run_id', 'episode_id', 'seed', 'stamp_sec'}
            - set(raw)
        )
        if missing:
            raise ValueError(f'episode heartbeat is missing fields: {missing}')
        try:
            return cls(
                session_id=raw['session_id'],
                run_id=raw['run_id'],
                episode_id=raw['episode_id'],
                seed=raw['seed'],
                stamp_sec=raw['stamp_sec'],
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'invalid episode heartbeat fields: {error}'
            ) from error


@dataclass(frozen=True)
class RecorderFault:
    """Recorder writer failure scoped to one episode identity."""

    recorder_name: str
    session_id: str
    run_id: str
    episode_id: int
    seed: int
    stamp_sec: float
    detail: str

    def __post_init__(self) -> None:
        """Reject faults that cannot safely stop the matching collection."""
        _required_string('recorder_name', self.recorder_name)
        _required_string('session_id', self.session_id)
        _required_string('run_id', self.run_id)
        _identity_integer('episode_id', self.episode_id)
        _identity_integer('seed', self.seed)
        _finite_stamp(self.stamp_sec)
        _required_string('detail', self.detail)

    @property
    def identity(self) -> tuple[str, str, int, int]:
        """Return the episode identity whose writer failed."""
        return self.session_id, self.run_id, self.episode_id, self.seed

    def to_json(self) -> str:
        """Serialize this fault for a ROS String payload."""
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, payload: str) -> 'RecorderFault':
        """Parse and validate a recorder-fault JSON payload."""
        raw = _parse_json_object(payload, 'recorder fault')
        required = {
            'recorder_name',
            'session_id',
            'run_id',
            'episode_id',
            'seed',
            'stamp_sec',
            'detail',
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f'recorder fault is missing fields: {missing}')
        try:
            return cls(**{name: raw[name] for name in required})
        except (TypeError, ValueError) as error:
            raise ValueError(f'invalid recorder fault fields: {error}') \
                from error


@dataclass(frozen=True)
class EpisodeEvent:
    """One episode lifecycle transition."""

    session_id: str
    run_id: str
    episode_id: int
    seed: int
    phase: str
    status: str
    instruction: str
    stamp_sec: float
    detail: str = ''
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject events that cannot safely delimit a recorded episode."""
        for name, value in (
            ('session_id', self.session_id),
            ('run_id', self.run_id),
            ('instruction', self.instruction),
            ('detail', self.detail),
        ):
            if not isinstance(value, str):
                raise ValueError(f'{name} must be a string')
        if not self.session_id.strip():
            raise ValueError('session_id must not be empty')
        if not self.run_id.strip():
            raise ValueError('run_id must not be empty')
        if not self.instruction.strip():
            raise ValueError('instruction must not be empty')
        for name, value in (
            ('episode_id', self.episode_id),
            ('seed', self.seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an integer')
            if value < 0:
                raise ValueError(f'{name} must not be negative')
        if not isinstance(self.phase, str):
            raise ValueError('phase must be a string')
        if self.phase not in EpisodePhase.ORDER:
            raise ValueError(f'unknown episode phase: {self.phase!r}')
        valid_statuses = {
            EpisodeStatus.STARTED,
            EpisodeStatus.RUNNING,
            *EpisodeStatus.TERMINAL,
        }
        if not isinstance(self.status, str):
            raise ValueError('status must be a string')
        if self.status not in valid_statuses:
            raise ValueError(f'unknown episode status: {self.status!r}')
        if isinstance(self.stamp_sec, bool):
            raise ValueError('stamp_sec must be a finite non-negative number')
        try:
            stamp_sec = float(self.stamp_sec)
        except (TypeError, ValueError) as error:
            raise ValueError(
                'stamp_sec must be a finite non-negative number'
            ) from error
        if not math.isfinite(stamp_sec) or stamp_sec < 0.0:
            raise ValueError('stamp_sec must be a finite non-negative number')
        if not isinstance(self.extra, dict):
            raise ValueError('extra must be a JSON object')
        try:
            json.dumps(self.extra, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                'extra must contain finite JSON-serializable values'
            ) from error

    @property
    def identity(self) -> tuple[str, str, int, int]:
        """
        Return the process/run/episode/seed identity used by recorders.

        The deterministic scene layout is selected by ``seed``. Treating an
        event with the same numeric episode ID but a different seed as part of
        the active episode would merge observations with incompatible ground
        truth, so the seed is part of the boundary contract rather than mere
        metadata.
        """
        return self.session_id, self.run_id, self.episode_id, self.seed

    def to_json(self) -> str:
        """Serialize this event for the ROS String payload."""
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @property
    def is_terminal(self) -> bool:
        """Return whether this event ends the episode."""
        return self.status in EpisodeStatus.TERMINAL

    @classmethod
    def from_json(cls, payload: str) -> 'EpisodeEvent':
        """Parse an event from a ROS String payload."""
        try:
            raw = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f'invalid episode event JSON: {error}') from error
        if not isinstance(raw, dict):
            raise ValueError('episode event payload must be a JSON object')
        missing = sorted(
            {
                'session_id',
                'run_id',
                'episode_id',
                'seed',
                'phase',
                'status',
                'instruction',
                'stamp_sec',
            } - set(raw)
        )
        if missing:
            raise ValueError(f'episode event is missing fields: {missing}')
        extra = raw.get('extra', {})
        if extra is None:
            extra = {}
        if not isinstance(extra, dict):
            raise ValueError('extra must be a JSON object')
        try:
            return cls(
                session_id=raw['session_id'],
                run_id=raw['run_id'],
                episode_id=raw['episode_id'],
                seed=raw['seed'],
                phase=raw['phase'],
                status=raw['status'],
                instruction=raw['instruction'],
                stamp_sec=raw['stamp_sec'],
                detail=raw.get('detail', ''),
                extra=dict(extra),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'invalid episode event fields: {error}'
            ) from error
