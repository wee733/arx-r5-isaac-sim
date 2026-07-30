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

"""Tests for the authored workcell's physical grasp gate."""

from pathlib import Path

from arx_r5_isaac_sim_bringup.simulation import (
    _build_argument_parser,
    AuthoredObjectAttachmentController,
    BilateralFingerContactTracker,
    GRIPPER_CLOSE_COMMAND_THRESHOLD,
    GRIPPER_OPEN_COMMAND_THRESHOLD,
)


OBJECT = '/World/Workspace/TaggedCube'
LEFT = '/R5a/link7'
RIGHT = '/R5a/link8'


class _Robot:
    """Minimal articulation state used by the controller gate tests."""

    dof_names = ('joint7', 'joint8')

    def __init__(self, aperture, mimic_aperture=None):
        self._aperture = aperture
        self._mimic_aperture = (
            aperture if mimic_aperture is None else mimic_aperture
        )

    def get_joint_positions(self):
        """Return both simulated finger joint positions."""
        return (self._aperture, self._mimic_aperture)


class _Commands:
    """Stand-in for the ROS joint-command reader."""

    def __init__(self, position):
        self.position = position

    def commanded_position(self):
        """Return the gripper position last commanded over ROS."""
        return self.position


class _Intent:
    """Stand-in for the VLA Bool grasp-intent reader."""

    def __init__(self, close_requested):
        self.value = close_requested

    def close_requested(self):
        """Return the latest close/hold state."""
        return self.value


class _Stage:
    """Minimal USD stage recording runtime joint removal."""

    def __init__(self):
        self.removed_prims = []

    def RemovePrim(self, path):
        """Record one removed prim path."""
        self.removed_prims.append(path)


def _update(
    tracker,
    actor0,
    collider0,
    actor1,
    collider1,
    active=True,
):
    tracker.update(actor0, collider0, actor1, collider1, active)


def test_attachment_requires_simultaneous_contact_from_both_fingers():
    """One finger, aperture, or distance alone must never fake a grasp."""
    tracker = BilateralFingerContactTracker(OBJECT, (LEFT, RIGHT))

    _update(tracker, OBJECT, f'{OBJECT}/Body', LEFT, f'{LEFT}/collider')
    assert not tracker.has_bilateral_contact

    _update(tracker, OBJECT, f'{OBJECT}/Body', '/World/Table', '/World/Table')
    assert not tracker.has_bilateral_contact

    _update(tracker, RIGHT, f'{RIGHT}/collider', OBJECT, f'{OBJECT}/Body')
    assert tracker.has_bilateral_contact


def test_lost_contact_and_release_clear_the_bilateral_gate():
    """A LOST event or gripper release invalidates the physical grasp."""
    tracker = BilateralFingerContactTracker(OBJECT, (LEFT, RIGHT))
    left_collider = f'{LEFT}/collider'
    right_collider = f'{RIGHT}/collider'
    object_collider = f'{OBJECT}/Body'
    _update(tracker, OBJECT, object_collider, LEFT, left_collider)
    _update(tracker, OBJECT, object_collider, RIGHT, right_collider)
    assert tracker.has_bilateral_contact

    _update(
        tracker,
        LEFT,
        left_collider,
        OBJECT,
        object_collider,
        active=False,
    )
    assert not tracker.has_bilateral_contact

    _update(tracker, OBJECT, object_collider, LEFT, left_collider)
    assert tracker.has_bilateral_contact
    tracker.clear()
    assert not tracker.has_bilateral_contact


def _controller_for_gate(
    commanded,
    measured_aperture=None,
    mimic_aperture=None,
):
    """
    Build a controller whose only live inputs are contact and command.

    ``commanded`` is what ros2_control asked for; ``measured_aperture`` is where
    the fingers actually stopped. They differ whenever the gripper closes on an
    object, which is exactly the case the gate has to handle.
    """
    controller = object.__new__(AuthoredObjectAttachmentController)
    if measured_aperture is None:
        measured_aperture = commanded
    controller._robot = _Robot(measured_aperture, mimic_aperture)
    controller._commands = _Commands(commanded)
    controller._intent = None
    controller._maximum_distance = 0.10
    controller._close_command_threshold = GRIPPER_CLOSE_COMMAND_THRESHOLD
    controller._open_command_threshold = GRIPPER_OPEN_COMMAND_THRESHOLD
    controller._attached = False
    controller._required_contact_steps = 3
    controller._bilateral_contact_steps = 0
    controller._contacts = BilateralFingerContactTracker(
        OBJECT,
        (LEFT, RIGHT),
    )
    controller._distance_to_grasp_frame = lambda: 0.05
    return controller


def test_controller_does_not_attach_from_command_and_distance_alone():
    """A close command without both PhysX contacts must remain unattached."""
    controller = _controller_for_gate(0.0)
    created = []
    controller._create_joint = lambda: created.append(True)

    _update(
        controller._contacts,
        OBJECT,
        f'{OBJECT}/Body',
        LEFT,
        f'{LEFT}/collider',
    )
    controller.update()
    assert not created

    _update(
        controller._contacts,
        OBJECT,
        f'{OBJECT}/Body',
        RIGHT,
        f'{RIGHT}/collider',
    )
    for _ in range(2):
        controller.update()
        assert not created
    controller.update()
    assert created == [True]


def test_wide_block_stalls_the_fingers_but_still_attaches():
    """
    The gate must key off the command, not where the fingers stopped.

    joint7 travels 0..0.044 and both fingers move symmetrically, so the
    fingertip aperture is twice the joint value: an 0.08 m block stalls each
    finger near 0.040. Any threshold on the measured position that admits that
    grasp would also admit a fully open gripper.
    """
    controller = _controller_for_gate(0.0, measured_aperture=0.040)
    created = []
    controller._create_joint = lambda: created.append(True)
    for finger in (LEFT, RIGHT):
        _update(
            controller._contacts,
            OBJECT,
            f'{OBJECT}/Body',
            finger,
            f'{finger}/collider',
        )

    for _ in range(3):
        controller.update()

    assert created == [True]
    assert 0.040 > GRIPPER_OPEN_COMMAND_THRESHOLD


def test_asymmetric_physical_close_still_attaches():
    """An off-centre block may stop the two real fingers at different DOFs."""
    controller = _controller_for_gate(
        0.0,
        measured_aperture=0.0301,
        mimic_aperture=0.0220,
    )
    created = []
    controller._create_joint = lambda: created.append(True)
    _update(
        controller._contacts,
        OBJECT,
        f'{OBJECT}/Body',
        LEFT,
        f'{LEFT}/collider',
    )
    _update(
        controller._contacts,
        OBJECT,
        f'{OBJECT}/Body',
        RIGHT,
        f'{RIGHT}/collider',
    )

    controller.update()
    controller.update()
    controller.update()

    assert created == [True]


def test_no_command_yet_never_attaches():
    """Before any command arrives, contact alone must not glue the object on."""
    controller = _controller_for_gate(0.0)
    controller._commands = _Commands(None)
    created = []
    controller._create_joint = lambda: created.append(True)
    for finger in (LEFT, RIGHT):
        _update(
            controller._contacts,
            OBJECT,
            f'{OBJECT}/Body',
            finger,
            f'{finger}/collider',
        )

    for _ in range(5):
        controller.update()

    assert not created
    assert controller._bilateral_contact_steps == 0


def test_open_command_with_contact_does_not_attach():
    """Brushing the block while open must never create the grasp joint."""
    controller = _controller_for_gate(0.044)
    created = []
    controller._create_joint = lambda: created.append(True)
    for finger in (LEFT, RIGHT):
        _update(
            controller._contacts,
            OBJECT,
            f'{OBJECT}/Body',
            finger,
            f'{finger}/collider',
        )

    for _ in range(5):
        controller.update()

    assert not created


def test_distant_object_does_not_attach():
    """Contact far from grasp_frame is a collision, not a grasp."""
    controller = _controller_for_gate(0.0)
    controller._distance_to_grasp_frame = lambda: 0.5
    created = []
    controller._create_joint = lambda: created.append(True)
    for finger in (LEFT, RIGHT):
        _update(
            controller._contacts,
            OBJECT,
            f'{OBJECT}/Body',
            finger,
            f'{finger}/collider',
        )

    for _ in range(5):
        controller.update()

    assert not created


def test_transient_bilateral_contact_resets_consecutive_evidence():
    """A contact loss must restart the persistence gate from step zero."""
    controller = _controller_for_gate(0.0)
    created = []
    controller._create_joint = lambda: created.append(True)
    object_collider = f'{OBJECT}/Body'
    left_collider = f'{LEFT}/collider'
    right_collider = f'{RIGHT}/collider'

    _update(
        controller._contacts,
        OBJECT,
        object_collider,
        LEFT,
        left_collider,
    )
    _update(
        controller._contacts,
        OBJECT,
        object_collider,
        RIGHT,
        right_collider,
    )
    controller.update()
    controller.update()
    assert controller._bilateral_contact_steps == 2
    assert not created

    _update(
        controller._contacts,
        RIGHT,
        right_collider,
        OBJECT,
        object_collider,
        active=False,
    )
    controller.update()
    assert controller._bilateral_contact_steps == 0

    _update(
        controller._contacts,
        RIGHT,
        right_collider,
        OBJECT,
        object_collider,
    )
    controller.update()
    controller.update()
    assert not created
    controller.update()
    assert created == [True]


def test_controller_releases_when_an_open_command_arrives():
    """An open command must remove the physical fixed joint."""
    controller = _controller_for_gate(0.044, measured_aperture=0.040)
    controller._stage = _Stage()
    controller._attached = True
    controller._bilateral_contact_steps = 2
    _update(
        controller._contacts,
        OBJECT,
        f'{OBJECT}/Body',
        LEFT,
        f'{LEFT}/collider',
    )

    controller.update()
    assert controller._stage.removed_prims == [
        AuthoredObjectAttachmentController._JOINT_PATH,
    ]
    assert not controller.attached
    assert controller._bilateral_contact_steps == 0
    assert not controller._contacts.has_bilateral_contact


def test_stalled_fingers_do_not_trigger_a_spurious_release():
    """
    Holding a wide block must not read as an open gripper.

    This is the regression the command-driven gate exists for: with the old
    aperture thresholds an 0.08 m block held at 0.040 was above the open
    threshold, so the object was released the instant it was grasped.
    """
    controller = _controller_for_gate(0.0, measured_aperture=0.040)
    controller._stage = _Stage()
    controller._attached = True

    for _ in range(5):
        controller.update()

    assert controller._stage.removed_prims == []
    assert controller.attached


def test_vla_close_intent_survives_controller_hold_position():
    """Cancel/hold after contact must not be reinterpreted as an open."""
    controller = _controller_for_gate(0.040, measured_aperture=0.040)
    controller._stage = _Stage()
    controller._intent = _Intent(True)
    controller._attached = True

    for _ in range(5):
        controller.update()

    assert controller._stage.removed_prims == []
    assert controller.attached


def test_vla_close_intent_can_create_the_first_attachment():
    """Explicit close intent must reach FixedJoint creation without logging."""
    controller = _controller_for_gate(0.040, measured_aperture=0.040)
    controller._intent = _Intent(True)
    created = []
    controller._create_joint = lambda: created.append(True)
    for finger in (LEFT, RIGHT):
        _update(
            controller._contacts,
            OBJECT,
            f'{OBJECT}/Body',
            finger,
            f'{finger}/collider',
        )

    for _ in range(3):
        controller.update()

    assert created == [True]


def test_vla_open_intent_releases_even_if_joint_command_stays_closed():
    """The explicit release state wins over a stale zero joint target."""
    controller = _controller_for_gate(0.0, measured_aperture=0.025)
    controller._stage = _Stage()
    controller._intent = _Intent(False)
    controller._attached = True

    controller.update()

    assert controller._stage.removed_prims == [
        AuthoredObjectAttachmentController._JOINT_PATH,
    ]
    assert not controller.attached


def test_missing_vla_intent_never_attaches_on_contact():
    """An unevaluated Bool subscriber must fail safe during startup."""
    controller = _controller_for_gate(0.0)
    controller._intent = _Intent(None)
    created = []
    controller._create_joint = lambda: created.append(True)
    for finger in (LEFT, RIGHT):
        _update(
            controller._contacts,
            OBJECT,
            f'{OBJECT}/Body',
            finger,
            f'{finger}/collider',
        )

    for _ in range(5):
        controller.update()

    assert created == []
    assert controller._bilateral_contact_steps == 0
    assert not controller._contacts.has_bilateral_contact


def test_release_drops_the_object_without_an_open_command():
    """An episode reset must be able to clear a grasp unconditionally."""
    controller = _controller_for_gate(0.0)
    controller._stage = _Stage()
    controller._attached = True

    controller.release()

    assert controller._stage.removed_prims == [
        AuthoredObjectAttachmentController._JOINT_PATH,
    ]
    assert not controller.attached


def test_release_is_safe_when_nothing_is_attached():
    """Resetting between episodes must not remove a joint that never existed."""
    controller = _controller_for_gate(0.0)
    controller._stage = _Stage()
    controller._bilateral_contact_steps = 2

    controller.release()

    assert controller._stage.removed_prims == []
    assert controller._bilateral_contact_steps == 0


def test_physical_grasp_defaults_limit_force_and_filter_contact():
    """CLI defaults must retain the conservative physical grasp contract."""
    args = _build_argument_parser().parse_args([])

    assert args.gripper_drive_max_force == 8.0
    assert args.grasp_contact_steps == 3


def test_joint_frames_are_derived_from_one_world_anchor():
    """Both local frames must reconstruct one pose without PhysX snapping."""
    source = (
        Path(__file__).resolve().parents[1]
        / 'arx_r5_isaac_sim_bringup'
        / 'simulation.py'
    ).read_text(encoding='utf-8')
    create_start = source.index('    def _create_joint(self) -> None:')
    remove_start = source.index('    def _remove_joint(self) -> None:')
    create_source = source[create_start:remove_start]

    assert 'ComputeRelativeTransform' not in create_source
    assert 'anchor_world * body_world.GetInverse()' in create_source
    assert 'anchor_world * object_world.GetInverse()' in create_source
    assert 'local0 * body_world' in create_source
    assert 'local1 * object_world' in create_source


def test_contact_report_api_and_bilateral_gate_are_both_required():
    """The runtime must consume real PhysX contact events before attaching."""
    source = (
        Path(__file__).resolve().parents[1]
        / 'arx_r5_isaac_sim_bringup'
        / 'simulation.py'
    ).read_text(encoding='utf-8')

    assert 'PhysxContactReportAPI.Apply(object_prim)' in source
    assert 'subscribe_contact_report_events(' in source
    assert 'ContactEventType.CONTACT_FOUND' in source
    assert 'ContactEventType.CONTACT_PERSIST' in source
    assert 'ContactEventType.CONTACT_LOST' in source
    assert 'not self._contacts.has_bilateral_contact' in source
    assert 'drive.CreateMaxForceAttr(gripper_max_force)' in source
    assert 'for joint_name in ALL_JOINTS' in source
    assert 'GRIPPER_COMMAND_JOINT, GRIPPER_MIMIC_JOINT' in source
    assert 'joint8 mirrors joint7 commands' in source
    assert 'import_config.parse_mimic = False' in source
    assert '_disable_authored_gripper_mimic(stage)' in source
    assert 'joint_prim.RemoveAPI(' in source
    assert 'PhysxSchema.PhysxMimicJointAPI' in source
    assert 'joint8 cannot combine a PhysX mimic constraint' in source
    assert 'CreateMaxJointVelocityAttr(GRIPPER_MAX_VELOCITY)' in source
    assert 'drive.CreateTypeAttr(UsdPhysics.Tokens.acceleration)' in source
    # The legacy AprilTag gate keys off the ROS command rather than the
    # measured aperture. VLA collection uses an even stronger explicit Bool
    # intent, because action cancellation may replace the zero command with a
    # measured hold position.
    assert 'self._commands.commanded_position()' in source
    assert 'commanded > self._close_command_threshold' in source
    assert 'commanded >= self._open_command_threshold' in source
    assert 'outputs:positionCommand' in source
    assert 'GRIPPER_INTENT_TOPIC' in source
    assert 'SubscribeGripperIntent' in source
    assert 'close_requested = self._intent.close_requested()' in source
    assert "inputs:messageName', 'Bool'" in source
