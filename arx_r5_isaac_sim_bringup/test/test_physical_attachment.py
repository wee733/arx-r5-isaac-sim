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
from types import SimpleNamespace

from arx_r5_isaac_sim_bringup.simulation import (
    _build_argument_parser,
    AuthoredObjectAttachmentController,
    BilateralFingerContactTracker,
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


def _controller_for_gate(aperture, mimic_aperture=None):
    controller = object.__new__(AuthoredObjectAttachmentController)
    controller._robot = _Robot(aperture, mimic_aperture)
    controller._config = SimpleNamespace(
        close_threshold=0.028,
        open_threshold=0.035,
        maximum_distance=0.10,
    )
    controller._attached = False
    controller._required_contact_steps = 3
    controller._bilateral_contact_steps = 0
    controller._contacts = BilateralFingerContactTracker(
        OBJECT,
        (LEFT, RIGHT),
    )
    controller._distance_to_grasp_frame = lambda: 0.05
    return controller


def test_controller_does_not_attach_from_aperture_and_distance_alone():
    """A closed command without both PhysX contacts must remain unattached."""
    controller = _controller_for_gate(0.0028)
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


def test_asymmetric_physical_close_uses_the_two_finger_mean():
    """An off-centre block may stop the two real fingers at different DOFs."""
    controller = _controller_for_gate(0.0301, 0.0220)
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


def test_transient_bilateral_contact_resets_consecutive_evidence():
    """A contact loss must restart the persistence gate from step zero."""
    controller = _controller_for_gate(0.0028)
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


def test_controller_releases_when_the_gripper_opens():
    """Crossing the open threshold must remove the physical fixed joint."""
    controller = _controller_for_gate(0.040)
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
    assert 'sum(finger_positions.values()) / len(finger_positions)' in source
