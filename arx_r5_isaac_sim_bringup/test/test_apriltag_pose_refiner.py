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

"""Pure regression tests for AprilTag image and pose refinement."""

from types import SimpleNamespace

from arx_r5_isaac_sim_bringup.apriltag_pose_math import (
    _matrix_quaternion,
    _quaternion_matrix,
    image_to_grayscale,
    is_stamp_rewind,
    refine_corners_subpixel,
    refine_corners_subpixel_from_image,
    refine_pose,
    validate_image_message,
)

import cv2

import numpy as np

import pytest


def _image_message(*, encoding, width, height, step, data):
    """Build a ROS-shaped image value without importing ROS packages."""
    return SimpleNamespace(
        encoding=encoding,
        width=width,
        height=height,
        step=step,
        data=data,
    )


def test_mono8_decoder_honors_ros_row_step_padding():
    """Padding bytes must never be interpreted as image pixels."""
    message = _image_message(
        encoding='mono8',
        width=3,
        height=2,
        step=5,
        data=bytes([10, 20, 30, 201, 202, 40, 50, 60, 203, 204]),
    )

    grayscale = image_to_grayscale(message)

    assert grayscale.flags.c_contiguous
    assert grayscale.tolist() == [[10, 20, 30], [40, 50, 60]]


@pytest.mark.parametrize(
    'encoding, conversion',
    (
        ('rgb8', cv2.COLOR_RGB2GRAY),
        ('bgr8', cv2.COLOR_BGR2GRAY),
        ('rgba8', cv2.COLOR_RGBA2GRAY),
        ('bgra8', cv2.COLOR_BGRA2GRAY),
    ),
)
def test_color_decoder_supports_padding_and_channel_order(
    encoding,
    conversion,
):
    """Every advertised ROS color encoding must use its matching conversion."""
    channels = 4 if encoding in ('rgba8', 'bgra8') else 3
    pixels = np.arange(2 * 2 * channels, dtype=np.uint8).reshape(
        2,
        2,
        channels,
    )
    step = 2 * channels + 3
    padded = np.full((2, step), 255, dtype=np.uint8)
    padded[:, :2 * channels] = pixels.reshape(2, 2 * channels)
    message = _image_message(
        encoding=encoding,
        width=2,
        height=2,
        step=step,
        data=padded.tobytes(),
    )

    grayscale = image_to_grayscale(message)

    assert grayscale == pytest.approx(cv2.cvtColor(pixels, conversion))


def test_image_decoder_rejects_unsupported_or_truncated_data():
    """Malformed image messages must take the node's explicit drop path."""
    with pytest.raises(ValueError, match='unsupported image encoding'):
        image_to_grayscale(_image_message(
            encoding='16UC1',
            width=2,
            height=2,
            step=4,
            data=bytes(8),
        ))
    with pytest.raises(ValueError, match='expected at least 12'):
        image_to_grayscale(_image_message(
            encoding='rgb8',
            width=2,
            height=2,
            step=6,
            data=bytes(11),
        ))


def test_layout_validation_does_not_decode_pixels():
    """The image callback must validate a large buffer without reading it."""
    class LengthOnlyData:
        """Expose only a synthetic buffer size."""

        def __len__(self):
            """Return the advertised data length."""
            return 1920 * 3 * 1200

    message = _image_message(
        encoding='rgb8',
        width=1920,
        height=1200,
        step=1920 * 3,
        data=LengthOnlyData(),
    )

    assert validate_image_message(message) == ('rgb8', 1920, 1200, 5760, 3)


@pytest.mark.parametrize(
    'encoding, channels',
    (('mono8', 1), ('rgb8', 3), ('bgr8', 3), ('rgba8', 4), ('bgra8', 4)),
)
def test_roi_subpixel_refinement_supports_padding_and_all_encodings(
    encoding,
    channels,
):
    """Only the padded tag ROI is decoded for every supported encoding."""
    grayscale = np.zeros((64, 64), dtype=np.uint8)
    grayscale[20:45, 20:45] = 255
    if channels == 1:
        pixels = grayscale.reshape(64, 64, 1)
    else:
        pixels = np.repeat(grayscale[:, :, None], channels, axis=2)
        if channels == 4:
            pixels[:, :, 3] = 255
    step = 64 * channels + 11
    padded = np.full((64, step), 17, dtype=np.uint8)
    padded[:, :64 * channels] = pixels.reshape(64, 64 * channels)
    message = _image_message(
        encoding=encoding,
        width=64,
        height=64,
        step=step,
        data=padded.tobytes(),
    )
    official = np.asarray([
        [19.0, 19.0],
        [45.0, 19.0],
        [45.0, 45.0],
        [19.0, 45.0],
    ])

    refined = refine_corners_subpixel_from_image(message, official)

    np.testing.assert_allclose(refined, [
        [19.569607, 19.569609],
        [44.430386, 19.569609],
        [44.430386, 44.430390],
        [19.569607, 44.430390],
    ], atol=1e-5)


def test_roi_subpixel_refinement_expands_at_image_edge():
    """A clipped ROI must remain large enough for cornerSubPix."""
    image = np.zeros((32, 32), dtype=np.uint8)
    image[2:15, 2:15] = 255
    message = _image_message(
        encoding='mono8',
        width=32,
        height=32,
        step=35,
        data=np.pad(image, ((0, 0), (0, 3))).tobytes(),
    )
    corners = np.asarray([
        [1.0, 1.0],
        [15.0, 1.0],
        [15.0, 15.0],
        [1.0, 15.0],
    ])

    refined = refine_corners_subpixel_from_image(message, corners)

    np.testing.assert_allclose(refined, [
        [1.5696, 1.5696],
        [14.4304, 1.5696],
        [14.4304, 14.4304],
        [1.5696, 14.4304],
    ], atol=1e-4)


def test_exact_stamp_rewind_detection_is_stream_local_and_strict():
    """Equal stamps are duplicates; only an older stamp means a reset."""
    assert not is_stamp_rewind(None, (1, 0))
    assert not is_stamp_rewind((1, 2), (1, 2))
    assert not is_stamp_rewind((1, 2), (1, 3))
    assert is_stamp_rewind((2, 0), (1, 999_999_999))


def test_subpixel_refinement_preserves_input_and_corner_order():
    """A refined copy must not rewrite the official CUDA corners."""
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20:45, 20:45] = 255
    official = np.asarray([
        [19.0, 19.0],
        [45.0, 19.0],
        [45.0, 45.0],
        [19.0, 45.0],
    ], dtype=np.float32)
    original = official.copy()

    refined = refine_corners_subpixel(image, official, window=(4, 4))

    assert official == pytest.approx(original)
    np.testing.assert_allclose(refined, [
        [19.569607, 19.569609],
        [44.430386, 19.569609],
        [44.430386, 44.430390],
        [19.569607, 44.430390],
    ], atol=1e-5)


def test_refinement_selects_the_low_error_front_facing_ippe_solution():
    """The authored ZED view must reject cuAprilTag's ambiguous tilt."""
    camera_matrix = np.asarray([
        [735.99995232, 0.0, 960.0],
        [0.0, 735.99995232, 600.0],
        [0.0, 0.0, 1.0],
    ])
    corners = np.asarray([
        [1295.0, 798.0],
        [1291.0, 845.0],
        [1242.0, 844.0],
        [1246.0, 798.0],
    ])
    native_quaternion = (
        0.1059446707,
        0.1490738392,
        0.6972664595,
        0.6930888295,
    )

    translation, quaternion, error = refine_pose(
        corners,
        camera_matrix,
        0.04,
        native_quaternion,
    )

    assert translation == pytest.approx(
        [0.25006297, 0.17943480, 0.59657873],
        abs=1e-6,
    )
    assert error == pytest.approx(0.2061879, abs=1e-6)
    tag_normal = _quaternion_matrix(quaternion)[:, 2]
    expected_normal = np.asarray([0.0, -0.17364818, 0.98480775])
    assert float(tag_normal @ expected_normal) > 0.999


def test_refinement_rejects_invalid_geometry():
    """Malformed detections must not enter OpenCV's native solver."""
    with pytest.raises(ValueError, match='4-by-2'):
        refine_pose(
            [[0.0, 0.0]],
            np.eye(3),
            0.04,
            [0.0, 0.0, 0.0, 1.0],
        )


@pytest.mark.parametrize('yaw_degrees', (0.0, 90.0, 180.0, 270.0))
def test_cuda_corner_order_preserves_decoded_in_plane_axes(yaw_degrees):
    """Screen rotation must not rotate the decoded tag frame by 90 degrees."""
    camera_matrix = np.asarray([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 400.0],
        [0.0, 0.0, 1.0],
    ])
    base_rotation, _ = cv2.Rodrigues(
        np.asarray([0.12, -0.18, 0.04], dtype=np.float64)
    )
    yaw = np.deg2rad(yaw_degrees)
    yaw_rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    expected_rotation = base_rotation @ yaw_rotation
    expected_translation = np.asarray([0.08, -0.03, 0.65])
    half_size = 0.02
    cuda_object_points = np.asarray([
        [-half_size, -half_size, 0.0],
        [half_size, -half_size, 0.0],
        [half_size, half_size, 0.0],
        [-half_size, half_size, 0.0],
    ])
    rotation_vector, _ = cv2.Rodrigues(expected_rotation)
    corners, _ = cv2.projectPoints(
        cuda_object_points,
        rotation_vector,
        expected_translation,
        camera_matrix,
        None,
    )

    translation, quaternion, error = refine_pose(
        corners.reshape(4, 2),
        camera_matrix,
        0.04,
        _matrix_quaternion(expected_rotation),
    )

    assert translation == pytest.approx(expected_translation, abs=1e-8)
    assert _quaternion_matrix(quaternion) == pytest.approx(
        expected_rotation,
        abs=1e-8,
    )
    assert error < 1e-8
