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

"""Pure image and planar-pose helpers for the AprilTag ROS adapter."""

from math import acos, isfinite

import cv2

import numpy as np


CUDA_CORNERS_TO_IPPE = (3, 2, 1, 0)
_IMAGE_CHANNELS = {
    'mono8': 1,
    'rgb8': 3,
    'bgr8': 3,
    'rgba8': 4,
    'bgra8': 4,
}
_COLOR_TO_GRAY = {
    'rgb8': cv2.COLOR_RGB2GRAY,
    'bgr8': cv2.COLOR_BGR2GRAY,
    'rgba8': cv2.COLOR_RGBA2GRAY,
    'bgra8': cv2.COLOR_BGRA2GRAY,
}


def is_stamp_rewind(previous, current) -> bool:
    """Return whether one exact ``(sec, nanosec)`` stamp moved backward."""
    if previous is None:
        return False
    return tuple(current) < tuple(previous)


def validate_image_message(message) -> tuple:
    """Validate ROS Image layout without decoding or copying its pixels."""
    encoding = str(message.encoding).strip().lower()
    if encoding not in _IMAGE_CHANNELS:
        supported = ', '.join(sorted(_IMAGE_CHANNELS))
        raise ValueError(
            f'unsupported image encoding {message.encoding!r}; '
            f'expected one of: {supported}'
        )

    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    if width <= 0 or height <= 0:
        raise ValueError('image width and height must be positive')
    channels = _IMAGE_CHANNELS[encoding]
    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError(
            f'image step {step} is smaller than {row_bytes} bytes of pixels'
        )
    required_bytes = step * height
    try:
        available_bytes = memoryview(message.data).nbytes
    except TypeError:
        available_bytes = len(message.data)
    if available_bytes < required_bytes:
        raise ValueError(
            f'image data has {available_bytes} bytes; expected at least '
            f'{required_bytes}'
        )
    return encoding, width, height, step, channels


def _image_buffer(message, required_bytes: int) -> np.ndarray:
    """Return a one-dimensional uint8 view when the ROS buffer permits it."""
    try:
        encoded = np.frombuffer(message.data, dtype=np.uint8)
    except TypeError:
        encoded = np.asarray(message.data, dtype=np.uint8).reshape(-1)
    return encoded[:required_bytes]


def image_roi_to_grayscale(
    message,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> np.ndarray:
    """Decode one half-open image ROI while respecting row padding."""
    encoding, width, height, step, channels = validate_image_message(message)
    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)
    if not (
        0 <= x_min < x_max <= width and
        0 <= y_min < y_max <= height
    ):
        raise ValueError('image ROI must be non-empty and inside the image')

    rows = _image_buffer(message, step * height).reshape(height, step)
    byte_min = x_min * channels
    byte_max = x_max * channels
    packed = np.ascontiguousarray(rows[y_min:y_max, byte_min:byte_max])
    roi_height = y_max - y_min
    roi_width = x_max - x_min
    if encoding == 'mono8':
        return packed.reshape(roi_height, roi_width)
    color = packed.reshape(roi_height, roi_width, channels)
    return cv2.cvtColor(color, _COLOR_TO_GRAY[encoding])


def image_to_grayscale(message) -> np.ndarray:
    """
    Decode a supported ROS Image into contiguous mono8 pixels.

    The function deliberately uses ``step`` instead of assuming tightly
    packed rows, because Isaac Sim and ROS image transports may add padding.
    It is kept independent of ROS message imports so the math tests can run
    on a minimal NumPy/OpenCV CI runner.
    """
    _, width, height, _, _ = validate_image_message(message)
    return image_roi_to_grayscale(message, 0, 0, width, height)


def refine_corners_subpixel(
    grayscale,
    corners,
    window=(4, 4),
) -> np.ndarray:
    """Refine four official detector corners without changing their order."""
    image = np.asarray(grayscale)
    # cornerSubPix updates its point array in place. Always own this buffer so
    # the caller can republish the official CUDA corner fields unchanged.
    points = np.array(corners, dtype=np.float32, copy=True)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('grayscale image must be a two-dimensional uint8 array')
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError('corners must be a finite 4-by-2 array')
    width = int(window[0])
    height = int(window[1])
    if width <= 0 or height <= 0:
        raise ValueError('subpixel window dimensions must be positive')
    if image.shape[1] < 2 * width + 5 or image.shape[0] < 2 * height + 5:
        raise ValueError('grayscale image is too small for the subpixel window')
    if (
        np.any(points[:, 0] < 0.0) or
        np.any(points[:, 0] >= image.shape[1]) or
        np.any(points[:, 1] < 0.0) or
        np.any(points[:, 1] >= image.shape[0])
    ):
        raise ValueError('corners must lie inside the grayscale image')

    refined = np.ascontiguousarray(points.reshape(4, 1, 2))
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01,
    )
    cv2.cornerSubPix(
        image,
        refined,
        (width, height),
        (-1, -1),
        criteria,
    )
    refined = refined.reshape(4, 2).astype(np.float64)
    if not np.all(np.isfinite(refined)):
        raise RuntimeError('cornerSubPix returned non-finite corners')
    return refined


def _expand_extent(
    lower: int,
    upper: int,
    limit: int,
    minimum_size: int,
) -> tuple:
    """Expand a clipped half-open interval to a required minimum size."""
    if limit < minimum_size:
        raise ValueError('image is too small for the subpixel window')
    missing = max(0, minimum_size - (upper - lower))
    grow_lower = min(lower, (missing + 1) // 2)
    lower -= grow_lower
    missing -= grow_lower
    grow_upper = min(limit - upper, missing)
    upper += grow_upper
    missing -= grow_upper
    if missing:
        lower -= missing
    return lower, upper


def refine_corners_subpixel_from_image(
    message,
    corners,
    window=(4, 4),
) -> np.ndarray:
    """Decode only the tag ROI and refine its four corners to subpixels."""
    _, image_width, image_height, _, _ = validate_image_message(message)
    points = np.asarray(corners, dtype=np.float64)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError('corners must be a finite 4-by-2 array')
    if (
        np.any(points[:, 0] < 0.0) or
        np.any(points[:, 0] >= image_width) or
        np.any(points[:, 1] < 0.0) or
        np.any(points[:, 1] >= image_height)
    ):
        raise ValueError('corners must lie inside the image')

    window_width = int(window[0])
    window_height = int(window[1])
    if window_width <= 0 or window_height <= 0:
        raise ValueError('subpixel window dimensions must be positive')
    x_min = max(
        0,
        int(np.floor(np.min(points[:, 0]))) - window_width - 2,
    )
    x_max = min(
        image_width,
        int(np.ceil(np.max(points[:, 0]))) + window_width + 3,
    )
    y_min = max(
        0,
        int(np.floor(np.min(points[:, 1]))) - window_height - 2,
    )
    y_max = min(
        image_height,
        int(np.ceil(np.max(points[:, 1]))) + window_height + 3,
    )
    x_min, x_max = _expand_extent(
        x_min,
        x_max,
        image_width,
        2 * window_width + 5,
    )
    y_min, y_max = _expand_extent(
        y_min,
        y_max,
        image_height,
        2 * window_height + 5,
    )
    grayscale_roi = image_roi_to_grayscale(
        message,
        x_min,
        y_min,
        x_max,
        y_max,
    )
    local_corners = points - np.asarray([x_min, y_min])
    refined = refine_corners_subpixel(
        grayscale_roi,
        local_corners,
        window=window,
    )
    return refined + np.asarray([x_min, y_min])


def _quaternion_matrix(quaternion) -> np.ndarray:
    """Convert a finite xyzw quaternion to a 3-by-3 rotation matrix."""
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.shape != (4,) or not isfinite(norm) or norm <= 1e-12:
        raise ValueError('quaternion must contain four finite values')
    x, y, z, w = values / norm
    return np.asarray([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])


def _matrix_quaternion(matrix: np.ndarray) -> tuple:
    """Convert a proper 3-by-3 rotation matrix to an xyzw quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError('rotation matrix must be finite and 3-by-3')
    quaternion = np.empty(4, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion[:] = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            quaternion[:] = (
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            )
        elif index == 1:
            scale = np.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            quaternion[:] = (
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            )
        else:
            scale = np.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            quaternion[:] = (
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the unsigned angular distance between two rotation matrices."""
    cosine = (float(np.trace(left.T @ right)) - 1.0) / 2.0
    return acos(max(-1.0, min(1.0, cosine)))


def refine_pose(
    corners,
    camera_matrix,
    tag_size: float,
    native_quaternion,
):
    """Refine one CUDA cuAprilTag pose without changing decoded tag axes."""
    image_points = np.asarray(corners, dtype=np.float64)
    intrinsic_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if image_points.shape != (4, 2):
        raise ValueError('corners must be a 4-by-2 array')
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError('camera_matrix must be 3-by-3')
    if not np.all(np.isfinite(image_points)):
        raise ValueError('corners must be finite')
    if not np.all(np.isfinite(intrinsic_matrix)):
        raise ValueError('camera_matrix must be finite')
    if not isfinite(tag_size) or tag_size <= 0.0:
        raise ValueError('tag_size must be finite and positive')

    native_rotation = _quaternion_matrix(native_quaternion)
    half_size = tag_size / 2.0
    object_points = np.asarray([
        [-half_size, half_size, 0.0],
        [half_size, half_size, 0.0],
        [half_size, -half_size, 0.0],
        [-half_size, -half_size, 0.0],
    ])
    # Isaac ROS 4.5's CUDA backend publishes decoded tag corners in
    # (-x,-y), (+x,-y), (+x,+y), (-x,+y) order. IPPE_SQUARE requires
    # (-x,+y), (+x,+y), (+x,-y), (-x,-y). Keep this mapping tied to the
    # decoded tag axes: sorting or cyclically rolling screen coordinates
    # would rotate the grasp/drop frame by 90-degree increments.
    ordered_points = np.ascontiguousarray(
        image_points[list(CUDA_CORNERS_TO_IPPE)]
    )
    success, rotations, translations, errors = cv2.solvePnPGeneric(
        object_points,
        ordered_points,
        intrinsic_matrix,
        None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    candidates = []
    if success:
        for rotation_vector, translation, error in zip(
            rotations,
            translations,
            np.ravel(errors),
        ):
            translation = np.ravel(translation)
            error = float(error)
            if (
                translation.shape != (3,) or
                not np.all(np.isfinite(translation)) or
                translation[2] <= 0.0 or
                not isfinite(error)
            ):
                continue
            rotation, _ = cv2.Rodrigues(rotation_vector)
            if not np.all(np.isfinite(rotation)):
                continue
            normal_alignment = float(
                rotation[:, 2] @ native_rotation[:, 2]
            )
            if normal_alignment <= 0.0:
                continue
            candidates.append((
                error,
                _rotation_distance(native_rotation, rotation),
                translation,
                rotation,
            ))

    if not candidates:
        raise RuntimeError('IPPE did not return a forward-facing tag pose')
    minimum_error = min(candidate[0] for candidate in candidates)
    equivalent = [
        candidate for candidate in candidates
        if candidate[0] <= minimum_error + 1e-4
    ]
    error, _, translation, rotation = min(
        equivalent,
        key=lambda candidate: candidate[1],
    )
    return (
        tuple(float(value) for value in translation),
        _matrix_quaternion(rotation),
        error,
    )
