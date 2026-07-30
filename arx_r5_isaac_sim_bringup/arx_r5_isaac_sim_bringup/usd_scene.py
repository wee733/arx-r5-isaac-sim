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

"""Load the authored ARX workcell and dual-camera ROS contracts."""

from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


@dataclass(frozen=True)
class CameraPublisherConfig:
    """One authored USD Camera and its ROS image/TF contract."""

    name: str
    camera_prim_path: str
    parent_prim_path: str
    parent_frame: str
    optical_frame: str
    expected_parent_to_optical_translation: Tuple[float, float, float]
    expected_optical_forward: Tuple[float, float, float]
    width: int
    height: int
    frame_skip_count: int
    color_image_topic: str
    color_info_topic: str
    depth_image_topic: str
    depth_info_topic: str


@dataclass(frozen=True)
class UsdSceneConfig:
    """Validated paths and sensor profiles for the authored USD scene."""

    robot_prim_path: str
    base_prim_path: str
    world_frame: str
    base_frame: str
    expected_world_to_base_translation: Tuple[float, float, float]
    expected_world_to_base_rotation: Tuple[float, float, float, float]
    sensor_rigid_body_paths: Tuple[str, ...]
    source_object_prim_path: str
    source_collision_prim_path: str
    grasp_body_prim_path: str
    grasp_frame_prim_path: str
    source_object_mass_kg: float
    cameras: Dict[str, CameraPublisherConfig]
    # The VLA workcell carries no AprilTags. Both stay None when the optional
    # `tags` section is absent, and the runtime skips texture repair.
    source_tag_texture_prim: Optional[str]
    target_tag_texture_prim: Optional[str]

    @property
    def has_tag_textures(self) -> bool:
        """Return whether this scene declares AprilTag texture shaders."""
        return (
            self.source_tag_texture_prim is not None and
            self.target_tag_texture_prim is not None
        )


def _absolute_prim_path(value, field_name: str) -> str:
    path = str(value or '').strip()
    if not path.startswith('/') or path == '/':
        raise ValueError(f'{field_name} must be an absolute USD prim path')
    return path


def _frame(value, field_name: str) -> str:
    frame = str(value or '').strip().strip('/')
    if not frame or any(character.isspace() for character in frame):
        raise ValueError(f'{field_name} must be a non-empty ROS frame ID')
    return frame


def _topic(value, field_name: str) -> str:
    topic = str(value or '').strip()
    if not topic.startswith('/') or topic == '/':
        raise ValueError(f'{field_name} must be an absolute ROS topic')
    return topic


def _positive_integer(value, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f'{field_name} must be greater than zero')
    return parsed


def _finite_vector(value, size: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'{field_name} must contain {size} numeric values')
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'{field_name} must contain {size} numeric values'
        ) from error
    if not all(isfinite(component) for component in vector):
        raise ValueError(f'{field_name} must contain only finite values')
    return vector


def _unit_vector(value, size: int, field_name: str) -> tuple[float, ...]:
    vector = _finite_vector(value, size, field_name)
    magnitude = sqrt(sum(component * component for component in vector))
    if magnitude <= 1e-12:
        raise ValueError(f'{field_name} must not be a zero vector')
    return tuple(component / magnitude for component in vector)


def load_usd_scene_config(path: str | Path) -> UsdSceneConfig:
    """Load the camera profiles without importing Isaac Sim or USD modules."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'USD scene config not found: {config_path}')
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or raw.get('schema_version') != 1:
        raise ValueError('USD scene config must use schema_version 1')

    robot = raw.get('robot', {})
    camera_map = raw.get('cameras', {})
    if not isinstance(camera_map, dict) or not camera_map:
        raise ValueError('USD scene config must define at least one camera')

    cameras = {}
    frames = set()
    topics = set()
    for name, camera_raw in camera_map.items():
        normalized_name = str(name).strip()
        if not normalized_name or not normalized_name.replace('_', '').isalnum():
            raise ValueError(f'invalid camera profile name: {name!r}')
        if not isinstance(camera_raw, dict):
            raise ValueError(f'cameras.{normalized_name} must be a mapping')
        optical_frame = _frame(
            camera_raw.get('optical_frame'),
            f'cameras.{normalized_name}.optical_frame',
        )
        camera_topics = (
            _topic(
                camera_raw.get('color_image_topic'),
                f'cameras.{normalized_name}.color_image_topic',
            ),
            _topic(
                camera_raw.get('color_info_topic'),
                f'cameras.{normalized_name}.color_info_topic',
            ),
            _topic(
                camera_raw.get('depth_image_topic'),
                f'cameras.{normalized_name}.depth_image_topic',
            ),
            _topic(
                camera_raw.get('depth_info_topic'),
                f'cameras.{normalized_name}.depth_info_topic',
            ),
        )
        if optical_frame in frames:
            raise ValueError(f'duplicate camera optical frame: {optical_frame}')
        duplicate_topics = topics.intersection(camera_topics)
        if duplicate_topics:
            raise ValueError(f'duplicate camera topics: {sorted(duplicate_topics)}')
        frames.add(optical_frame)
        topics.update(camera_topics)
        cameras[normalized_name] = CameraPublisherConfig(
            name=normalized_name,
            camera_prim_path=_absolute_prim_path(
                camera_raw.get('camera_prim_path'),
                f'cameras.{normalized_name}.camera_prim_path',
            ),
            parent_prim_path=_absolute_prim_path(
                camera_raw.get('parent_prim_path'),
                f'cameras.{normalized_name}.parent_prim_path',
            ),
            parent_frame=_frame(
                camera_raw.get('parent_frame'),
                f'cameras.{normalized_name}.parent_frame',
            ),
            optical_frame=optical_frame,
            expected_parent_to_optical_translation=_finite_vector(
                camera_raw.get('expected_parent_to_optical_translation'),
                3,
                f'cameras.{normalized_name}.'
                'expected_parent_to_optical_translation',
            ),
            expected_optical_forward=_unit_vector(
                camera_raw.get('expected_optical_forward'),
                3,
                f'cameras.{normalized_name}.expected_optical_forward',
            ),
            width=_positive_integer(
                camera_raw.get('width'),
                f'cameras.{normalized_name}.width',
            ),
            height=_positive_integer(
                camera_raw.get('height'),
                f'cameras.{normalized_name}.height',
            ),
            frame_skip_count=int(camera_raw.get('frame_skip_count', 0)),
            color_image_topic=camera_topics[0],
            color_info_topic=camera_topics[1],
            depth_image_topic=camera_topics[2],
            depth_info_topic=camera_topics[3],
        )
        if cameras[normalized_name].frame_skip_count < 0:
            raise ValueError(
                f'cameras.{normalized_name}.frame_skip_count must be non-negative'
            )

    rigid_body_paths = tuple(
        _absolute_prim_path(path_value, 'robot.sensor_rigid_body_paths')
        for path_value in robot.get('sensor_rigid_body_paths', [])
    )
    source_object = raw.get('source_object', {})
    if not isinstance(source_object, dict):
        raise ValueError('source_object must be a mapping')
    source_object_mass_kg = float(source_object.get('mass_kg', 0.0))
    if source_object_mass_kg <= 0.0:
        raise ValueError('source_object.mass_kg must be greater than zero')

    tags = raw.get('tags')
    if tags is None:
        source_tag_texture_prim = None
        target_tag_texture_prim = None
    elif isinstance(tags, dict):
        source_tag_texture_prim = _absolute_prim_path(
            tags.get('source_texture_prim'), 'tags.source_texture_prim'
        )
        target_tag_texture_prim = _absolute_prim_path(
            tags.get('target_texture_prim'), 'tags.target_texture_prim'
        )
    else:
        raise ValueError('tags must be a mapping when present')

    return UsdSceneConfig(
        robot_prim_path=_absolute_prim_path(
            robot.get('robot_prim_path'), 'robot.robot_prim_path'
        ),
        base_prim_path=_absolute_prim_path(
            robot.get('base_prim_path'), 'robot.base_prim_path'
        ),
        world_frame=_frame(robot.get('world_frame', 'world'), 'robot.world_frame'),
        base_frame=_frame(robot.get('base_frame', 'base_link'), 'robot.base_frame'),
        expected_world_to_base_translation=_finite_vector(
            robot.get('expected_world_to_base_translation'),
            3,
            'robot.expected_world_to_base_translation',
        ),
        expected_world_to_base_rotation=_unit_vector(
            robot.get('expected_world_to_base_rotation'),
            4,
            'robot.expected_world_to_base_rotation',
        ),
        sensor_rigid_body_paths=rigid_body_paths,
        source_object_prim_path=_absolute_prim_path(
            source_object.get('prim_path'), 'source_object.prim_path'
        ),
        source_collision_prim_path=_absolute_prim_path(
            source_object.get('collision_prim_path'),
            'source_object.collision_prim_path',
        ),
        grasp_body_prim_path=_absolute_prim_path(
            source_object.get('grasp_body_prim_path'),
            'source_object.grasp_body_prim_path',
        ),
        grasp_frame_prim_path=_absolute_prim_path(
            source_object.get('grasp_frame_prim_path'),
            'source_object.grasp_frame_prim_path',
        ),
        source_object_mass_kg=source_object_mass_kg,
        cameras=cameras,
        source_tag_texture_prim=source_tag_texture_prim,
        target_tag_texture_prim=target_tag_texture_prim,
    )
