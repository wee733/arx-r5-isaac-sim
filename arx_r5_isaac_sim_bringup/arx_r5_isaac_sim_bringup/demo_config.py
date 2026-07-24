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

"""Load and validate the manipulation workcell configuration."""

from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Tuple

import yaml


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


def _vector(values, size: int, field_name: str) -> tuple:
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f'{field_name} must contain {size} numeric values')
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{field_name} must contain only numeric values') from error
    if not all(isfinite(value) for value in vector):
        raise ValueError(f'{field_name} must contain only finite values')
    return vector


def _positive(value, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f'{field_name} must be greater than zero')
    return number


@dataclass(frozen=True)
class TableConfig:
    """Static table dimensions and placement."""

    top_center: Vector3
    top_size: Vector3
    leg_size: Vector3
    leg_inset: float


@dataclass(frozen=True)
class SourceObjectConfig:
    """Tagged cube geometry and placement."""

    center: Vector3
    size: Vector3
    tag_id: int
    tag_size: float
    tag_quad_size: float
    tag_texture: Path


@dataclass(frozen=True)
class DropTargetConfig:
    """Tagged destination and its desired tag-to-link6 goal transform."""

    center: Vector3
    tag_id: int
    tag_size: float
    tag_quad_size: float
    tag_texture: Path
    link6_offset_in_tag: Vector3
    link6_rotation_in_tag: Quaternion


@dataclass(frozen=True)
class AttachmentConfig:
    """Deterministic visual attachment thresholds."""

    grasp_frame_offset: Vector3
    close_threshold: float
    open_threshold: float
    maximum_distance: float


@dataclass(frozen=True)
class SourceZoneConfig:
    """Optional fixed-frame discovery bounds for source objects."""

    enabled: bool
    frame: str
    minimum: Vector3
    maximum: Vector3


@dataclass(frozen=True)
class DemoConfig:
    """Complete manipulation workcell configuration."""

    tag_family: str
    table: TableConfig
    source_object: SourceObjectConfig
    drop_target: DropTargetConfig
    attachment: AttachmentConfig
    source_zone: SourceZoneConfig


def _tag_config(raw: dict, base_path: Path, field_name: str):
    tag_id = int(raw.get('tag_id', -1))
    if tag_id < 0:
        raise ValueError(f'{field_name}.tag_id must be non-negative')
    tag_size = _positive(raw.get('tag_size'), f'{field_name}.tag_size')
    tag_quad_size = _positive(
        raw.get('tag_quad_size'),
        f'{field_name}.tag_quad_size',
    )
    if tag_quad_size <= tag_size:
        raise ValueError(f'{field_name}.tag_quad_size must exceed tag_size')
    texture = (base_path / str(raw.get('tag_texture', ''))).resolve()
    if not texture.is_file():
        raise FileNotFoundError(f'{field_name} texture not found: {texture}')
    return tag_id, tag_size, tag_quad_size, texture


def load_demo_config(path: str | Path) -> DemoConfig:
    """Load a tabletop demo configuration from YAML."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'demo config not found: {config_path}')
    with config_path.open('r', encoding='utf-8') as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict) or raw.get('schema_version') != 1:
        raise ValueError('demo config must use schema_version 1')

    tag_family = str(raw.get('tag_family', '')).strip()
    if not tag_family:
        raise ValueError('tag_family must be non-empty')

    table_raw = raw.get('table', {})
    table = TableConfig(
        top_center=_vector(table_raw.get('top_center'), 3, 'table.top_center'),
        top_size=_vector(table_raw.get('top_size'), 3, 'table.top_size'),
        leg_size=_vector(table_raw.get('leg_size'), 3, 'table.leg_size'),
        leg_inset=_positive(table_raw.get('leg_inset'), 'table.leg_inset'),
    )
    if any(dimension <= 0.0 for dimension in table.top_size + table.leg_size):
        raise ValueError('table dimensions must be greater than zero')

    source_raw = raw.get('source_object', {})
    source_tag = _tag_config(
        source_raw,
        config_path.parent,
        'source_object',
    )
    source = SourceObjectConfig(
        center=_vector(source_raw.get('center'), 3, 'source_object.center'),
        size=_vector(source_raw.get('size'), 3, 'source_object.size'),
        tag_id=source_tag[0],
        tag_size=source_tag[1],
        tag_quad_size=source_tag[2],
        tag_texture=source_tag[3],
    )
    if any(dimension <= 0.0 for dimension in source.size):
        raise ValueError('source_object.size values must be greater than zero')

    drop_raw = raw.get('drop_target', {})
    drop_tag = _tag_config(drop_raw, config_path.parent, 'drop_target')
    drop_rotation = _vector(
        drop_raw.get('link6_rotation_in_tag'),
        4,
        'drop_target.link6_rotation_in_tag',
    )
    rotation_magnitude = sqrt(sum(value * value for value in drop_rotation))
    if rotation_magnitude <= 1e-12:
        raise ValueError('drop_target.link6_rotation_in_tag must be non-zero')
    drop = DropTargetConfig(
        center=_vector(drop_raw.get('center'), 3, 'drop_target.center'),
        tag_id=drop_tag[0],
        tag_size=drop_tag[1],
        tag_quad_size=drop_tag[2],
        tag_texture=drop_tag[3],
        link6_offset_in_tag=_vector(
            drop_raw.get('link6_offset_in_tag'),
            3,
            'drop_target.link6_offset_in_tag',
        ),
        link6_rotation_in_tag=tuple(
            value / rotation_magnitude for value in drop_rotation
        ),
    )
    if source.tag_id == drop.tag_id:
        raise ValueError('source and drop target tags must use different IDs')
    if abs(source.tag_size - drop.tag_size) > 1e-9:
        raise ValueError('source and drop target tag sizes must match')

    attachment_raw = raw.get('attachment', {})
    attachment = AttachmentConfig(
        grasp_frame_offset=_vector(
            attachment_raw.get('grasp_frame_offset'),
            3,
            'attachment.grasp_frame_offset',
        ),
        close_threshold=_positive(
            attachment_raw.get('close_threshold'),
            'attachment.close_threshold',
        ),
        open_threshold=_positive(
            attachment_raw.get('open_threshold'),
            'attachment.open_threshold',
        ),
        maximum_distance=_positive(
            attachment_raw.get('maximum_distance'),
            'attachment.maximum_distance',
        ),
    )
    if attachment.close_threshold >= attachment.open_threshold:
        raise ValueError('attachment close threshold must be below open threshold')

    source_zone_raw = raw.get('source_zone', {})
    if not isinstance(source_zone_raw, dict):
        raise ValueError('source_zone must be a mapping')
    source_zone_enabled = source_zone_raw.get('enabled', False)
    if not isinstance(source_zone_enabled, bool):
        raise ValueError('source_zone.enabled must be a boolean')
    source_zone_frame = source_zone_raw.get('frame', '')
    if not isinstance(source_zone_frame, str):
        raise ValueError('source_zone.frame must be a string')
    source_zone_frame = source_zone_frame.strip()
    if source_zone_enabled:
        if not source_zone_frame:
            raise ValueError(
                'source_zone.frame must be non-empty when enabled'
            )
        for field in ('min_xyz', 'max_xyz'):
            if field not in source_zone_raw:
                raise ValueError(
                    f'source_zone.{field} is required when enabled'
                )
    source_zone = SourceZoneConfig(
        enabled=source_zone_enabled,
        frame=source_zone_frame,
        minimum=_vector(
            source_zone_raw.get('min_xyz', [0.0, 0.0, 0.0]),
            3,
            'source_zone.min_xyz',
        ),
        maximum=_vector(
            source_zone_raw.get('max_xyz', [0.0, 0.0, 0.0]),
            3,
            'source_zone.max_xyz',
        ),
    )
    if any(
        minimum > maximum
        for minimum, maximum in zip(
            source_zone.minimum,
            source_zone.maximum,
        )
    ):
        raise ValueError(
            'source_zone.min_xyz must not exceed source_zone.max_xyz'
        )

    return DemoConfig(
        tag_family=tag_family,
        table=table,
        source_object=source,
        drop_target=drop,
        attachment=attachment,
        source_zone=source_zone,
    )
