#!/usr/bin/env python3
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

"""Reproduce the ARX R5A URDF FK and XRDF collision-sphere model audit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np


REFERENCE_URDF_RELATIVE = Path('urdf/r5a_cumotion.urdf')
XRDF_RELATIVE = Path('xrdf/r5a.xrdf')
DEFAULT_SEED = 20260721
DEFAULT_SAMPLES = 10_000
BASE_FRAME_FALLBACK = 'base_link'
TARGET_LINK = 'link6'
MILLIMETERS_PER_METER = 1000.0


class AuditError(RuntimeError):
    """Report an invalid input or a model-contract violation."""


@dataclass(frozen=True)
class Joint:
    """Store the URDF fields needed for forward kinematics."""

    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class CollisionMesh:
    """Describe one mesh collision element in its link frame."""

    link: str
    uri: str
    scale: np.ndarray
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray


@dataclass(frozen=True)
class UrdfModel:
    """Store the parsed kinematic and collision subset of a URDF."""

    name: str
    links: frozenset[str]
    joints: dict[str, Joint]
    child_to_joint: dict[str, Joint]
    collisions: dict[str, tuple[CollisionMesh, ...]]


@dataclass(frozen=True)
class XrdfModel:
    """Store the cuMotion fields required by this audit."""

    base_frame: str
    cspace_joint_names: tuple[str, ...]
    collision_geometry: str
    buffers: dict[str, float]
    spheres: dict[str, tuple[tuple[np.ndarray, float], ...]]


def _vector(value: str | None, default: Sequence[float]) -> np.ndarray:
    """Parse a finite three-vector from a URDF attribute."""
    if value is None:
        result = np.asarray(default, dtype=float)
    else:
        try:
            result = np.asarray([float(item) for item in value.split()], dtype=float)
        except ValueError as exc:
            raise AuditError(f'invalid numeric vector {value!r}') from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise AuditError(f'expected a finite three-vector, got {value!r}')
    return result


def _optional_float(element: ET.Element | None, attribute: str) -> float | None:
    """Parse one optional finite floating-point XML attribute."""
    if element is None or element.get(attribute) is None:
        return None
    try:
        result = float(element.attrib[attribute])
    except ValueError as exc:
        raise AuditError(
            f'invalid {attribute} value {element.attrib[attribute]!r}'
        ) from exc
    if not math.isfinite(result):
        raise AuditError(f'{attribute} must be finite, got {result}')
    return result


def _origin(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the translation and fixed-axis RPY of an optional origin."""
    if element is None:
        return np.zeros(3), np.zeros(3)
    return (
        _vector(element.get('xyz'), (0.0, 0.0, 0.0)),
        _vector(element.get('rpy'), (0.0, 0.0, 0.0)),
    )


def _parse_urdf(path: Path) -> UrdfModel:
    """Parse and validate the URDF subset used by the model audit."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise AuditError(f'cannot parse URDF {path}: {exc}') from exc
    if root.tag != 'robot':
        raise AuditError(f'{path} does not contain a URDF robot root')

    links = frozenset(link.attrib.get('name', '') for link in root.findall('link'))
    if '' in links:
        raise AuditError(f'{path} contains an unnamed link')

    joints: dict[str, Joint] = {}
    child_to_joint: dict[str, Joint] = {}
    for element in root.findall('joint'):
        name = element.attrib.get('name')
        joint_type = element.attrib.get('type')
        parent_element = element.find('parent')
        child_element = element.find('child')
        if not name or not joint_type or parent_element is None or child_element is None:
            raise AuditError(f'{path} contains an incomplete joint element')
        parent = parent_element.attrib.get('link')
        child = child_element.attrib.get('link')
        if not parent or not child or parent not in links or child not in links:
            raise AuditError(f'joint {name} refers to an unknown parent or child link')
        if name in joints or child in child_to_joint:
            raise AuditError(f'joint {name} makes the URDF kinematic tree ambiguous')

        origin_xyz, origin_rpy = _origin(element.find('origin'))
        axis_element = element.find('axis')
        axis = _vector(
            axis_element.get('xyz') if axis_element is not None else None,
            (1.0, 0.0, 0.0),
        )
        if joint_type in ('revolute', 'continuous', 'prismatic'):
            norm = float(np.linalg.norm(axis))
            if norm <= np.finfo(float).eps:
                raise AuditError(f'joint {name} has a zero-length axis')
            axis = axis / norm
        limit = element.find('limit')
        joint = Joint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            axis=axis,
            lower=_optional_float(limit, 'lower'),
            upper=_optional_float(limit, 'upper'),
        )
        joints[name] = joint
        child_to_joint[child] = joint

    collisions: dict[str, tuple[CollisionMesh, ...]] = {}
    for link in root.findall('link'):
        link_name = link.attrib['name']
        link_collisions: list[CollisionMesh] = []
        for collision in link.findall('collision'):
            geometry = collision.find('geometry')
            mesh = geometry.find('mesh') if geometry is not None else None
            if mesh is None or not mesh.get('filename'):
                continue
            origin_xyz, origin_rpy = _origin(collision.find('origin'))
            link_collisions.append(
                CollisionMesh(
                    link=link_name,
                    uri=mesh.attrib['filename'],
                    scale=_vector(mesh.get('scale'), (1.0, 1.0, 1.0)),
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                )
            )
        collisions[link_name] = tuple(link_collisions)

    return UrdfModel(
        name=root.attrib.get('name', ''),
        links=links,
        joints=joints,
        child_to_joint=child_to_joint,
        collisions=collisions,
    )


def _indent(line: str) -> int:
    """Return YAML indentation while rejecting tabs."""
    prefix = line[: len(line) - len(line.lstrip())]
    if '\t' in prefix:
        raise AuditError('tabs are not supported in the XRDF YAML indentation')
    return len(prefix)


def _top_level_block(lines: list[str], key: str) -> list[str]:
    """Extract a named top-level mapping block from the XRDF YAML subset."""
    start = None
    for index, line in enumerate(lines):
        if _indent(line) == 0 and line.strip() == f'{key}:':
            start = index + 1
            break
    if start is None:
        raise AuditError(f'XRDF is missing top-level {key!r}')
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].strip() and _indent(lines[index]) == 0:
            end = index
            break
    return lines[start:end]


def _yaml_scalar(value: str) -> str:
    """Remove matching simple YAML quotes from a scalar."""
    result = value.strip()
    if len(result) >= 2 and result[0] == result[-1] and result[0] in "'\"":
        result = result[1:-1]
    return result


def _yaml_float(value: str, context: str) -> float:
    """Parse one finite number from the intentionally small XRDF subset."""
    try:
        result = float(_yaml_scalar(value))
    except ValueError as exc:
        raise AuditError(f'invalid XRDF number for {context}: {value!r}') from exc
    if not math.isfinite(result):
        raise AuditError(f'XRDF number for {context} must be finite')
    return result


def _yaml_vector(value: str, context: str) -> np.ndarray:
    """Parse an inline three-number YAML sequence without requiring PyYAML."""
    scalar = value.strip()
    if not scalar.startswith('[') or not scalar.endswith(']'):
        raise AuditError(f'{context} must be an inline YAML sequence')
    items = [item.strip() for item in scalar[1:-1].split(',')]
    result = np.asarray([_yaml_float(item, context) for item in items], dtype=float)
    if result.shape != (3,):
        raise AuditError(f'{context} must contain exactly three numbers')
    return result


def _parse_xrdf(path: Path) -> XrdfModel:
    """Parse the stable XRDF subset with only the Python standard library."""
    try:
        raw_lines = path.read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        raise AuditError(f'cannot read XRDF {path}: {exc}') from exc
    lines = [
        line.rstrip()
        for line in raw_lines
        if line.strip() and not line.lstrip().startswith('#')
    ]

    base_frame = BASE_FRAME_FALLBACK
    for line in _top_level_block(lines, 'modifiers'):
        match = re.fullmatch(r'\s*-\s+set_base_frame:\s*(.+)', line)
        if match:
            base_frame = _yaml_scalar(match.group(1))
            break

    cspace_joint_names: tuple[str, ...] | None = None
    for line in _top_level_block(lines, 'cspace'):
        match = re.fullmatch(r'\s*joint_names:\s*\[(.*)]\s*', line)
        if match:
            cspace_joint_names = tuple(
                _yaml_scalar(item) for item in match.group(1).split(',') if item.strip()
            )
            break
    if not cspace_joint_names:
        raise AuditError('XRDF cspace.joint_names is missing or empty')

    collision_block = _top_level_block(lines, 'collision')
    collision_geometry = ''
    buffers: dict[str, float] = {}
    in_buffers = False
    for line in collision_block:
        indentation = _indent(line)
        stripped = line.strip()
        if indentation == 2 and stripped.startswith('geometry:'):
            collision_geometry = _yaml_scalar(stripped.split(':', 1)[1])
            in_buffers = False
        elif indentation == 2 and stripped == 'buffer_distance:':
            in_buffers = True
        elif indentation <= 2:
            in_buffers = False
        elif in_buffers and indentation == 4 and ':' in stripped:
            link, value = stripped.split(':', 1)
            buffers[link.strip()] = _yaml_float(value, f'buffer_distance.{link.strip()}')
    if not collision_geometry:
        raise AuditError('XRDF collision.geometry is missing')

    geometry_block = _top_level_block(lines, 'geometry')
    in_selected_geometry = False
    in_spheres = False
    current_link: str | None = None
    pending_center: np.ndarray | None = None
    sphere_lists: dict[str, list[tuple[np.ndarray, float]]] = {}
    for line in geometry_block:
        indentation = _indent(line)
        stripped = line.strip()
        if indentation == 2 and stripped.endswith(':'):
            in_selected_geometry = stripped[:-1].strip() == collision_geometry
            in_spheres = False
            current_link = None
            continue
        if not in_selected_geometry:
            continue
        if indentation == 4 and stripped == 'spheres:':
            in_spheres = True
            continue
        if not in_spheres:
            continue
        if indentation == 6 and stripped.endswith(':'):
            if pending_center is not None:
                raise AuditError('XRDF sphere center has no matching radius')
            current_link = stripped[:-1].strip()
            sphere_lists.setdefault(current_link, [])
        elif indentation == 8 and stripped.startswith('- center:'):
            if current_link is None:
                raise AuditError('XRDF sphere center appears before its link')
            if pending_center is not None:
                raise AuditError('XRDF sphere center has no matching radius')
            pending_center = _yaml_vector(
                stripped.split(':', 1)[1],
                f'geometry.{collision_geometry}.spheres.{current_link}.center',
            )
        elif indentation == 10 and stripped.startswith('radius:'):
            if current_link is None or pending_center is None:
                raise AuditError('XRDF sphere radius appears before its center')
            radius = _yaml_float(
                stripped.split(':', 1)[1],
                f'geometry.{collision_geometry}.spheres.{current_link}.radius',
            )
            if radius <= 0.0:
                raise AuditError(f'XRDF sphere on {current_link} has non-positive radius')
            sphere_lists[current_link].append((pending_center, radius))
            pending_center = None
    if pending_center is not None:
        raise AuditError('XRDF sphere center has no matching radius')
    if not sphere_lists or any(not spheres for spheres in sphere_lists.values()):
        raise AuditError(f'XRDF geometry {collision_geometry!r} has incomplete spheres')

    return XrdfModel(
        base_frame=base_frame,
        cspace_joint_names=cspace_joint_names,
        collision_geometry=collision_geometry,
        buffers=buffers,
        spheres={link: tuple(spheres) for link, spheres in sphere_lists.items()},
    )


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    """Return the URDF fixed-axis RPY rotation Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from a URDF origin."""
    result = np.eye(4)
    result[:3, :3] = _rpy_rotation(rpy)
    result[:3, 3] = xyz
    return result


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a homogeneous Rodrigues rotation about a normalized axis."""
    x_axis, y_axis, z_axis = axis
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    result = np.eye(4)
    result[:3, :3] = np.asarray(
        [
            [
                cosine + x_axis * x_axis * complement,
                x_axis * y_axis * complement - z_axis * sine,
                x_axis * z_axis * complement + y_axis * sine,
            ],
            [
                y_axis * x_axis * complement + z_axis * sine,
                cosine + y_axis * y_axis * complement,
                y_axis * z_axis * complement - x_axis * sine,
            ],
            [
                z_axis * x_axis * complement - y_axis * sine,
                z_axis * y_axis * complement + x_axis * sine,
                cosine + z_axis * z_axis * complement,
            ],
        ]
    )
    return result


def _joint_motion(joint: Joint, position: float) -> np.ndarray:
    """Build the URDF joint motion transform in the joint frame."""
    if joint.joint_type in ('revolute', 'continuous'):
        return _axis_rotation(joint.axis, position)
    if joint.joint_type == 'prismatic':
        result = np.eye(4)
        result[:3, 3] = joint.axis * position
        return result
    if joint.joint_type == 'fixed':
        return np.eye(4)
    raise AuditError(f'unsupported joint type {joint.joint_type!r} on {joint.name}')


def _chain(model: UrdfModel, base: str, target: str) -> tuple[Joint, ...]:
    """Return the unique parent-to-child joint chain from base to target."""
    if base not in model.links or target not in model.links:
        raise AuditError(f'{model.name} does not contain both {base} and {target}')
    reverse_chain: list[Joint] = []
    seen = {target}
    current = target
    while current != base:
        joint = model.child_to_joint.get(current)
        if joint is None:
            raise AuditError(f'{target} is not a descendant of {base} in {model.name}')
        reverse_chain.append(joint)
        current = joint.parent
        if current in seen:
            raise AuditError(f'{model.name} has a cycle on the path to {target}')
        seen.add(current)
    return tuple(reversed(reverse_chain))


def _forward_kinematics(chain: tuple[Joint, ...], positions: dict[str, float]) -> np.ndarray:
    """Compute base-to-target FK with T_origin followed by T_joint(q)."""
    result = np.eye(4)
    for joint in chain:
        result = result @ _transform(joint.origin_xyz, joint.origin_rpy)
        result = result @ _joint_motion(joint, positions.get(joint.name, 0.0))
    return result


def _rotation_angle(rotation: np.ndarray) -> float:
    """Return the geodesic SO(3) angle robustly in the range zero to pi."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    skew_vector = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    sine = float(np.linalg.norm(skew_vector) / 2.0)
    return math.atan2(sine, cosine)


def _distribution(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    """Return stable scalar summary statistics after applying a unit scale."""
    scaled = np.asarray(values, dtype=float) * scale
    return {
        'mean': float(np.mean(scaled)),
        'rms': float(np.sqrt(np.mean(np.square(scaled)))),
        'median': float(np.median(scaled)),
        'p95': float(np.percentile(scaled, 95.0)),
        'p99': float(np.percentile(scaled, 99.0)),
        'max': float(np.max(scaled)),
    }


def _joint_contract(
    reference: UrdfModel,
    vendor: UrdfModel,
    xrdf: XrdfModel,
    reference_chain: tuple[Joint, ...],
    vendor_chain: tuple[Joint, ...],
) -> list[dict[str, Any]]:
    """Validate joint mapping and report origin differences in common frames."""
    reference_by_name = {joint.name: joint for joint in reference_chain}
    vendor_by_name = {joint.name: joint for joint in vendor_chain}
    reference_movable = tuple(
        joint.name for joint in reference_chain if joint.joint_type != 'fixed'
    )
    vendor_movable = tuple(joint.name for joint in vendor_chain if joint.joint_type != 'fixed')
    if reference_movable != xrdf.cspace_joint_names:
        raise AuditError(
            'XRDF cspace joints do not exactly match the reference base-to-link6 chain: '
            f'{xrdf.cspace_joint_names} != {reference_movable}'
        )
    if vendor_movable != reference_movable:
        raise AuditError(
            f'vendor movable chain {vendor_movable} does not match {reference_movable}'
        )

    differences: list[dict[str, Any]] = []
    for name in xrdf.cspace_joint_names:
        reference_joint = reference_by_name[name]
        vendor_joint = vendor_by_name[name]
        if (
            reference_joint.parent != vendor_joint.parent
            or reference_joint.child != vendor_joint.child
            or reference_joint.joint_type != vendor_joint.joint_type
        ):
            raise AuditError(f'joint mapping or type differs for {name}')
        if not np.allclose(reference_joint.axis, vendor_joint.axis, atol=1.0e-12):
            raise AuditError(f'normalized joint axes differ for {name}')
        relative_rotation = _rpy_rotation(reference_joint.origin_rpy).T @ _rpy_rotation(
            vendor_joint.origin_rpy
        )
        origin_delta = vendor_joint.origin_xyz - reference_joint.origin_xyz
        differences.append(
            {
                'name': name,
                'parent': reference_joint.parent,
                'child': reference_joint.child,
                'type': reference_joint.joint_type,
                'axis_in_joint_frame': reference_joint.axis.tolist(),
                'vendor_minus_reference_origin_xyz_mm': (
                    origin_delta * MILLIMETERS_PER_METER
                ).tolist(),
                'origin_translation_delta_mm': float(
                    np.linalg.norm(origin_delta) * MILLIMETERS_PER_METER
                ),
                'origin_rotation_delta_deg': math.degrees(
                    _rotation_angle(relative_rotation)
                ),
                'reference_limits_rad': [reference_joint.lower, reference_joint.upper],
                'vendor_declared_limits_rad': [vendor_joint.lower, vendor_joint.upper],
            }
        )
    return differences


def _audit_fk(
    reference: UrdfModel,
    vendor: UrdfModel,
    xrdf: XrdfModel,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare vendor and cuMotion link6 FK over reference-limited joint samples."""
    reference_chain = _chain(reference, xrdf.base_frame, TARGET_LINK)
    vendor_chain = _chain(vendor, xrdf.base_frame, TARGET_LINK)
    origin_differences = _joint_contract(
        reference, vendor, xrdf, reference_chain, vendor_chain
    )

    limits: list[tuple[float, float]] = []
    for name in xrdf.cspace_joint_names:
        joint = reference.joints[name]
        if joint.lower is None or joint.upper is None or joint.lower >= joint.upper:
            raise AuditError(f'reference joint {name} needs finite ordered position limits')
        limits.append((joint.lower, joint.upper))
    lower = np.asarray([limit[0] for limit in limits])
    upper = np.asarray([limit[1] for limit in limits])
    positions_matrix = np.random.default_rng(seed).uniform(lower, upper, (samples, len(limits)))
    translation_errors = np.empty(samples)
    rotation_errors = np.empty(samples)
    reference_positions = np.empty((samples, 3))
    vendor_positions = np.empty((samples, 3))
    for index, row in enumerate(positions_matrix):
        positions = dict(zip(xrdf.cspace_joint_names, row, strict=True))
        reference_pose = _forward_kinematics(reference_chain, positions)
        vendor_pose = _forward_kinematics(vendor_chain, positions)
        reference_positions[index] = reference_pose[:3, 3]
        vendor_positions[index] = vendor_pose[:3, 3]
        translation_errors[index] = np.linalg.norm(
            vendor_pose[:3, 3] - reference_pose[:3, 3]
        )
        rotation_errors[index] = _rotation_angle(
            reference_pose[:3, :3].T @ vendor_pose[:3, :3]
        )

    worst = int(np.argmax(translation_errors))
    return {
        'status': 'complete',
        'reference': 'r5a_cumotion.urdf',
        'comparison': 'vendor X5liteaa0.urdf',
        'base_frame': xrdf.base_frame,
        'target_link': TARGET_LINK,
        'joint_names': list(xrdf.cspace_joint_names),
        'joint_mapping': 'same-name, same parent/child, same type, same normalized axis',
        'sampling': {
            'generator': 'numpy.random.default_rng (PCG64)',
            'distribution': 'independent uniform over reference URDF position limits',
            'seed': seed,
            'sample_count': samples,
            'reference_limits_rad': {
                name: list(limit) for name, limit in zip(xrdf.cspace_joint_names, limits)
            },
        },
        'transform_convention': (
            'URDF SI units; fixed-axis RPY Rz(yaw)*Ry(pitch)*Rx(roll); '
            'T_parent_child(q)=T_origin*T_axis(q)'
        ),
        'joint_origin_differences': origin_differences,
        'link6_translation_error_mm': _distribution(
            translation_errors, MILLIMETERS_PER_METER
        ),
        'link6_orientation_error_deg': _distribution(
            np.degrees(rotation_errors)
        ),
        'worst_translation_sample': {
            'sample_index': worst,
            'joint_positions_rad': {
                name: float(value)
                for name, value in zip(xrdf.cspace_joint_names, positions_matrix[worst])
            },
            'reference_link6_xyz_m': reference_positions[worst].tolist(),
            'vendor_link6_xyz_m': vendor_positions[worst].tolist(),
            'translation_error_mm': float(
                translation_errors[worst] * MILLIMETERS_PER_METER
            ),
        },
    }


def _package_name(description_share: Path) -> str:
    """Read the ROS package name used to resolve package mesh URIs."""
    package_xml = description_share / 'package.xml'
    try:
        root = ET.parse(package_xml).getroot()
    except (ET.ParseError, OSError) as exc:
        raise AuditError(f'cannot parse {package_xml}: {exc}') from exc
    name = root.findtext('name')
    if not name or not name.strip():
        raise AuditError(f'{package_xml} has no package name')
    return name.strip()


def _resolve_mesh_uri(
    uri: str,
    description_share: Path,
    package_name: str,
    reference_urdf: Path,
) -> Path:
    """Resolve local package, file, absolute, or URDF-relative mesh paths."""
    if uri.startswith('package://'):
        remainder = uri[len('package://'):]
        uri_package, separator, relative = remainder.partition('/')
        if not separator or uri_package != package_name:
            raise AuditError(
                f'cannot resolve mesh URI {uri!r}; expected package://{package_name}/...'
            )
        result = description_share / unquote(relative)
    elif uri.startswith('file://'):
        parsed = urlparse(uri)
        if parsed.netloc not in ('', 'localhost'):
            raise AuditError(f'non-local mesh URI is not supported: {uri!r}')
        result = Path(unquote(parsed.path))
    else:
        candidate = Path(unquote(uri))
        result = candidate if candidate.is_absolute() else reference_urdf.parent / candidate
    result = result.resolve()
    if not result.is_file():
        raise AuditError(f'mesh does not exist: {result} (from {uri!r})')
    return result


def _load_link_mesh(
    link: str,
    collision_specs: tuple[CollisionMesh, ...],
    description_share: Path,
    package_name: str,
    reference_urdf: Path,
    trimesh_module: Any,
) -> tuple[Any, list[str]]:
    """Load, scale, transform, and concatenate a link's collision meshes."""
    if not collision_specs:
        raise AuditError(f'{link} has no mesh collision geometry in the reference URDF')
    meshes = []
    source_paths = []
    for collision in collision_specs:
        path = _resolve_mesh_uri(
            collision.uri, description_share, package_name, reference_urdf
        )
        try:
            loaded = trimesh_module.load_mesh(str(path), process=False)
            if isinstance(loaded, trimesh_module.Scene):
                loaded = loaded.to_geometry()
        except Exception as exc:
            raise AuditError(f'cannot load collision mesh {path}: {exc}') from exc
        if not isinstance(loaded, trimesh_module.Trimesh):
            raise AuditError(f'{path} did not load as a triangle mesh')
        try:
            mesh = loaded.copy()
            mesh.apply_scale(collision.scale)
            mesh.apply_transform(_transform(collision.origin_xyz, collision.origin_rpy))
        except Exception as exc:
            raise AuditError(f'cannot transform collision mesh {path}: {exc}') from exc
        meshes.append(mesh)
        source_paths.append(str(path))
    try:
        combined = (
            meshes[0]
            if len(meshes) == 1
            else trimesh_module.util.concatenate(meshes)
        )
    except Exception as exc:
        raise AuditError(f'cannot combine collision meshes for {link}: {exc}') from exc
    if len(combined.vertices) == 0 or len(combined.faces) == 0:
        raise AuditError(f'{link} collision mesh has no triangles')
    return combined, source_paths


def _sample_surface(mesh: Any, count: int, seed: int) -> np.ndarray:
    """Sample mesh area uniformly with a deterministic NumPy generator."""
    triangles = np.asarray(mesh.triangles, dtype=float)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2.0
    total_area = float(np.sum(areas))
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise AuditError('collision mesh has zero or invalid surface area')
    cumulative = np.cumsum(areas)
    rng = np.random.default_rng(seed)
    face_indices = np.searchsorted(cumulative, rng.random(count) * cumulative[-1])
    selected = triangles[face_indices]
    first = np.sqrt(rng.random(count))
    second = rng.random(count)
    return (
        (1.0 - first)[:, np.newaxis] * selected[:, 0]
        + (first * (1.0 - second))[:, np.newaxis] * selected[:, 1]
        + (first * second)[:, np.newaxis] * selected[:, 2]
    )


def _sphere_gap(
    points: np.ndarray,
    spheres: tuple[tuple[np.ndarray, float], ...],
    buffer_distance: float,
) -> np.ndarray:
    """Return point-to-union signed gaps; positive values are uncovered."""
    centers = np.asarray([sphere[0] for sphere in spheres])
    radii = np.asarray([sphere[1] for sphere in spheres]) + buffer_distance
    result = np.empty(len(points))
    chunk_size = 100_000
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        distances = np.linalg.norm(
            chunk[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        result[start:start + len(chunk)] = np.min(
            distances - radii[np.newaxis, :], axis=1
        )
    return result


def _gap_metrics(gaps: np.ndarray) -> dict[str, Any]:
    """Summarize signed sphere-union gaps in millimeters."""
    return {
        'signed_gap_mm': _distribution(gaps, MILLIMETERS_PER_METER),
        'outside_fraction': float(np.mean(gaps > 0.0)),
        'outside_count': int(np.count_nonzero(gaps > 0.0)),
    }


def _audit_meshes(
    reference: UrdfModel,
    xrdf: XrdfModel,
    description_share: Path,
    reference_urdf: Path,
    samples_per_link: int,
    seed: int,
) -> dict[str, Any]:
    """Compare XRDF sphere unions with URDF collision-mesh surfaces."""
    try:
        import trimesh
    except ImportError as exc:
        return {
            'status': 'unavailable',
            'reason': f'optional dependency trimesh could not be imported: {exc}',
            'install_hint': f'{sys.executable} -m pip install trimesh',
        }

    package_name = _package_name(description_share)
    per_link: dict[str, Any] = {}
    global_vertex_gap = -math.inf
    global_buffered_vertex_gap = -math.inf
    worst_link = ''
    worst_point = np.zeros(3)
    for index, link in enumerate(sorted(xrdf.spheres)):
        if link not in reference.links:
            raise AuditError(f'XRDF collision spheres refer to unknown link {link}')
        buffer_distance = xrdf.buffers.get(link, 0.0)
        if buffer_distance < 0.0:
            raise AuditError(f'XRDF collision buffer for {link} is negative')
        mesh, source_paths = _load_link_mesh(
            link,
            reference.collisions.get(link, ()),
            description_share,
            package_name,
            reference_urdf,
            trimesh,
        )
        vertices = np.asarray(mesh.vertices, dtype=float)
        surface_points = _sample_surface(mesh, samples_per_link, seed + index + 1)
        vertex_nominal = _sphere_gap(vertices, xrdf.spheres[link], 0.0)
        vertex_buffered = _sphere_gap(vertices, xrdf.spheres[link], buffer_distance)
        surface_nominal = _sphere_gap(surface_points, xrdf.spheres[link], 0.0)
        surface_buffered = _sphere_gap(surface_points, xrdf.spheres[link], buffer_distance)
        vertex_worst_index = int(np.argmax(vertex_nominal))
        vertex_max = float(vertex_nominal[vertex_worst_index])
        if vertex_max > global_vertex_gap:
            global_vertex_gap = vertex_max
            global_buffered_vertex_gap = float(vertex_buffered[vertex_worst_index])
            worst_link = link
            worst_point = vertices[vertex_worst_index]
        per_link[link] = {
            'sphere_count': len(xrdf.spheres[link]),
            'buffer_distance_mm': buffer_distance * MILLIMETERS_PER_METER,
            'mesh_sources': source_paths,
            'mesh_vertex_count': int(len(vertices)),
            'mesh_face_count': int(len(mesh.faces)),
            'surface_sample_count': samples_per_link,
            'surface_area_uniform_nominal': _gap_metrics(surface_nominal),
            'surface_area_uniform_buffered': _gap_metrics(surface_buffered),
            'vertex_extrema': {
                'note': 'vertices are tessellation-biased; use this only for extrema',
                'max_nominal_gap_mm': vertex_max * MILLIMETERS_PER_METER,
                'max_buffered_gap_mm': float(
                    np.max(vertex_buffered) * MILLIMETERS_PER_METER
                ),
                'worst_nominal_vertex_xyz_m': vertices[vertex_worst_index].tolist(),
            },
        }

    links_outside_buffer = [
        link
        for link, result in per_link.items()
        if result['surface_area_uniform_buffered']['outside_count'] > 0
    ]
    return {
        'status': 'complete',
        'trimesh_version': getattr(trimesh, '__version__', 'unknown'),
        'xrdf_geometry': xrdf.collision_geometry,
        'signed_gap_definition': (
            'min_i(norm(point-center_i)-radius_i-buffer); positive means outside '
            'the union of spheres'
        ),
        'surface_sampling': {
            'method': 'triangle-area-weighted barycentric sampling',
            'seed': seed,
            'samples_per_link': samples_per_link,
        },
        'global_vertex_extrema': {
            'note': 'vertex extrema reproduce corner gaps but are not area-weighted statistics',
            'worst_link': worst_link,
            'worst_vertex_xyz_m': worst_point.tolist(),
            'max_nominal_gap_mm': global_vertex_gap * MILLIMETERS_PER_METER,
            'max_buffered_gap_mm_at_same_vertex': (
                global_buffered_vertex_gap * MILLIMETERS_PER_METER
            ),
        },
        'links_with_sampled_surface_outside_buffer': links_outside_buffer,
        'links': per_link,
    }


def _sha256(path: Path) -> str:
    """Hash one input file without modifying it."""
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    except OSError as exc:
        raise AuditError(f'cannot hash {path}: {exc}') from exc
    return digest.hexdigest()


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Run all requested checks and assemble the machine-readable report."""
    description_share = args.description_share.expanduser().resolve()
    vendor_urdf = args.vendor_urdf.expanduser().resolve()
    reference_urdf = description_share / REFERENCE_URDF_RELATIVE
    xrdf_path = description_share / XRDF_RELATIVE
    for label, path in (
        ('description share', description_share),
        ('reference URDF', reference_urdf),
        ('XRDF', xrdf_path),
        ('vendor URDF', vendor_urdf),
    ):
        expected = path.is_dir() if label == 'description share' else path.is_file()
        if not expected:
            raise AuditError(f'{label} does not exist at {path}')

    reference = _parse_urdf(reference_urdf)
    vendor = _parse_urdf(vendor_urdf)
    xrdf = _parse_xrdf(xrdf_path)
    fk_report = _audit_fk(reference, vendor, xrdf, args.samples, args.seed)
    if args.skip_mesh:
        mesh_report: dict[str, Any] = {
            'status': 'disabled',
            'reason': 'disabled by --skip-mesh',
        }
    else:
        mesh_report = _audit_meshes(
            reference,
            xrdf,
            description_share,
            reference_urdf,
            args.mesh_samples,
            args.seed,
        )
    status = 'complete' if mesh_report['status'] in ('complete', 'disabled') else 'partial'
    return {
        'schema_version': 1,
        'audit': 'ARX R5A vendor-vs-cuMotion model consistency',
        'status': status,
        'read_only': True,
        'dependencies': {
            'required': {'python': sys.version.split()[0], 'numpy': np.__version__},
            'mesh_audit': 'optional trimesh; PyYAML is not used',
        },
        'inputs': {
            'description_share': str(description_share),
            'reference_urdf': {
                'path': str(reference_urdf),
                'sha256': _sha256(reference_urdf),
            },
            'xrdf': {'path': str(xrdf_path), 'sha256': _sha256(xrdf_path)},
            'vendor_urdf': {
                'path': str(vendor_urdf),
                'sha256': _sha256(vendor_urdf),
            },
        },
        'fk_audit': fk_report,
        'sphere_mesh_audit': mesh_report,
    }


def _format_number(value: float) -> str:
    """Format audit table numbers without hiding sub-millimeter results."""
    return f'{value:.6f}'


def _render_text(report: dict[str, Any]) -> str:
    """Render a compact human-readable view of the JSON-equivalent report."""
    fk = report['fk_audit']
    translation = fk['link6_translation_error_mm']
    orientation = fk['link6_orientation_error_deg']
    lines = [
        'ARX R5A model audit',
        f"status: {report['status']} (read-only)",
        f"reference: {report['inputs']['reference_urdf']['path']}",
        f"vendor:    {report['inputs']['vendor_urdf']['path']}",
        f"xrdf:      {report['inputs']['xrdf']['path']}",
        '',
        'FK audit: complete',
        f"  frames: {fk['base_frame']} -> {fk['target_link']}",
        f"  joints: {', '.join(fk['joint_names'])}",
        f"  mapping: {fk['joint_mapping']}",
        f"  samples: {fk['sampling']['sample_count']} (seed {fk['sampling']['seed']})",
        f"  convention: {fk['transform_convention']}",
        '  link6 translation disagreement [mm]: '
        f"median={_format_number(translation['median'])}, "
        f"p95={_format_number(translation['p95'])}, "
        f"max={_format_number(translation['max'])}",
        '  link6 orientation disagreement [deg]: '
        f"median={_format_number(orientation['median'])}, "
        f"p95={_format_number(orientation['p95'])}, "
        f"max={_format_number(orientation['max'])}",
        '  joint origin deltas (vendor minus reference):',
    ]
    for joint in fk['joint_origin_differences']:
        lines.append(
            f"    {joint['name']}: translation="
            f"{_format_number(joint['origin_translation_delta_mm'])} mm, rotation="
            f"{_format_number(joint['origin_rotation_delta_deg'])} deg"
        )

    mesh = report['sphere_mesh_audit']
    lines.extend(['', f"Sphere-vs-mesh audit: {mesh['status']}"])
    if mesh['status'] != 'complete':
        lines.append(f"  reason: {mesh['reason']}")
        if mesh.get('install_hint'):
            lines.append(f"  install hint: {mesh['install_hint']}")
        return '\n'.join(lines)

    lines.extend(
        [
            f"  definition: {mesh['signed_gap_definition']}",
            '  area-uniform sampled surface after XRDF buffer:',
            '    link       buffer   outside       p95 gap    max gap',
        ]
    )
    for link, result in mesh['links'].items():
        buffered = result['surface_area_uniform_buffered']
        gaps = buffered['signed_gap_mm']
        lines.append(
            f"    {link:<10} {result['buffer_distance_mm']:>6.3f} mm "
            f"{buffered['outside_fraction']:>9.3%} "
            f"{gaps['p95']:>11.6f} {gaps['max']:>11.6f}"
        )
    global_extrema = mesh['global_vertex_extrema']
    lines.extend(
        [
            '  mesh-vertex worst gap (not area weighted): '
            f"{_format_number(global_extrema['max_nominal_gap_mm'])} mm nominal, "
            f"{_format_number(global_extrema['max_buffered_gap_mm_at_same_vertex'])} "
            f"mm buffered on {global_extrema['worst_link']}",
            '  note: positive signed gaps are mesh surface outside the sphere union.',
        ]
    )
    return '\n'.join(lines)


def _positive_integer(value: str) -> int:
    """Parse a strictly positive CLI integer."""
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{value!r} is not an integer') from exc
    if result <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return result


def _nonnegative_integer(value: str) -> int:
    """Parse a non-negative CLI integer suitable for NumPy SeedSequence."""
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{value!r} is not an integer') from exc
    if result < 0:
        raise argparse.ArgumentTypeError('value must be non-negative')
    return result


def _parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            'Read-only comparison of vendor X5liteaa0.urdf against the ARX R5A '
            'cuMotion URDF and XRDF.'
        )
    )
    parser.add_argument(
        '--description-share',
        required=True,
        type=Path,
        help=(
            'path to the isaac_ros_manipulation_arx_r5a_robot_description '
            'package share directory'
        ),
    )
    parser.add_argument(
        '--vendor-urdf',
        required=True,
        type=Path,
        help='path to the vendor X5liteaa0.urdf',
    )
    parser.add_argument(
        '--samples',
        type=_positive_integer,
        default=DEFAULT_SAMPLES,
        help=f'joint-space FK samples (default: {DEFAULT_SAMPLES})',
    )
    parser.add_argument(
        '--seed',
        type=_nonnegative_integer,
        default=DEFAULT_SEED,
        help=f'deterministic NumPy seed (default: {DEFAULT_SEED})',
    )
    parser.add_argument(
        '--mesh-samples',
        type=_positive_integer,
        default=DEFAULT_SAMPLES,
        help=f'area-uniform mesh samples per XRDF link (default: {DEFAULT_SAMPLES})',
    )
    parser.add_argument(
        '--output',
        choices=('text', 'json'),
        default='text',
        help='report encoding written to stdout (default: text)',
    )
    mesh_group = parser.add_mutually_exclusive_group()
    mesh_group.add_argument(
        '--skip-mesh',
        action='store_true',
        help='run FK only and do not import the optional trimesh dependency',
    )
    mesh_group.add_argument(
        '--require-mesh',
        action='store_true',
        help='return a non-zero status if trimesh is unavailable',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and write exactly one text or JSON report to stdout."""
    args = _parser().parse_args(argv)
    try:
        report = _build_report(args)
    except AuditError as exc:
        print(f'audit_model.py: error: {exc}', file=sys.stderr)
        return 2
    if args.output == 'json':
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))
    mesh_unavailable = report['sphere_mesh_audit']['status'] == 'unavailable'
    if mesh_unavailable and args.require_mesh:
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
