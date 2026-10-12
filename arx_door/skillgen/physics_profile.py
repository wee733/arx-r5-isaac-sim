# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Explicit PhysX execution profiles, independent of simulator startup."""

from __future__ import annotations

import math

PHYSICS_PROFILES = ('lab_default', 'source_cpu')

# Reviewed installed Sim5.1 PhysxSceneAPI schema fallbacks and Core teacher
# defaults. Authored source values take precedence. These apply only on opt-in;
# the Lab profile and actor/material/collision settings are left unchanged.
SOURCE_CPU_SCENE = {
    'broadphaseType': 'MBP', 'enableGPUDynamics': False,
    'enableCCD': True, 'solverType': 'TGS', 'enableStabilization': False,
    'frictionCorrelationDistance': 0.025, 'frictionOffsetThreshold': 0.04,
    'bounceThreshold': 0.0, 'enableExternalForcesEveryIteration': False,
    'minPositionIterationCount': 1, 'maxPositionIterationCount': 255,
    'minVelocityIterationCount': 0, 'maxVelocityIterationCount': 255,
    'frictionType': 'patch', 'timeStepsPerSecond': 120,
}
SOURCE_CPU_REFERENCE = 'generated/door_quality/fresh51_skillgen_target_chain_20261002/source_cpu_physics_spec.json'
SOURCE_CPU_GRAVITY = {'gravityDirection': [0.0, 0.0, -1.0], 'gravityMagnitude': 9.8100004196167}
CFG_FIELDS = {
    'enable_ccd': 'enableCCD', 'enable_stabilization': 'enableStabilization',
    'friction_correlation_distance': 'frictionCorrelationDistance',
    'friction_offset_threshold': 'frictionOffsetThreshold',
    'bounce_threshold_velocity': 'bounceThreshold',
    'enable_external_forces_every_iteration': 'enableExternalForcesEveryIteration',
    'min_position_iteration_count': 'minPositionIterationCount',
    'max_position_iteration_count': 'maxPositionIterationCount',
    'min_velocity_iteration_count': 'minVelocityIterationCount',
    'max_velocity_iteration_count': 'maxVelocityIterationCount',
}


def physics_device(profile: str, device: str) -> str:
    """The source profile selects CPU physics; RTX rendering remains available."""
    if profile not in PHYSICS_PROFILES:
        raise ValueError(f'unknown physics profile: {profile}')
    return 'cpu' if profile == 'source_cpu' else device


def configuration_identity(configuration: dict | None) -> dict | None:
    """Old saved configurations used the unchanged Lab profile implicitly."""
    if configuration is None:
        return None
    return {'physics_profile': 'lab_default', **configuration}


def _value(prim, name: str):
    attr = prim.GetAttribute(name)
    return attr.Get() if attr else None


def source_cpu_scene_settings(stage, scene_path: str = '/physicsScene') -> dict:
    """Read source values before Lab overwrites them, retaining schema provenance."""
    prim = stage.GetPrimAtPath(scene_path)
    if not prim or prim.GetTypeName() != 'PhysicsScene':
        raise ValueError(f'source_cpu requires the source PhysicsScene at {scene_path}')
    values, origins, current_schema = {}, {}, {}
    for key, fallback in SOURCE_CPU_SCENE.items():
        name = 'physxScene:' + key
        attr = prim.GetAttribute(name)
        value = attr.Get() if attr else None
        # A replay SDK's schema fallback is not evidence of the recording SDK's
        # setting. Pin missing values to the reviewed Core/Sim5.1 source spec.
        authored = bool(attr and attr.HasAuthoredValueOpinion())
        values[key] = value if authored else fallback
        current_schema[key] = value
        origins[key] = ('authored_source' if authored
                        else 'reviewed_sim51_schema_fallback')
    for key in ('broadphaseType', 'enableGPUDynamics', 'enableCCD', 'solverType', 'timeStepsPerSecond'):
        if values[key] != SOURCE_CPU_SCENE[key]:
            raise ValueError(f'source_cpu source scene has incompatible {key}={values[key]}')
    gravity = {}
    for key, fallback in SOURCE_CPU_GRAVITY.items():
        attr = prim.GetAttribute('physics:' + key)
        authored = bool(attr and attr.HasAuthoredValueOpinion())
        value = attr.Get() if authored else fallback
        gravity[key] = list(value) if key == 'gravityDirection' else float(value)
    direction, magnitude = gravity['gravityDirection'], gravity['gravityMagnitude']
    if (not all(math.isfinite(value) for value in (*direction, magnitude)) or magnitude <= 0
            or not math.isclose(sum(value * value for value in direction), 1.0, abs_tol=1e-6)):
        raise ValueError('source_cpu requires finite gravity with a unit direction and positive magnitude')
    return {'scene_path': scene_path, 'values': values, 'value_origins': origins,
            'current_schema_values': current_schema, 'gravity': gravity,
            'reference_source_profile': 'Core_Sim5.1_CPU', 'reviewed_reference': SOURCE_CPU_REFERENCE}


def configure_physics_profile(sim_cfg, profile: str, source_settings: dict | None = None) -> None:
    """Configure both Lab ABIs before constructing their SimulationContext."""
    physics_device(profile, sim_cfg.device)
    if profile == 'lab_default':
        return
    if source_settings is None:
        raise ValueError('source_cpu requires reviewed source scene settings')
    backend = getattr(sim_cfg, 'physics', None) or getattr(sim_cfg, 'physx', None)
    if backend is None:
        raise RuntimeError('selected Lab has no PhysX configuration')
    for field in (*CFG_FIELDS, 'solver_type'):
        if not hasattr(backend, field):
            raise RuntimeError(f'selected Lab cannot configure source_cpu {field}')
    if not math.isclose(sim_cfg.dt, 1.0 / 120.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('source_cpu requires 1/120 second physics dt')
    sim_cfg.device = 'cpu'
    gravity = source_settings['gravity']
    sim_cfg.gravity = tuple(value * gravity['gravityMagnitude'] for value in gravity['gravityDirection'])
    for field, attr in CFG_FIELDS.items():
        setattr(backend, field, source_settings['values'][attr])
    backend.solver_type = 1


def physics_material_bindings(stage) -> dict:
    """Record existing physical bindings; this does not edit friction or materials."""
    bindings = {}
    for prim in stage.Traverse():
        for relationship in prim.GetRelationships():
            if relationship.GetName().startswith('material:binding:physics'):
                targets = relationship.GetTargets()
                if targets:
                    bindings[str(relationship.GetPath())] = [str(path) for path in targets]
    return bindings


def physics_runtime_evidence(stage, sim_cfg, simulation_device: str, profile: str,
                             *, source_settings: dict | None = None,
                             original_bindings: dict | None = None) -> dict:
    """Read the live post-reset scene, including actual schema values and new defaults."""
    path = sim_cfg.physics_prim_path
    prim = stage.GetPrimAtPath(path)
    if not prim or prim.GetTypeName() != 'PhysicsScene':
        raise RuntimeError(f'live PhysicsScene is missing at {path}')
    scene_values = {key: _value(prim, 'physxScene:' + key) for key in SOURCE_CPU_SCENE}
    gravity = {'gravityDirection': _value(prim, 'physics:gravityDirection'),
               'gravityMagnitude': _value(prim, 'physics:gravityMagnitude')}
    if gravity['gravityDirection'] is not None:
        gravity['gravityDirection'] = list(gravity['gravityDirection'])
    material_path = path + '/defaultMaterial'
    material = stage.GetPrimAtPath(material_path)
    bindings = physics_material_bindings(stage)
    existing_preserved = (original_bindings is None or
                          all(bindings.get(key) == value for key, value in original_bindings.items()))
    return {
        'schema': 'arx-door-physics-runtime-v1', 'physics_profile': profile,
        'configured_device': str(sim_cfg.device), 'simulation_device': str(simulation_device),
        'use_fabric': bool(sim_cfg.use_fabric), 'physics_dt_s': float(sim_cfg.dt),
        'physics_scene_path': path, 'physics_scene_applied_schemas': list(prim.GetAppliedSchemas()),
        'physics_scene_values': scene_values,
        'physics_scene_gravity': gravity,
        'source_scene_settings': source_settings,
        'original_physics_material_binding_count': len(original_bindings or {}),
        'original_physics_material_bindings_preserved': existing_preserved,
        'scene_physics_material_bindings': {key: value for key, value in bindings.items()
                                            if key.startswith(path + '.')},
        'lab_default_material': {'path': material_path, 'exists': bool(material),
            'values': {name: _value(material, 'physics:' + name) if material else None
                       for name in ('staticFriction', 'dynamicFriction', 'restitution')}},
    }


def validate_physics_runtime(evidence: dict) -> None:
    """Fail closed when an opt-in source profile was silently overwritten."""
    if evidence['physics_profile'] == 'lab_default':
        return
    if evidence['configured_device'] != 'cpu' or evidence['simulation_device'] != 'cpu':
        raise RuntimeError('source_cpu requires actual CPU physics execution')
    if 'PhysxSceneAPI' not in evidence['physics_scene_applied_schemas']:
        raise RuntimeError('source_cpu requires visible live PhysxSceneAPI schema')
    if not math.isclose(evidence['physics_dt_s'], 1.0 / 120.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError('source_cpu runtime physics dt differs from 1/120 second')
    if not evidence['original_physics_material_bindings_preserved']:
        raise RuntimeError('source_cpu source physical material bindings changed')
    for key, expected in evidence['source_scene_settings']['values'].items():
        actual = evidence['physics_scene_values'].get(key)
        equal = (math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)
                 if isinstance(expected, float) and isinstance(actual, (float, int))
                 else actual == expected)
        if not equal:
            raise RuntimeError(f'source_cpu runtime mismatch: {key}={actual}, source={expected}')
    expected = evidence['source_scene_settings']['gravity']
    actual = evidence['physics_scene_gravity']
    if (actual['gravityDirection'] is None or actual['gravityMagnitude'] is None
            or any(not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)
                   for a, b in zip(actual['gravityDirection'], expected['gravityDirection']))
            or not math.isclose(actual['gravityMagnitude'], expected['gravityMagnitude'], rel_tol=1e-6, abs_tol=1e-8)):
        raise RuntimeError('source_cpu runtime gravity differs from the source scene')
