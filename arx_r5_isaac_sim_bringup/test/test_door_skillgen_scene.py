# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Check that a mid-motion snapshot becomes a genuinely closed trial scene."""

from pathlib import Path
import runpy

import numpy as np
import pytest

pytest.importorskip('pxr')
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_camera_canonicalization_supports_runtime_double_quaternion():
    """Pinned XformPrimView writes Quatd; preserve mounting and accept updates."""
    module = runpy.run_path(str(ROOT / 'scripts/prepare_door_skillgen_scene.py'))
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, '/Camera')
    camera.AddTranslateOp().Set(Gf.Vec3d(.1, -.01, .07))
    camera.AddOrientOp().Set(Gf.Quatf(.5, .5, -.5, -.5))
    before = np.asarray(camera.GetLocalTransformation()).copy()
    module['_canonicalize_xform'](camera.GetPrim())
    np.testing.assert_allclose(np.asarray(camera.GetLocalTransformation()), before, atol=1e-6)
    camera.GetPrim().GetAttribute('xformOp:orient').Set(Gf.Quatd(1, 0, 0, 0))
    np.testing.assert_allclose(np.asarray(camera.GetLocalTransformation())[:3, :3], np.eye(3))


def test_prepared_scene_closes_door_and_moves_mount(tmp_path):
    """Drive targets alone must not leave collision geometry at the old gap."""
    module = runpy.run_path(str(ROOT / 'scripts/prepare_door_skillgen_scene.py'))
    output = tmp_path / 'scene.usd'
    manifest = module['prepare_scene'](
        ROOT / 'generated/door_teaching/upright02/live_scene.usd',
        ROOT / 'generated/door_teaching/upright02/candidate_006',
        output,
        5.0,
    )
    stage = Usd.Stage.Open(str(output))
    # 输出目录变化后，纹理仍须解析到仓库中的实体。
    texture_paths = []
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset:
                value = attribute.Get()
                if value and value.path.endswith('.png'):
                    texture_paths.append(value.resolvedPath)
    assert len(texture_paths) == 6
    assert all(path and Path(path).is_file() for path in texture_paths)
    source_camera = stage.GetPrimAtPath('/R5a/link6/TeachingWristCamera')
    tracking_camera = stage.GetPrimAtPath('/World/SkillGenWristCamera')
    assert tracking_camera and tracking_camera.GetTypeName() == 'Camera'
    assert tracking_camera.GetParent().GetPath() == '/World'
    np.testing.assert_allclose(
        np.asarray(UsdGeom.Xformable(tracking_camera).GetLocalTransformation()),
        np.asarray(UsdGeom.Xformable(source_camera).GetLocalTransformation()),
    )
    for name in ('focalLength', 'horizontalAperture', 'verticalAperture', 'clippingRange'):
        assert tracking_camera.GetAttribute(name).Get() == source_camera.GetAttribute(name).Get()
    cache = UsdGeom.XformCache()
    expected = {
        'door_panel': (-0.378, -0.331, 0.0),
        'door_handle': (0.313, -0.3082, 0.928),
        'latch_link': (0.371, -0.3082, 0.928),
        'door_handle/grasp_target': (0.313, -0.3752, 0.928),
    }
    for name, position in expected.items():
        transform = np.asarray(cache.GetLocalToWorldTransform(
            stage.GetPrimAtPath(f'/World/Door/{name}')
        )).T
        np.testing.assert_allclose(transform[:3, 3], position, atol=1e-6)
        np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-6)
    support = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/BaseSupport'))
    np.testing.assert_allclose(
        np.asarray(support.ExtractTranslation())[:2],
        manifest['placement']['position'][:2], atol=1e-7,
    )
