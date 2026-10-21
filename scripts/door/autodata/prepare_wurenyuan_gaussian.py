#!/usr/bin/env python3
"""Compose the authored NuRec field with the same central-floor registration."""
import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

p=argparse.ArgumentParser()
p.add_argument('--registration',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[3]
a.output.mkdir(parents=True,exist_ok=False)
registration=json.loads(a.registration.read_text())
source=Usd.Stage.Open(str(root/'assets/wurenyuan/wry.usd'))
cache=UsdGeom.XformCache()
mesh_prior=np.asarray(cache.GetLocalToWorldTransform(source.GetPrimAtPath('/World/rec_mesh4_28'))).T
gauss_prior=np.asarray(cache.GetLocalToWorldTransform(source.GetPrimAtPath('/World/gauss'))).T
field_local=np.asarray(UsdGeom.Xformable(source.GetPrimAtPath('/World/gauss/gauss')).GetLocalTransformation()).T
level_world=np.asarray(registration['source_to_center'])@np.linalg.inv(mesh_prior)
gauss_center=level_world@gauss_prior
field_center=gauss_center@field_local
bounds=np.asarray(registration['crop_in_leveled_frame_m'])-np.asarray(registration['center_shift_m'])
corners=np.array(list(itertools.product(*zip(bounds[0],bounds[1]))))
inverse=np.linalg.inv(field_center)
local=corners@inverse[:3,:3].T+inverse[:3,3]
crop=np.stack([local.min(axis=0),local.max(axis=0)])
shutil.copy2(root/'assets/wurenyuan/point_cloud.usdz',a.output/'point_cloud.usdz')
stage=Usd.Stage.CreateNew(str(a.output/'wurenyuan_gaussian_center.usda'))
world=UsdGeom.Xform.Define(stage,'/Wurenyuan');stage.SetDefaultPrim(world.GetPrim())
UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
gaussian=UsdGeom.Xform.Define(stage,'/Wurenyuan/Gaussian')
gaussian.GetPrim().GetReferences().AddReference('./point_cloud.usdz','/World/gauss')
gaussian.ClearXformOpOrder();gaussian.AddTransformOp().Set(Gf.Matrix4d(gauss_center.T.tolist()))
field=stage.GetPrimAtPath('/Wurenyuan/Gaussian/gauss')
assert field.GetTypeName()=='Volume'
for label,value in zip(('minBounds','maxBounds'),crop):
    field.GetAttribute('omni:nurec:crop:'+label).Set(Gf.Vec3f(*value))
stage.GetRootLayer().customLayerData=Usd.Stage.Open(str(a.output/'point_cloud.usdz')).GetRootLayer().customLayerData
stage.GetRootLayer().Save()
report=dict(source='assets/wurenyuan/point_cloud.usdz',authored_alignment='assets/wurenyuan/wry.usd:/World/gauss',
    representation='native NuRec Volume / OmniNuRecFieldAsset',mesh_registration=str(a.registration),
    authored_gaussian_to_world=gauss_prior.tolist(),authored_world_to_center=level_world.tolist(),
    gaussian_root_to_center=gauss_center.tolist(),field_to_center=field_center.tolist(),
    requested_center_bounds_m=bounds.tolist(),native_field_crop_bounds=crop.tolist(),
    crop_scope='native field-axis AABB enclosing the requested leveled central region; rotation produces a conservative boundary',
    physics='visual field only; original task collision geometry is retained')
(a.output/'registration.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
