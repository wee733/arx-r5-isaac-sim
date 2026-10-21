#!/usr/bin/env python3
"""Create a separate full-body asset package without altering supplied assets."""
import argparse
import json
import shutil
import sys
from pathlib import Path
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--finger-decomposition', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
from arx_door.envs.assets import _materialize, _extract
source = root/'generated/door_autodata/setup_20261018/assets'
shutil.copytree(source, a.output)
robot = Usd.Stage.Open(str(root/'assets/b2arx_robot/my_robot/my_robot.usd'))
robot_entry = _extract(_materialize(robot), str(robot.GetDefaultPrim().GetPath()), a.output/'robot.usd')
contact_changes=[]
if a.finger_decomposition:
    materialized=Usd.Stage.Open(str(a.output/'robot.usd'))
    for prim in materialized.Traverse():
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI) and any(f'/R5a_link{i}/collisions/' in str(prim.GetPath()) for i in (7,8)):
            mesh=UsdPhysics.MeshCollisionAPI(prim)
            contact_changes.append(dict(path=str(prim.GetPath()),before=str(mesh.GetApproximationAttr().Get()),after='convexDecomposition'))
            mesh.GetApproximationAttr().Set('convexDecomposition')
    if len(contact_changes)!=2:raise RuntimeError('expected two authored finger collision meshes')
    materialized.GetRootLayer().Save()
work = Usd.Stage.Open(str(a.output/'workcell.usd'))
removed = []
for prim in list(work.Traverse()):
    if 'BaseSupport' in str(prim.GetPath()):
        removed.append(str(prim.GetPath()))
for path in sorted(removed, key=len, reverse=True):
    work.RemovePrim(path)
work.GetRootLayer().Save()
# UsdFileCfg creates an Xform at the spawn root. A Cube authored at that root
# consequently loses its geometric type in composition. Keep geometry below it.
ground = Usd.Stage.Open(str(a.output/'ground.usd'))
wrapped = Usd.Stage.CreateInMemory()
wrapped.SetDefaultPrim(UsdGeom.Xform.Define(wrapped,'/Asset').GetPrim())
UsdGeom.SetStageMetersPerUnit(wrapped,1.0);UsdGeom.SetStageUpAxis(wrapped,'Z')
Sdf.CopySpec(ground.GetRootLayer(),'/Asset',wrapped.GetRootLayer(),'/Asset/geometry')
wrapped.Export(str(a.output/'ground.usd'))
manifest = json.loads((a.output/'manifest.json').read_text())
manifest['embodiment'] = 'b2arx_floating'
manifest['assets']['robot']['initial_pose'] = dict(position=[0.4, -0.8, 0.65], quaternion_xyzw=[0,0,0.7071067811865475,0.7071067811865476],scale=[1,1,1])
manifest['assets']['robot'].update(robot_entry)
manifest['assets']['robot']['initial_pose'] = dict(position=[0.4,-1.2,.65],quaternion_xyzw=[0,0,.7071067811865475,.7071067811865476],scale=[1,1,1])
manifest['assets']['robot'].pop('static_validation', None)
manifest['removed_support_prims'] = removed
manifest['contact_geometry_changes'] = contact_changes
manifest['env_spacing_m'] = 5.0
(a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
cache = UsdGeom.XformCache()
audit = {'removed_support_prims':removed, 'assets':{}}
for name in ('robot','door','workcell'):
    s=Usd.Stage.Open(str(a.output/f'{name}.usd'))
    rows=[]
    for prim in s.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetName()=='grasp_target':
            rows.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),schemas=prim.GetAppliedSchemas(),
                matrix=[list(row) for row in cache.GetLocalToWorldTransform(prim)],
                attributes={attr.GetName():str(attr.Get()) for attr in prim.GetAttributes() if attr.GetName().startswith(('physics:','drive:','physx'))},
                relationships={rel.GetName():list(map(str,rel.GetTargets())) for rel in prim.GetRelationships()}))
    audit['assets'][name]=rows
(a.output/'physics_audit.json').write_text(json.dumps(audit,indent=2))
print('B2ARX_ASSETS',a.output,'removed',removed)
