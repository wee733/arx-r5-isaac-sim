#!/usr/bin/env python3
"""Compose aligned Gaussian visuals and mesh geometry once for an Arena grid."""
import argparse
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Vt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--columns',type=int,default=8)
p.add_argument('--rows',type=int,default=3)
p.add_argument('--spacing-x',type=float,default=3.6)
p.add_argument('--spacing-y',type=float,default=5.4)
p.add_argument('--gaussian-asset',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[3]
mesh_path=root/'generated/door_autodata/b2arx_aug_20261019/scene_center/wurenyuan_center.usdc'
gauss_path=root/'generated/door_autodata/b2arx_gaussian_20261019/scene_center/wurenyuan_particlefield_center.usdc'
if a.gaussian_asset:gauss_path=a.gaussian_asset.resolve()
s=Usd.Stage.Open(str(mesh_path));mesh=UsdGeom.Mesh(s.GetPrimAtPath('/Wurenyuan/Reconstruction'))
vertices=np.asarray(mesh.GetPointsAttr().Get());faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
yaw_deg=-13.
theta=np.deg2rad(yaw_deg)
rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
leveled=vertices@rotation.T
triangle_min=leveled[faces].min(axis=1);triangle_max=leveled[faces].max(axis=1)
nx,ny=a.columns,a.rows
origins=np.array([[-(i-(nx-1)/2)*a.spacing_x,(j-(ny-1)/2)*a.spacing_y,0.] for i in range(nx) for j in range(ny)])
shift=np.array([-1.,0.,0.])
overlaps=[int(np.all((triangle_max>=o+[-1.3,-2.2,.15]-shift) &
                    (triangle_min<=o+[1.3,1.,2.3]-shift),axis=1).sum()) for o in origins]
if any(overlaps):raise RuntimeError(f'Rectangular layout intersects reconstructed obstacles: {overlaps}')
layout=dict(num_envs=len(origins),columns=nx,rows=ny,spacing_m=min(a.spacing_x,a.spacing_y),
    spacing_xy_m=[a.spacing_x,a.spacing_y],env_origins=origins.tolist(),
    background_translation=shift.tolist(),background_yaw_deg=yaw_deg,
    task_local_bounds=[[-1.3,-2.2,.15],[1.3,1.,2.3]],
    occupied_rectangle_m=[float(np.ptp(origins[:,0])+2.6),float(np.ptp(origins[:,1])+3.2)],
    overlapping_triangle_aabbs=overlaps)
(a.output.parent/'layout.json').write_text(json.dumps(layout,indent=2))
# Keep reconstructed obstacles as static collisions. A separate validated plane
# supplies ground contact; omit the noisy scanned floor from contact geometry.
keep=faces[np.all(vertices[faces,2]>.10,axis=1) & np.any(vertices[faces,2]<3.,axis=1)]
used,remap=np.unique(keep,return_inverse=True)
out=Usd.Stage.CreateNew(str(a.output/'wurenyuan_shared.usdc'))
gaussian_stage=Usd.Stage.Open(str(gauss_path))
out.GetRootLayer().customLayerData=gaussian_stage.GetRootLayer().customLayerData
world=UsdGeom.Xform.Define(out,'/Wurenyuan');out.SetDefaultPrim(world.GetPrim())
aligned=UsdGeom.Xform.Define(out,'/Wurenyuan/Aligned')
aligned.AddRotateZOp().Set(yaw_deg)
UsdGeom.SetStageMetersPerUnit(out,1.);UsdGeom.SetStageUpAxis(out,'Z')
gauss=out.DefinePrim('/Wurenyuan/Aligned/Gaussian','Xform');gauss.GetReferences().AddReference(str(gauss_path))
meshroot=out.DefinePrim('/Wurenyuan/Aligned/AlignedMesh','Xform');meshroot.GetReferences().AddReference(str(mesh_path))
UsdGeom.Imageable(meshroot).MakeInvisible()
meshroot.SetActive(False)
proxy=UsdGeom.Mesh.Define(out,'/Wurenyuan/Aligned/ObstacleCollision')
proxy.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices[used].astype(np.float32)))
proxy.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(keep),3,np.int32)))
proxy.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(remap.astype(np.int32)))
proxy.CreateSubdivisionSchemeAttr('none');UsdGeom.Imageable(proxy).MakeInvisible()
UsdPhysics.CollisionAPI.Apply(proxy.GetPrim()).CreateCollisionEnabledAttr(True)
UsdPhysics.MeshCollisionAPI.Apply(proxy.GetPrim()).CreateApproximationAttr('none')
UsdGeom.Xform.Define(out,'/Wurenyuan/Lighting')
dome=UsdLux.DomeLight.Define(out,'/Wurenyuan/Lighting/Dome');dome.CreateIntensityAttr(666.)
key=UsdLux.DistantLight.Define(out,'/Wurenyuan/Lighting/Key');key.CreateIntensityAttr(842.)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-50,0,20))
lights=[dome,key]
for i,origin in enumerate(origins):
 lamp=UsdLux.RectLight.Define(out,f'/Wurenyuan/Lighting/Fill_{i:02d}')
 lamp.CreateWidthAttr(3.);lamp.CreateHeightAttr(3.);lamp.CreateIntensityAttr(925.)
 UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(*(origin-shift+[0,-.5,3.8])))
 lights.append(lamp)
for lamp in lights:
 lamp.CreateEnableColorTemperatureAttr(True);lamp.CreateColorTemperatureAttr(5500.)
 UsdLux.LightAPI(lamp).GetShadowLinkCollectionAPI().CreateIncludeRootAttr(False)
out.GetRootLayer().Save()
report=dict(gaussian_source=str(gauss_path),mesh_source=str(mesh_path),
    gaussian_instances=1,mesh_instances=1,obstacle_triangles=len(keep),shared_yaw_deg=yaw_deg,
    scanned_floor_contact='excluded below 0.10 m; existing task plane provides ground contact',
    collision_height_range_m=[.10,3.],
    alignment='both derived assets preserve wry.usd relative alignment and same leveling transform')
(a.output/'shared_scene.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
