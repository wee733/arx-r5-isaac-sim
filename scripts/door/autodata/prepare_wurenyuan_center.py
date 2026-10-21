#!/usr/bin/env python3
"""Level the reconstructed floor and crop a central hall without editing originals."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf,Vt
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[3];a.output.mkdir(parents=True,exist_ok=False)
source=Usd.Stage.Open(str(root/'assets/wurenyuan/rec_mesh4_28.usd'))
mesh=UsdGeom.Mesh(next(p for p in source.Traverse() if p.IsA(UsdGeom.Mesh)))
vertices=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64)
counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
if not np.all(counts==3):raise ValueError('expected triangulated source mesh')
reference=Usd.Stage.Open(str(root/'assets/wurenyuan/wry.usd'))
prior=np.asarray(UsdGeom.Xformable(reference.GetPrimAtPath('/World/rec_mesh4_28')).GetLocalTransformation()).T
v=vertices@prior[:3,:3].T+prior[:3,3]
# The authored orientation gives a near-horizontal lower surface. Fit only its
# central floor, exclude overhead structure, walls and distant reconstruction noise.
mask=(v[:,0]>0)&(v[:,0]<40)&(abs(v[:,1])<10)&(v[:,2]>-1.6)&(v[:,2]<-.8)
floor=v[mask][::5];xyz=np.column_stack([floor[:,:2],np.ones(len(floor))]);inliers=np.ones(len(floor),bool)
for _ in range(8):
 coeff=np.linalg.lstsq(xyz[inliers],floor[inliers,2],rcond=None)[0]
 residual=floor[:,2]-xyz@coeff
 inliers=abs(residual)<.025
normal=np.array([-coeff[0],-coeff[1],1.]);normal/=np.linalg.norm(normal)
axis=np.cross(normal,[0,0,1.]);angle=np.arccos(np.clip(normal[2],-1,1))
r=Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle).as_matrix() if angle>1e-10 else np.eye(3)
level=np.eye(4);level[:3,:3]=r;floor_rot=floor[inliers]@r.T;level[2,3]=-np.median(floor_rot[:,2])
transform=level@prior
v=vertices@transform[:3,:3].T+transform[:3,3]
# Central 30 x 24 metre region in the leveled frame; preserve all surface
# elevations, including the original ceiling. Drop boundary-crossing triangles.
bounds=np.array([[5.,-12.,-.2],[35.,12.,12.]])
inside=np.all((v>=bounds[0])&(v<=bounds[1]),axis=1)
faces=indices.reshape(-1,3);kept=faces[np.all(inside[faces],axis=1)]
used,remap=np.unique(kept,return_inverse=True);points=v[used]
# Center the derived asset and put its reconstructed floor 25 mm below the
# validated task ground. This avoids z-fighting without changing robot physics.
center=np.array([20.,0.,.025]);points-=center;transform[:3,3]-=center
out=Usd.Stage.CreateNew(str(a.output/'wurenyuan_center.usdc'));world=UsdGeom.Xform.Define(out,'/Wurenyuan');out.SetDefaultPrim(world.GetPrim());UsdGeom.SetStageMetersPerUnit(out,1.);UsdGeom.SetStageUpAxis(out,'Z')
m=UsdGeom.Mesh.Define(out,'/Wurenyuan/Reconstruction');m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)));m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(kept),3,np.int32)));m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(remap.astype(np.int32)));m.CreateSubdivisionSchemeAttr('none');m.CreateDoubleSidedAttr(True)
mat=UsdShade.Material.Define(out,'/Wurenyuan/Looks/Reconstruction');shader=UsdShade.Shader.Define(out,str(mat.GetPath())+'/Surface');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.55,.55,.55));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.85);mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
out.GetRootLayer().Save()
report=dict(source=str(root/'assets/wurenyuan/rec_mesh4_28.usd'),authored_prior=str(root/'assets/wurenyuan/wry.usd'),floor_fit_points=len(floor),floor_fit_inliers=int(inliers.sum()),residual_tilt_deg=float(np.rad2deg(angle)),source_to_center=transform.tolist(),floor_inlier_rms_m=float(np.sqrt(np.mean(residual[inliers]**2))),crop_in_leveled_frame_m=bounds.tolist(),center_shift_m=center.tolist(),vertices_before=len(vertices),vertices_after=len(points),triangles_before=len(faces),triangles_after=len(kept),scope='central reconstructed geometry; source has no RGB texture or per-vertex colors; neutral material; visual background, validated task plane supplies ground contact')
(a.output/'registration.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
