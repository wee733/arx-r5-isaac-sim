#!/usr/bin/env python3
"""Keep a spatial subset of original NuRec Gaussian tensors without changing their values."""
import argparse,gzip,json,zipfile,struct
from pathlib import Path
import msgpack,numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[3]
registration=json.loads((root/'generated/door_autodata/b2arx_gaussian_20261019/scene_center/registration.json').read_text())
with zipfile.ZipFile(root/'assets/wurenyuan/point_cloud.usdz') as archive:
 package=msgpack.unpackb(gzip.decompress(archive.read('point_cloud.nurec')),raw=False,strict_map_key=False)
 data=package['nre_data'];state=data['state_dict'];key='.gaussians_nodes.gaussians.positions'
 positions=np.frombuffer(state[key],dtype=np.float16).reshape(state[key+'.shape']).astype(np.float32)
 transform=np.asarray(registration['field_to_center']);center=positions@transform[:3,:3].T+transform[:3,3]
 theta=np.deg2rad(-13);rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
 world=center@rotation.T+[-1,0,0]
 bounds=np.array([[-17.,-10.,-.30],[17.,9.,3.5]])
 mask=np.all((world>=bounds[0])&(world<=bounds[1]),axis=1);count=int(mask.sum());original=len(mask)
 arrays=[]
 for key in list(state):
  if not key.endswith('.shape'):continue
  shape=state[key]
  if shape and shape[0]==original:
   tensor=key[:-6];values=np.frombuffer(state[tensor],dtype=np.float16).reshape(shape)
   retained=np.ascontiguousarray(values[mask]);state[tensor]=retained.tobytes();state[key]=[count,*shape[1:]];arrays.append(tensor)
 assert len(arrays)==7
 payload=gzip.compress(msgpack.packb(package,use_bin_type=True),compresslevel=3)
 with zipfile.ZipFile(a.output/'point_cloud.usdz','w',compression=zipfile.ZIP_STORED) as target:
  for info in archive.infolist():
   entry=zipfile.ZipInfo(info.filename,info.date_time);entry.compress_type=zipfile.ZIP_STORED
   padding=(-target.fp.tell()-30-len(entry.filename.encode()))%64
   if 0<padding<4:padding+=64
   entry.extra=struct.pack('<HH',0xFFFF,padding-4)+bytes(padding-4) if padding else b''
   target.writestr(entry,payload if info.filename=='point_cloud.nurec' else archive.read(info.filename))
with zipfile.ZipFile(a.output/'point_cloud.usdz') as check:
 for info in check.infolist():assert (info.header_offset+30+len(info.filename.encode())+len(info.extra))%64==0
 # Tensor shapes and byte counts are checked before shipping the native archive.
 for name in arrays:assert len(state[name])==int(np.prod(state[name+'.shape']))*2
report=dict(source_gaussians=original,retained_gaussians=count,world_crop_bounds_m=bounds.tolist(),
 common_yaw_deg=-13,common_translation_m=[-1,0,0],preserved_tensors=arrays,
 method='Subset original FP16 tensors in native NuRec package; no resampling, recoloring, or retraining',
 gaussian_root_to_center=registration['gaussian_root_to_center'])
(a.output/'crop.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
header=(root/'generated/door_autodata/b2arx_gaussian_20261019/scene_center/wurenyuan_gaussian_center.usda').read_text().split('def Xform "Wurenyuan"')[0]
matrix='('+', '.join('('+', '.join(repr(float(v)) for v in row)+')' for row in zip(*registration['gaussian_root_to_center']))+')'
(a.output/'wurenyuan_native_center.usda').write_text(header+'''def Xform "Wurenyuan"
{
    def Xform "Gaussian" (
        prepend references = @./point_cloud.usdz@</World/gauss>
    )
    {
        matrix4d xformOp:transform = '''+matrix+'''
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
''')
