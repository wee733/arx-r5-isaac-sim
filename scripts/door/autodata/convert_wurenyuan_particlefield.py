"""Convert the user's packed NuRec field with the installed NVIDIA NRE writer.

Run inside nre-ga:26.04.01 with repository at /work and output at /output.
The original coordinate frame and SH coefficients are retained; alignment is
authored as a parent transform, so directional appearance rotates with geometry.
"""
import gzip
import json
import zipfile
from pathlib import Path

import msgpack
import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom
from nre.utils.io.export.gaussian_export_accessor import GaussianAttributes, ModelCapabilities
from nre.utils.io.export.gaussian_usd_writer import GaussianLightFieldWriter

root = Path('/output')
registration = json.loads((root/'scene_center/registration.json').read_text())
with zipfile.ZipFile('/work/assets/wurenyuan/point_cloud.usdz') as archive:
    with archive.open('point_cloud.nurec') as packed:
        with gzip.GzipFile(fileobj=packed) as stream:
            data = msgpack.unpack(stream, raw=False, strict_map_key=False)['nre_data']
config = data['config']['layers']['gaussians']
assert (config['density_activation'], config['scale_activation'], config['rotation_activation']) == ('sigmoid', 'exp', 'normalize')
assert config['precision'] == 16 and config['particle']['radiance_sph_degree'] == 3
assert not config['particle']['density_kernel_planar']
state = data['state_dict']
prefix = '.gaussians_nodes.gaussians.'
def read(name):
    key = prefix+name
    values = np.frombuffer(state[key], dtype=np.float16).reshape(state[key+'.shape']).astype(np.float32)
    assert np.isfinite(values).all(), name
    return values

positions = read('positions')
transform = np.asarray(registration['field_to_center'])
bounds = np.asarray(registration['requested_center_bounds_m'])
center_positions = positions @ transform[:3,:3].T + transform[:3,3]
keep = ((center_positions >= bounds[0]) & (center_positions <= bounds[1])).all(axis=1)
source_count = len(positions)
positions = positions[keep]
rotations = read('rotations')[keep]
norms = np.linalg.norm(rotations, axis=1, keepdims=True)
assert (norms > 0).all()
rotations /= norms
scales = np.exp(read('scales')[keep])
densities = 1/(1+np.exp(-np.clip(read('densities')[keep], -80, 80)))
attributes = GaussianAttributes(positions=positions, rotations=rotations,
    scales=scales, densities=densities, albedo_coefficients=read('features_albedo')[keep],
    specular_coefficients=read('features_specular')[keep])
assert np.isfinite(scales).all() and (scales > 0).all()
capabilities = ModelCapabilities(has_spherical_harmonics=True, has_rigid_tracks=False,
    has_deformation=False, can_deform_positions=False, can_deform_rotations=False,
    can_deform_scales=False, has_temporal_appearance=False, sh_degree=3,
    radiance_sph_O0=config['particle'].get('radiance_sph_O0', False))
output = root/'scene_center/wurenyuan_particlefield_center.usdc'
stage = Usd.Stage.CreateNew(str(output))
world = UsdGeom.Xform.Define(stage, '/World')
stage.SetDefaultPrim(world.GetPrim())
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.)
world.AddTransformOp().Set(Gf.Matrix4d(transform.T.tolist()))
writer = GaussianLightFieldWriter(stage, capabilities, 'wurenyuan', 0,
    half_geometry=False, half_features=True, linear_srgb=False)
prim = writer.create_prim(len(positions))
indices = torch.arange(len(positions), dtype=torch.int64)
writer.write_static_attributes(attributes, indices, do_not_cast_shadows=True, force_sh_0=False)
writer.write_first_frame_attributes(attributes, indices)
writer.finalize([positions])
writer.set_documentation('Original wurenyuan NuRec Gaussian parameters, centered and leveled; exact central crop. No retraining.')
stage.GetRootLayer().Save()
# Verify serialized geometry and full SH3, not just successful file creation.
check = Usd.Stage.Open(str(output))
p = check.GetPrimAtPath(prim.GetPath())
for attr, expected in ((writer.positions_attr, positions), (writer.scales_attr, scales),
                       (writer.opacities_attr, densities.flatten().astype(np.float16))):
    actual = np.asarray(p.GetAttribute(attr.GetName()).Get())
    assert np.array_equal(actual, expected), attr.GetName()
assert p.GetAttribute(writer.sh_degree_attr.GetName()).Get() == 3
sh = np.asarray(p.GetAttribute(writer.sh_coeffs_attr.GetName()).Get())
expected_sh = np.concatenate([attributes.albedo_coefficients[:,None,:], attributes.specular_coefficients.reshape(-1,15,3)],axis=1).reshape(-1,3).astype(np.float16)
assert np.array_equal(sh, expected_sh)
report = dict(source='assets/wurenyuan/point_cloud.usdz', source_nre_version=data['version'],
    writer='NVIDIA NRE GaussianLightFieldWriter', image='nvcr.io/nvidia/nre/nre-ga:26.04.01',
    output=output.name, representation=p.GetTypeName(), source_gaussians=source_count,
    retained_gaussians=len(positions), sh_degree=3, geometry_precision=32, feature_precision=16,
    crop_bounds_m=bounds.tolist(), field_to_center=transform.tolist(),
    activations={'scales':'exp','densities':'sigmoid','rotations':'normalize wxyz'},
    serialized_geometry_and_sh_verified=True, retrained=False)
(root/'scene_center/particlefield_conversion.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
