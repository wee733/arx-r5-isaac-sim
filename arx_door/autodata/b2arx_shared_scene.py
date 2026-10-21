"""Shared reconstructed background and per-environment Arena reset/visual terms."""
import math
import numpy as np
from contextlib import contextmanager


@contextmanager
def rectangular_clone_layout(origins):
    """Supply identical rectangular positions to USD roots and the published ClonePlan.

    This pinned Lab version exposes only scalar spacing on InteractiveSceneCfg.
    Adapt its two grid call sites during construction, without changing the vendor tree.
    """
    import importlib
    import torch
    from isaaclab import cloner
    module=importlib.import_module('isaaclab.cloner.clone_plan')
    previous=cloner.grid_transforms, module.grid_transforms
    def grid(N, spacing=1., up_axis='z', device='cpu'):
        if N != len(origins) or up_axis != 'z':
            raise ValueError('Unexpected clone layout request')
        _, orientations=previous[0](N,spacing,up_axis,device)
        return torch.tensor(origins,dtype=torch.float32,device=device),orientations
    cloner.grid_transforms=module.grid_transforms=grid
    try:
        yield
    finally:
        cloner.grid_transforms,module.grid_transforms=previous


def reset_source_poses(env, env_ids, poses):
    """Reset term: distinct verified start poses, translated by each env origin."""
    import torch
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    values=torch.tensor([[*p[:3],0.,0.,math.sin(p[3]/2),math.cos(p[3]/2)] for p in poses],
                        device=env.device,dtype=torch.float32)[ids]
    values[:,:3]+=env.scene.env_origins[ids]
    robot=env.scene['robot']
    robot.write_root_pose_to_sim(values,env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros((len(ids),6),device=env.device),env_ids=ids)


def randomize_panel_appearance(env, env_ids, seed):
    """Startup event: original authored textures with independent bounded UV/material variation."""
    from pxr import UsdShade,Sdf,Gf
    import omni.usd
    from .b2arx_variations import restore_door_material
    stage=omni.usd.get_context().get_stage()
    ids=range(env.num_envs) if env_ids is None else env_ids.tolist()
    records=[]
    for i in ids:
        rng=np.random.default_rng(np.random.SeedSequence([seed,i]))
        spec=dict(env_id=i,albedo_gain=float(rng.uniform(.94,1.04)),
                  roughness_scale=float(rng.uniform(.85,1.15)),uv_scale=float(rng.uniform(.9,1.15)),
                  uv_rotation_deg=float(rng.uniform(-8,8)),uv_translation=rng.uniform(0,.12,2).tolist())
        paths=restore_door_material(stage,spec,f'/World/envs/env_{i}')
        for path in paths:
            if not path.endswith('/UV'):continue
            base=path.rsplit('/',1)[0]
            uv=UsdShade.Shader.Get(stage,path)
            transformed=UsdShade.Shader.Define(stage,base+'/EpisodeUV')
            transformed.CreateIdAttr('UsdTransform2d')
            transformed.CreateInput('in',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
            transformed.CreateInput('scale',Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(spec['uv_scale']))
            transformed.CreateInput('rotation',Sdf.ValueTypeNames.Float).Set(spec['uv_rotation_deg'])
            transformed.CreateInput('translation',Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*spec['uv_translation']))
            transformed.CreateOutput('result',Sdf.ValueTypeNames.Float2)
            for name in ('color','normal'):
                shader=UsdShade.Shader.Get(stage,base+'/'+name)
                shader.GetInput('st').ConnectToSource(transformed.ConnectableAPI(),'result')
                for axis in ('S','T'):shader.CreateInput('wrap'+axis,Sdf.ValueTypeNames.Token).Set('repeat')
        records.append(spec)
    env._shared_scene_appearance=records


def configure_shared_lighting(stage, origins, temperature=5500.):
    """One lighting condition for the entire round; eliminate cloned global lights."""
    from pxr import UsdLux,UsdGeom,Gf
    authored=stage.GetPrimAtPath('/World/SharedReconstruction/Lighting')
    if authored:
        from pxr import Usd
        return dict(temperature_k=temperature,shared_light_paths=[str(p.GetPath()) for p in Usd.PrimRange(authored)
                    if p.HasAPI(UsdLux.LightAPI)],source='authored before first Hydra sync')
    disabled=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight):
            UsdLux.LightAPI(prim).GetIntensityAttr().Set(0.)
            disabled.append(str(prim.GetPath()))
    dome=UsdLux.DomeLight.Define(stage,'/World/SharedLighting/Dome')
    dome.CreateIntensityAttr(300.)
    sun=UsdLux.DistantLight.Define(stage,'/World/SharedLighting/Key')
    sun.CreateIntensityAttr(900.)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50,0,20))
    lights=[dome,sun]
    for i,(x,y,z) in enumerate(origins):
        lamp=UsdLux.RectLight.Define(stage,f'/World/SharedLighting/Ceiling_{i:02d}')
        lamp.CreateWidthAttr(3.);lamp.CreateHeightAttr(3.);lamp.CreateIntensityAttr(900.)
        UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(float(x),float(y)-.5,3.8))
        lights.append(lamp)
    for light in lights:
        light.CreateEnableColorTemperatureAttr(True);light.CreateColorTemperatureAttr(temperature)
        # The captured field already contains scene lighting. Start the visual
        # preview with shadow-free fill; a controlled preview can re-enable links.
        UsdLux.LightAPI(light).GetShadowLinkCollectionAPI().CreateIncludeRootAttr(False)
    return dict(temperature_k=temperature,shared_light_paths=[str(p.GetPath()) for p in lights],
                disabled_cloned_global_lights=disabled,
                background='captured Gaussian radiance; lighting variations illuminate inserted geometry')
