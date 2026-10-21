"""Episode-fixed visual variation around the user's authored door materials."""
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def restore_door_material(stage, spec, env_root='/World/envs/env_0'):
    """Restore and vary one door's own material namespace, without changing lights."""
    from pxr import Usd,UsdGeom,UsdShade,UsdLux,Gf,Sdf,UsdPhysics
    source=Usd.Stage.Open(str(ROOT/'assets/door/door.usda'))
    restored=[]
    for prim in source.Traverse():
        path=str(prim.GetPath())
        if not prim.IsA(UsdShade.Shader) or not any(n in path for n in ('White_glossy_laminate','Warm_tan_fibreboard_back')):continue
        target=stage.GetPrimAtPath(path.replace('/Door/',env_root+'/Door/',1))
        if not target.IsValid():raise ValueError('missing original door material '+str(target.GetPath()))
        # Restore authored shader inputs AND their connections, including the
        # original texture color space and normal map; do not invent a new tint.
        for attr in prim.GetAttributes():
            if not attr.GetName().startswith('inputs:'):continue
            out=target.CreateAttribute(attr.GetName(),attr.GetTypeName())
            out.ClearConnections()
            value=attr.Get()
            if isinstance(value,Sdf.AssetPath):value=Sdf.AssetPath(value.resolvedPath or str(ROOT/'assets/door'/value.path))
            if value is not None:out.Set(value)
            connections=attr.GetConnections()
            if connections:out.SetConnections([Sdf.Path(str(c).replace('/Door/',env_root+'/Door/',1)) for c in connections])
        if prim.GetName()=='PreviewSurface':
            rough=target.GetAttribute('inputs:roughness');rough.Set(float(rough.Get())*spec.get('roughness_scale',1.))
        if prim.GetName()=='color':
            # Multiplication is neutral at 1.0 and preserves the original texture.
            gain=spec.get('albedo_gain',1.)
            target.CreateAttribute('inputs:scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(gain,gain,gain,1.))
        restored.append(str(target.GetPath()))
    if not restored:raise ValueError('no original panel material restored')
    return restored


def apply_variation(stage, spec):
    from pxr import Usd,UsdGeom,UsdShade,UsdLux,Gf,Sdf,UsdPhysics
    restored=restore_door_material(stage,spec)
    lights=[]
    for p in stage.Traverse():
        if p.IsA(UsdLux.DomeLight) or p.IsA(UsdLux.DistantLight):
            light=UsdLux.LightAPI(p);light.GetColorAttr().Set(Gf.Vec3f(1,1,1))
            light.GetEnableColorTemperatureAttr().Set(True)
            light.GetColorTemperatureAttr().Set(float(spec.get('temperature_k',5500)))
            light.GetIntensityAttr().Set(float(spec.get('dome_intensity',300) if p.IsA(UsdLux.DomeLight) else spec.get('key_intensity',1000)))
            lights.append(str(p.GetPath()))
            if p.IsA(UsdLux.DistantLight):
                xf=UsdGeom.Xformable(p);xf.ClearXformOpOrder();xf.AddRotateXYZOp(opSuffix='variation').Set(Gf.Vec3f(*spec.get('light_angles_deg',[-50,0,20])))
    # This subtree is an explicit visual background. The physical task's ground,
    # contact materials and monitored door/robot collisions are preserved.
    bgpath='/World/AugmentationBackground'
    if stage.GetPrimAtPath(bgpath):stage.RemovePrim(bgpath)
    gaussian_scene=spec.get('scene')=='wurenyuan_gaussian'
    ground=stage.GetPrimAtPath('/World/ground')
    matte_ground=gaussian_scene and spec.get('ground_visual')=='matte'
    if ground:
        if gaussian_scene and not matte_ground:UsdGeom.Imageable(ground).MakeInvisible()
        else:UsdGeom.Imageable(ground).MakeVisible()
        for p in Usd.PrimRange(ground):
            if p.IsA(UsdGeom.Gprim):
                UsdGeom.PrimvarsAPI(p).CreatePrimvar('isMatteObject',Sdf.ValueTypeNames.Bool).Set(matte_ground)
    if spec.get('scene') in ('wurenyuan','wurenyuan_gaussian'):
        bg=UsdGeom.Xform.Define(stage,bgpath)
        meshroot=stage.DefinePrim(bgpath+'/Reconstruction','Xform')
        meshroot.GetReferences().AddReference(str(spec.get('scene_asset',ROOT/'assets/wurenyuan/rec_mesh4_28.usd')))
        mat=np.asarray(spec['scene_transform'],dtype=float)
        bg.AddTransformOp().Set(Gf.Matrix4d(mat.T.tolist()))
        for p in Usd.PrimRange(meshroot):
            if p.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
            if p.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
        UsdGeom.Imageable(meshroot).MakeVisible()
        # Indoor lights below the scanned roof; a distant sun alone is blocked
        # by the retained ceiling and cannot provide useful camera exposure.
        for i,(position,intensity,width) in enumerate([((0.,-1.,3.5),spec.get('indoor_intensity',1200.),3.),((2.,1.,3.),350.,2.)]):
            lamp=UsdLux.RectLight.Define(stage,bgpath+f'/IndoorLight{i}')
            lamp.CreateWidthAttr(width);lamp.CreateHeightAttr(width);lamp.CreateIntensityAttr(float(intensity))
            lamp.CreateEnableColorTemperatureAttr(True);lamp.CreateColorTemperatureAttr(float(spec.get('temperature_k',5500)))
            UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(*position))
            lights.append(str(lamp.GetPath()))
        if gaussian_scene:
            import carb
            from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering
            asset=Usd.Stage.Open(str(spec['scene_asset']))
            settings=carb.settings.get_settings()
            for key,value in asset.GetRootLayer().customLayerData.get('renderSettings',{}).items():
                if key!='rtx:rendermode':settings.set('/'+key.replace(':','/'),value)
            settings.set('/rtx/rendermode','RayTracedLighting')
            if matte_ground:
                settings.set('/rtx/matteObject/enabled',True)
                settings.set('/rtx/post/matteObject/enableShadowCatcher',True)
            ok,nurec,spg,problems=setup_for_rendering(stage)
            if not ok or not nurec:raise RuntimeError('NuRec setup failed: '+str(problems))
            fields=[p for p in Usd.PrimRange(meshroot) if p.GetTypeName()=='OmniNuRecFieldAsset']
            particles=[p for p in Usd.PrimRange(meshroot) if p.GetTypeName()=='ParticleField3DGaussianSplat']
            if not (len(fields)==2 or len(particles)==1):raise RuntimeError('expected native NuRec fields or one ParticleField3DGaussianSplat')
            spec={**spec,'nurec_setup':{'success':ok,'nurec':nurec,'spg':spg,'fields':[str(p.GetPath()) for p in fields+particles]},
                  'gaussian_representation':'ParticleField3DGaussianSplat' if particles else 'legacy NuRec Volume',
                  'ground_visual_mode':'matte shadow catcher' if matte_ground else 'hidden',
                  'background_relighting':'captured radiance; simulated lights affect inserted robot and door'}
    return {**spec,'restored_shader_prims':restored,'lights':lights,
        'reference_material':'assets/door/door.usda and original textures',
        'scene_scope':('native Gaussian radiance field; task-ground collision retained with hidden visual' if gaussian_scene else
                       'reconstructed visual context; task-ground physics remains the validated baseline')}
