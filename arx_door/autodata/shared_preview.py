"""Live, explicit rendering diagnostics for the shared-scene preview."""
def apply(stage, command):
    import carb
    from pxr import UsdGeom, UsdLux, UsdShade, UsdUtils, Sdf, Gf
    settings = carb.settings.get_settings()
    fabric_notices=None
    if command.get('enable_fabric_edits'):
        import usdrt
        from isaaclab.cloner._fabric_notices import get_bindings
        bindings=get_bindings()
        fid=usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().GetId(stage).ToLongInt()).GetFabricId().id
        fabric_notices={'was_enabled':bindings.is_enabled(fid)}
        bindings.set_enable(fid,True)
        fabric_notices['enabled']=bindings.is_enabled(fid)
    if command.get('floor_underlay'):
        if stage.GetPrimAtPath('/World/SharedReconstruction/FloorUnderlay'):
            stage.RemovePrim('/World/SharedReconstruction/FloorUnderlay')
        mesh=UsdGeom.Mesh.Define(stage,'/World/SharedReconstruction/FloorUnderlay')
        # Shared root is translated by -1 m in X. This visual support sits below
        # the reconstructed floor and keeps transparent splats from revealing black.
        floor_z=float(command.get('floor_underlay_z',-.6))
        mesh.CreatePointsAttr([(-16,-10,floor_z),(18,-10,floor_z),(18,9,floor_z),(-16,9,floor_z)])
        mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material.Get(
            stage,'/World/ground/geometry/_shared/dependency_1'))
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('doNotCastShadows',Sdf.ValueTypeNames.Bool).Set(True)
    if command.get('clear_scene_partitions'):
        for prim in stage.Traverse():
            for name in ('primvars:omni:scenePartition','omni:scenePartition'):
                if prim.HasAttribute(name):
                    prim.RemoveProperty(name)
    if command.get('test_local_light'):
        lamp=UsdLux.SphereLight.Define(stage,'/World/DiagnosticLight')
        lamp.CreateIntensityAttr(10000.);lamp.CreateRadiusAttr(.25)
        xf=UsdGeom.Xformable(lamp);xf.ClearXformOpOrder();xf.AddTranslateOp().Set(Gf.Vec3d(6.4,-1.,3.))
    if 'gaussian_asset' in command:
        prim=stage.GetPrimAtPath('/World/SharedReconstruction/Aligned/Gaussian')
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(command['gaussian_asset'])
        from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering
        ok,nurec,spg,problems=setup_for_rendering(stage)
        if not ok or not nurec:
            raise RuntimeError(str(problems))
    if 'crop_max_local_z' in command:
        for prim in stage.Traverse():
            attr=prim.GetAttribute('omni:nurec:crop:maxBounds')
            if attr and attr.Get() is not None:
                value=attr.Get();attr.Set(Gf.Vec3f(value[0],value[1],float(command['crop_max_local_z'])))
    if 'light_exposure' in command:
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/World/SharedLighting') and prim.HasAPI(UsdLux.LightAPI):
                UsdLux.LightAPI(prim).CreateExposureAttr(float(command['light_exposure']))
    if command.get('exclude_reference_mesh'):
        stage.GetPrimAtPath('/World/SharedReconstruction/Aligned/AlignedMesh').SetActive(False)
    if command.get('collision_purpose_guide'):
        for path in ('/World/ground','/World/SharedReconstruction/Aligned/ObstacleCollision'):
            UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreatePurposeAttr('guide')
    camera=UsdGeom.Camera.Get(stage,'/World/ArenaOverview')
    if 'projection' in command:
        camera.CreateProjectionAttr(command['projection'])
    if 'horizontal_aperture' in command:
        camera.CreateHorizontalApertureAttr(float(command['horizontal_aperture']))
        camera.CreateVerticalApertureAttr(float(command['horizontal_aperture'])*1080/1920)
    if command.get('hide_collision_visuals'):
        for prim in stage.Traverse():
            path=str(prim.GetPath())
            if path.endswith('/FloorUnderlay'):
                continue
            if prim.IsA(UsdGeom.Gprim) and (path.startswith('/World/ground') or
                    (path.startswith('/World/SharedReconstruction') and prim.IsA(UsdGeom.Mesh))):
                UsdGeom.Imageable(prim).MakeInvisible()
                for key, value in {'isMatteObject':False,'doNotCastShadows':True,
                                   'invisibleToSecondaryRays':True}.items():
                    UsdGeom.PrimvarsAPI(prim).CreatePrimvar(key,Sdf.ValueTypeNames.Bool).Set(value)
    for key, value in command.get('settings', {}).items():
        if not key.startswith('/rtx/'):
            raise ValueError('Only RTX rendering settings are accepted')
        settings.set(key, value)
    for path, visible in command.get('visibility', {}).items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(path)
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible() if visible else imageable.MakeInvisible()
    for path, value in command.get('light_intensity', {}).items():
        UsdLux.LightAPI(stage.GetPrimAtPath(path)).GetIntensityAttr().Set(float(value))
    if 'light_shadows' in command:
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                UsdLux.LightAPI(prim).GetShadowLinkCollectionAPI().CreateIncludeRootAttr(
                    bool(command['light_shadows']))
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'ParticleField3DGaussianSplat':
            for key, value in command.get('field_primvars', {}).items():
                UsdGeom.PrimvarsAPI(prim).CreatePrimvar(key, Sdf.ValueTypeNames.Bool).Set(bool(value))
    keys = ['/rtx/rendermode', '/rtx/ambientOcclusion/enabled',
            '/rtx/post/tonemap/op', '/rtx/post/histogram/enabled',
            '/rtx/matteObject/enabled', '/rtx/directLighting/enabled']
    report = {key: settings.get(key) for key in keys}
    report['fabric_notices']=fabric_notices
    report['background_world_transform'] = [list(row) for row in UsdGeom.Xformable(
        stage.GetPrimAtPath('/World/SharedReconstruction')).ComputeLocalToWorldTransform(0)]
    report['aligned_world_transform'] = [list(row) for row in UsdGeom.Xformable(
        stage.GetPrimAtPath('/World/SharedReconstruction/Aligned')).ComputeLocalToWorldTransform(0)]
    report['gaussian_fields']=[{'path':str(p.GetPath()),'type':str(p.GetTypeName())} for p in stage.Traverse()
                              if p.GetTypeName() in ('ParticleField3DGaussianSplat','OmniNuRecFieldAsset')]
    return report
