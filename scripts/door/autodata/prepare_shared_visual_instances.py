#!/usr/bin/env python3
"""Instance immutable link visuals; keep articulation/collision and door-panel materials editable."""
import argparse,json
from pathlib import Path
from pxr import Usd,UsdGeom
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
report={}
for name in ('robot','door','workcell'):
 source=(a.source/(name+'.usd')).resolve();stage=Usd.Stage.Open(str(source))
 paths=[p.GetPath() for p in stage.Traverse() if p.GetName() in ('visuals','Visuals') and 'door_panel' not in str(p.GetPath())]
 result=Usd.Stage.CreateNew(str(a.output/(name+'.usda')));root=result.DefinePrim('/Asset','Xform');result.SetDefaultPrim(root);root.GetReferences().AddReference(str(source),'/Asset')
 for path in paths:
  prim=result.OverridePrim(path);prim.GetReferences().AddReference(str(source),str(path));prim.SetInstanceable(True)
 if name=='workcell':
  for light in ('DistantLight','EnvironmentLight'):
   result.OverridePrim('/Asset/'+light).SetActive(False)
 result.GetRootLayer().Save()
 instances=[str(p.GetPath()) for p in result.Traverse() if p.IsInstance()]
 assert len(instances)==len(paths),(name,instances,paths)
 original=sum(len(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
 preserved=sum(len(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []) for p in Usd.PrimRange(result.GetPseudoRoot(),Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh))
 assert original==preserved,(name,original,preserved)
 report[name]=dict(instance_roots=instances,triangles_preserved=original,source=str(source))
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
