#!/usr/bin/env python3
"""Count all 72 attempts and export only physically accepted, replayed episodes."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--seeds',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[3]
    def local(path):
        s=str(path)
        return root/'generated/door_autodata'/s.removeprefix('/outputs/') if s.startswith('/outputs/') else Path(path)
    attempts=json.loads((a.campaign/'attempts.json').read_text())
    if len(attempts)!=72:raise ValueError(f'expected 72 completed attempts, found {len(attempts)}')
    held={'G04','G05'}
    def split(sid):return 'holdout' if sid in held else 'train'
    selected=[];rejected=[];audits=[];replayed=0
    urdf=ET.parse(root/'assets/b2arx_robot/my_robot.urdf').getroot()
    limits=np.array([[float(j.find('limit').get(k)) for k in ('lower','upper')]
        for name in [f'R5a_joint{i}' for i in range(1,7)] for j in urdf.findall('joint') if j.get('name')==name])
    for r in attempts:
        run=local(r['run']);replay=run.parent/f'replay_{r["env_id"]:02d}'
        result=json.loads((run/'result.json').read_text())
        pr=json.loads((replay/'result.json').read_text()) if (replay/'result.json').exists() else None
        good=result['success'] and pr is not None and pr['success']
        replayed+=int(bool(good))
        audit=dict(run=str(run),passed=False)
        if good:
            with h5py.File(run/'generated.hdf5') as f:
                ep=f['data/demo_0'];n=len(ep['actions'])
                ids=[result['joint_names'].index(f'R5a_joint{i}') for i in range(1,7)]
                q=ep['obs/whole_body_substeps/joint_pos'][:][:,:,ids]
                margin=float(np.minimum(q-limits[:,0],limits[:,1]-q).min())
                singular=float(ep['obs/whole_body/arm_min_singular_value'][:].min())
                alignment=all(len(ep[k])==n for k in ['obs/wrist_rgb','obs/ego_rgb','obs/joint_pos','obs/whole_body_substeps/joint_target'])
                good=margin>=.04 and singular>=.005 and alignment
                audit.update(passed=bool(good),frames=n,min_arm_joint_margin_rad=margin,min_arm_singular_value=singular,synchronized_lengths=alignment)
        audits.append(audit)
        row=dict(run=str(run.resolve()),pose_replay=str(replay.resolve()),source_id=r['source_id'],split=split(r['source_id']),data_kind='skillgen_augmentation')
        if good:selected.append(row)
        else:rejected.append(dict(**r,generation_success=result['success'],replay_success=pr['success'] if pr else None,
            physical=result.get('physical_metrics'),whole_body=result.get('whole_body_acceptance'),
            replay_physical=pr.get('physical_metrics') if pr else None,
            replay_whole_body=pr.get('whole_body_acceptance') if pr else None,
            quality_audit=audit,error=result.get('error')))
    for seed in json.loads(a.seeds.read_text()):
        selected.append(dict(run=str(local(seed['source']).resolve()),pose_replay=str(local(seed['replay_path']).resolve()),
            source_id=seed['id'],split=split(seed['id']),data_kind='expert_seed'))
    a.output.mkdir(exist_ok=False)
    (a.output/'manifest.json').write_text(json.dumps(selected,indent=2)+'\n')
    (a.output/'rejected.json').write_text(json.dumps(rejected,indent=2)+'\n')
    (a.output/'quality_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    summary=dict(attempted_augmentations=72,accepted_generation=sum(r['success'] for r in attempts),
        accepted_independent_replay=replayed,
        accepted_and_replayed_augmentations=len(selected)-10,expert_seeds=10,total_exportable=len(selected),
        train=sum(r['split']=='train' for r in selected),holdout=sum(r['split']=='holdout' for r in selected),
        holdout_source_ids=sorted(held),vla_training_executed=False)
    (a.output/'counts.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
