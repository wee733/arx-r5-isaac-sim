#!/usr/bin/env python3
"""B2ARX SkillGen -> LeRobot v2.1 with exact 30 Hz EEF actions and two RGB views."""
import argparse,json,shutil
from pathlib import Path
import h5py,numpy as np,pyarrow as pa,pyarrow.parquet as pq
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scipy.spatial.transform import Rotation
from arx_door.autodata.b2arx_pose_codec import encode,decode
from export_door_lerobot import statistics,write_json,write_jsonl,probe_video
from export_autodata_lerobot import local_path
TASK='Unlock and open the door while holding the knob.'


def validate_run(run):
    result=json.loads((run/'result.json').read_text())
    if not (result['success'] and result['physical_metrics']['accepted'] and result['whole_body_acceptance']['accepted']):
        raise ValueError(f'physical acceptance failed: {run}')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--split',choices=['train','holdout'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads(a.manifest.read_text());sources={}
    for row in rows:
        if row['source_id'] in sources and sources[row['source_id']]!=row['split']:raise ValueError('source leakage across splits')
        sources[row['source_id']]=row['split']
    selected=[r for r in rows if r['split']==a.split]
    if not selected or a.output.exists():raise ValueError('nonempty split and fresh output required')
    out=a.output.with_name(a.output.name+'.incomplete');out.mkdir(parents=True,exist_ok=False);meta=out/'meta';meta.mkdir()
    episodes=[];lineage=[];stats_rows=[];states=[];actions=[];offset=0;features={};texts=None
    for i,row in enumerate(selected):
        run=local_path(row['run']);pose=local_path(row['pose_replay'])
        result=validate_run(run);pr=validate_run(pose)
        expert=row.get('data_kind')=='expert_seed'
        if expert:
            if result.get('schema')!='b2arx-fresh-expert-v1' or 'actuator-command replay' not in pr['provenance']:
                raise ValueError('expert seed requires its independent full-body command replay')
        else:
            if 'generation' not in result or result['generation']['algorithm']!='official AutoData SkillGen':raise ValueError('requires official generation provenance')
            if 'pose-action replay' not in pr['provenance'] or pr.get('pose_codec')!='XYZ_ROTVEC float32 -> quaternion_xyzw float32':raise ValueError('requires decoded pose-action replay')
        if run.resolve()==pose.resolve():raise ValueError('replay is not independent')
        lang=json.loads((run/'language.json').read_text())
        if texts is None:texts=lang['instructions']
        if texts!=lang['instructions']:raise ValueError('language vocabulary differs')
        with h5py.File(run/'generated.hdf5') as f,h5py.File(pose/'generated.hdf5') as pf:
            ep=f['data/demo_0'];pe=pf['data/demo_0'];n=len(ep['actions'])
            camera=result['camera_evidence']
            if not camera['capture_protocol_verified'] or camera['frames_verified']!=n:
                raise ValueError('generation lacks synchronized RGB evidence')
            joints=ep['obs/joint_pos'][:];eef=ep['obs/datagen_info/eef_pose/link6'][:]
            state=np.concatenate([eef[:,:3,3],Rotation.from_matrix(eef[:,:3,:3]).as_rotvec(),joints[:,6:7],joints[:,:6]],axis=1).astype('float32')
            action=encode(ep['actions'][:]);decoded=decode(action)
            if state.shape!=(n,13) or action.shape!=(n,7) or not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError('bad numeric contract')
            if expert:
                np.testing.assert_allclose(ep['obs/whole_body_substeps/joint_target'][:],pe['obs/whole_body_substeps/joint_target'][:],atol=1e-6,rtol=0)
            for actual in ((ep['actions'][:],) if expert else (ep['actions'][:],pe['actions'][:])):
                np.testing.assert_allclose(actual[:,[0,1,2,7]],decoded[:,[0,1,2,7]],atol=1e-6,rtol=0)
                np.testing.assert_allclose(Rotation.from_quat(actual[:,3:7]).as_matrix(),Rotation.from_quat(decoded[:,3:7]).as_matrix(),atol=1e-6,rtol=0)
            if pr['frames']!=n:raise ValueError('incomplete replay')
            if action[:,6].min()<0 or action[:,6].max()>.044001:raise ValueError('wrong gripper units')
            np.testing.assert_allclose(ep['obs/diagnostics/timestamp'][:].reshape(-1),np.arange(n)/30,atol=1e-5)
            labels=np.asarray(lang['phase_ids'],dtype=np.int64)
            np.testing.assert_array_equal(labels,ep['obs/diagnostics/phase_id'][:].reshape(-1))
            for camera in ('wrist','ego'):
                if ep[f'obs/{camera}_rgb'].shape!=(n,480,640,3):raise ValueError('camera mismatch')
                for frame in sorted({0,n//2,n-1}):
                    if ep[f'obs/{camera}_rgb'][frame].std()<1.:raise ValueError('empty sampled training RGB')
                path=out/f'videos/chunk-000/observation.images.{camera}/episode_{i:06d}.mp4';path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(run/f'{camera}.mp4',path);probe=probe_video(path)
                if int(probe['nb_read_frames'])!=n or (probe['width'],probe['height'])!=(640,480) or probe['avg_frame_rate']!='30/1':raise ValueError('video frame/rate/size mismatch')
                features[f'observation.images.{camera}']={'dtype':'video','shape':[480,640,3],'names':['height','width','channels'],
                    'info':{'video.height':480,'video.width':640,'video.codec':'h264','video.pix_fmt':'yuv420p','video.fps':30,'video.channels':3,'video.is_depth_map':False,'has_audio':False}}
            numeric={'observation.state':state,'action':action};columns={k:pa.array(v.tolist(),type=pa.list_(pa.float32(),v.shape[1])) for k,v in numeric.items()}
            task_ids=np.full(n,len(texts),dtype=np.int64)
            columns.update(timestamp=pa.array(np.arange(n)/30,type=pa.float32()),frame_index=pa.array(np.arange(n)),episode_index=pa.array(np.full(n,i)),index=pa.array(np.arange(offset,offset+n)),task_index=pa.array(task_ids),**{'annotation.human.stage_description':pa.array(labels),'annotation.human.task_description':pa.array(task_ids)})
            dest=out/f'data/chunk-000/episode_{i:06d}.parquet';dest.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.table(columns),dest)
            delta=pe['obs/joint_pos'][:]-joints
            lineage.append(dict(episode_index=i,**row,generation=result.get('generation',{'algorithm':'fresh expert controller'}),variation=result.get('variation'),
                replay_contract='full-body actuator commands' if expert else 'decoded EEF pose actions',
                controller_comparison={'rmse':np.sqrt(np.mean(delta**2,axis=0)).tolist(),'max_abs':np.max(abs(delta),axis=0).tolist()},
                physical_metrics=result['physical_metrics'],whole_body_acceptance=result['whole_body_acceptance']))
            episodes.append({'episode_index':i,'tasks':[TASK]+[texts[j] for j in sorted(set(labels))],'length':n})
            stats_rows.append({'episode_index':i,'stats':{k:statistics(v) for k,v in numeric.items()}})
            states.append(state);actions.append(action);offset+=n
    pose_names=['eef_x_m','eef_y_m','eef_z_m','eef_rx_rad','eef_ry_rad','eef_rz_rad','finger_m']
    for k,names in [('observation.state',pose_names+[f'R5a_joint{i}_rad' for i in range(1,7)]),('action',pose_names)]:features[k]={'dtype':'float32','shape':[len(names)],'names':names}
    texts=texts+[TASK]
    for k in ('timestamp','frame_index','episode_index','index','task_index','annotation.human.stage_description','annotation.human.task_description'):features[k]={'dtype':'float32' if k=='timestamp' else 'int64','shape':[1],'names':None}
    write_json(meta/'info.json',dict(codebase_version='v2.1',robot_type='b2arx_standing',fps=30,total_episodes=len(episodes),total_frames=offset,total_tasks=len(texts),total_videos=len(episodes)*2,total_chunks=1,chunks_size=1000,splits={a.split:f'0:{len(episodes)}'},features=features,data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'))
    write_json(meta/'modality.json',{'state':{'eef':{'start':0,'end':6},'gripper':{'start':6,'end':7},'joints':{'start':7,'end':13}},'action':{'eef':{'start':0,'end':6},'gripper':{'start':6,'end':7}},'video':{k:{'original_key':f'observation.images.{k}'} for k in ('wrist','ego')},'annotation':{f'human.{kind}_description':{'original_key':f'annotation.human.{kind}_description'} for kind in ('stage','task')}})
    for name,data in [('episodes',episodes),('tasks',[{'task_index':i,'task':t} for i,t in enumerate(texts)]),('episodes_stats',stats_rows),('autodata_lineage',lineage)]:write_jsonl(meta/(name+'.jsonl'),data)
    write_json(meta/'stats.json',{k:statistics(np.concatenate(v)) for k,v in [('observation.state',states),('action',actions)]})
    write_json(meta/'export_contract.json',dict(source='official_autodata_skillgen_and_expert_seeds' if any(r.get('data_kind')=='expert_seed' for r in selected) else 'official_autodata_skillgen',split=a.split,action='original EEF target; float32 quaternion/rotation-vector round trip checked to 1e-6; generated episodes independently replay decoded EEF actions, expert seeds independently replay full-body actuator commands',action_reference_frame='current R5a installation link',controller='MountFrameIK at 120 Hz; target updates at 30 Hz; mirrored gripper',units=['m']*3+['rad']*3+['m'],alignment='pre-action RGB and state; post-action diagnostics',leg_controller='independent fixed standing PD; not learned locomotion',source_groups=sources,official_loader_verified=False,vla_training_executed=False))
    write_json(meta/'door_export_report.json',dict(schema='b2arx-autodata-lerobot-export-v1',episodes=len(episodes),frames=offset,physical_acceptance_and_independent_replays_verified=True,official_n1_7_loader_verified=False,eligible_for_training=True,training_effectiveness_evaluated=False))
    out.rename(a.output);print(a.output)
if __name__=='__main__':main()
