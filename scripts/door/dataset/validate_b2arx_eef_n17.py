#!/usr/bin/env python3
"""Exercise pinned official GR00T loading and SE(3) relative action chunks."""
import argparse,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from validate_door_n17 import write_json,override_gripper_state_range

ROOT=Path(__file__).resolve().parents[3]
PIN='d2b7e75b937e3ec9aa5dbc798f08b89692c49734'

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--groot-root',type=Path,required=True);p.add_argument('--stats-from',type=Path);a=p.parse_args()
    dataset=a.dataset.resolve();meta=dataset/'meta';report_path=meta/'n17_validation.json'
    write_json(report_path,{'official_n1_7_loader_verified':False,'status':'running'})
    commit=subprocess.check_output(['git','-C',str(a.groot_root),'rev-parse','HEAD'],text=True).strip()
    if commit!=PIN:raise ValueError('unreviewed GR00T source')
    if subprocess.check_output(['git','-C',str(a.groot_root),'status','--porcelain','--','gr00t'],text=True):raise ValueError('modified official source')
    sys.path.insert(0,str(a.groot_root.resolve()))
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import RelativeActionLoader,generate_stats,generate_rel_stats
    config_path=ROOT/'configs/dataset/b2arx_eef_n17.py'
    spec=importlib.util.spec_from_file_location('b2arx_eef_config',config_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=module.door_config
    info=json.loads((meta/'info.json').read_text());episodes=[json.loads(x) for x in (meta/'episodes.jsonl').read_text().splitlines()]
    if info['codebase_version']!='v2.1' or info['fps']!=30 or info['total_episodes']!=len(episodes):raise ValueError('invalid metadata')
    if a.stats_from:
        source=a.stats_from.resolve()/'meta';r=json.loads((source/'n17_validation.json').read_text());source_info=json.loads((source/'info.json').read_text())
        if not r['official_n1_7_loader_verified'] or 'train' not in source_info['splits']:raise ValueError('statistics source must be verified train')
        for name in ('stats.json','relative_stats.json'):shutil.copy2(source/name,meta/name)
    else:
        for name in ('stats.json','relative_stats.json'):
            path=meta/name
            if path.exists():path.replace(meta/(name+'.before_official'))
        generate_stats(dataset);generate_rel_stats(dataset,EmbodimentTag.NEW_EMBODIMENT)
        override_gripper_state_range(meta,0.,.044)
    loader=LeRobotEpisodeLoader(dataset,config,decoder_kwargs={'num_ffmpeg_threads':1})
    relative=RelativeActionLoader(dataset,EmbodimentTag.NEW_EMBODIMENT,'eef')
    stats=loader.get_dataset_statistics()
    for category,groups in [('state',{'eef':6,'gripper':1,'joints':6}),('action',{'eef':6,'gripper':1}),('relative_action',{'eef':(16,6)})]:
        for key,size in groups.items():
            for field in ('min','max','mean','std','q01','q99'):
                value=np.asarray(stats[category][key][field]);shape=size if isinstance(size,tuple) else (size,)
                if value.shape!=shape or not np.isfinite(value).all():raise ValueError('invalid official statistics')
    checks=[];offset=0
    for i,ep in enumerate(episodes):
        n=ep['length'];raw=pd.read_parquet(dataset/info['data_path'].format(episode_chunk=0,episode_index=i));data=loader._load_parquet_data(i)
        state=np.stack(raw['observation.state']);action=np.stack(raw['action'])
        if state.shape!=(n,13) or action.shape!=(n,7) or n<16:raise ValueError('invalid numeric dimensions')
        if not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError('nonfinite data')
        for col,expected in [('frame_index',np.arange(n)),('index',np.arange(offset,offset+n)),('episode_index',np.full(n,i))]:np.testing.assert_array_equal(raw[col],expected)
        np.testing.assert_allclose(raw['timestamp'],np.arange(n)/30,rtol=0,atol=1e-5)
        for key,expected in [('state.eef',state[:,:6]),('state.gripper',state[:,6:7]),('state.joints',state[:,7:]),('action.eef',action[:,:6]),('action.gripper',action[:,6:])]:np.testing.assert_array_equal(np.stack(data[key]),expected)
        language='language.'+config['language'].modality_keys[0]
        if any(t not in ep['tasks'] for t in data[language]):raise ValueError('language mismatch')
        chunks=np.asarray(relative.load_relative_actions(i));starts=np.arange(n-15);targets=action[starts[:,None]+np.arange(16)]
        sr=Rotation.from_rotvec(state[starts,3:6]).as_matrix();ar=Rotation.from_rotvec(targets[:,:,3:6].reshape(-1,3)).as_matrix().reshape(-1,16,3,3)
        inv=sr.transpose(0,2,1);xyz=np.einsum('nij,nkj->nki',inv,targets[:,:,:3]-state[starts,None,:3]);rv=Rotation.from_matrix((inv[:,None]@ar).reshape(-1,3,3)).as_rotvec().reshape(-1,16,3)
        np.testing.assert_allclose(chunks,np.concatenate([xyz,rv],axis=2),rtol=0,atol=2e-6)
        frames=np.array([0,n//2,n-1]);videos=loader._load_video_data(i,frames)
        for key in ('wrist','ego'):
            if videos[key].shape!=(3,480,640,3) or min(np.asarray(im).std() for im in videos[key])<1.:raise ValueError('invalid decoded video')
        checks.append(dict(episode_index=i,frames=n,relative_chunks=len(chunks),decoded_frames=frames.tolist()));offset+=n
        print(f'PASS EEF episode {i}: {n} rows, SE(3) chunks and both cameras',flush=True)
    if offset!=info['total_frames']:raise ValueError('frame count mismatch')
    training=dataset/'training';training.mkdir(exist_ok=True);shutil.copy2(config_path,training/config_path.name)
    shutil.copy2(ROOT/'arx_door/autodata/b2arx_pose_codec.py',training/'pose_codec.py')
    report=dict(official_n1_7_loader_verified=True,status='passed',official_commit=commit,episodes=len(episodes),frames=offset,checks=checks,action_contract='XYZ_ROTVEC in current R5a frame plus finger metres; 30 Hz; IK at 120 Hz',statistics_source=str(a.stats_from) if a.stats_from else 'this training split',model_training_executed=False)
    write_json(report_path,report)
    for name in ('export_contract.json','door_export_report.json'):
        target=meta/name;r=json.loads(target.read_text());r['official_n1_7_loader_verified']=True;r['official_loader_verified']=True;write_json(target,r)
    print(json.dumps({k:report[k] for k in ('episodes','frames','status')}))

if __name__=='__main__':main()
