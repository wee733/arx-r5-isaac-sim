#!/usr/bin/env python3
"""Report accepted physical sources; never infer a seed count from envs or files."""
import argparse,csv,json,os
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--pool-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def host(path):
 path=Path(path)
 if str(path).startswith('/outputs/'):return ROOT/'generated/door_autodata'/path.relative_to('/outputs')
 if str(path).startswith('/workspaces/arx/'):return ROOT/path.relative_to('/workspaces/arx')
 return path
report=json.loads(a.pool_report.read_text())
if not report['passed'] or report['official_pool_count']!=len(report['episodes']):raise ValueError('pool has not accepted all sources')
rows=[];task_roots=set()
for entry in report['episodes']:
 source,replay=host(entry['source']),host(entry['replay'])
 result=json.loads((source/'result.json').read_text());rep=json.loads((replay/'result.json').read_text())
 if not result['success'] or not rep['success']:raise ValueError('source/replay not successful')
 placement=result['placement_metrics'];metrics=result['whole_body_acceptance']['metrics'];physical=result['physical_metrics']
 owner='G' if 'gemini' in str(source).lower() else 'C'
 rows.append(dict(owner=owner,x_m=placement['body_position_m'][0],y_m=placement['body_position_m'][1],
  door_distance_m=placement['signed_distance_closed_plane_m'],mount_handle_distance_m=placement['mount_to_handle_distance_m'],
  yaw_deg=placement['yaw_deg'],standing_height_m=placement['standing_height_m'],frames=result['frames'],
  door_angle_deg=physical['final_door_angle_deg'],max_tilt_deg=metrics['max_tilt_deg'],
  max_drift_m=metrics['max_translation_m'],max_slip_m=metrics['max_foot_slip_m'],max_yaw_drift_deg=metrics['max_yaw_deg'],
  min_foot_ground_force_n=metrics['min_foot_force_n'],max_body_contact_n=metrics['max_body_contact_n'],
  max_leg_contact_n=metrics['max_leg_contact_n'],max_foot_non_ground_contact_n=metrics['max_foot_non_ground_contact_n'],
  max_arm_contact_n=physical['max_arm_contact_n'],min_wrist_clearance_m=physical['min_camera_clearance_m'],
  min_body_camera_clearance_m=entry['raw_audit']['min_body_camera_clearance_m'],
  min_arm_joint_margin_rad=entry['raw_audit']['min_joint_margin_rad'],min_arm_singular_value=entry['raw_audit']['min_jacobian_singular_value'],
  max_arm_speed_rad_s=metrics['max_arm_velocity_rad_s'],physical=True,replay=True,autodata_pool=True,
  source=str(source),replay_path=str(replay),raw_hdf5=str(source/'generated.hdf5'),overview_video=str(source/'overview.mp4')))
 relative=source.relative_to(ROOT/'generated/door_autodata');task_roots.add(ROOT/'generated/door_autodata'/relative.parts[0])
rows.sort(key=lambda r:(r['owner'],r['door_distance_m']));counts=Counter()
for row in rows:counts[row['owner']]+=1;row['id']=f"{row['owner']}{counts[row['owner']]:02}"
with (a.output/'seed_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(a.output/'seed_table.json').write_text(json.dumps(rows,indent=2))
failures=Counter();attempts=[];geometric={};standing=[]
for root in task_roots:
 for file in root.rglob('result.json'):
  d=json.loads(file.read_text())
  if d.get('scope')=='standing_only_not_door_success':standing.append(dict(path=str(file),success=d['success']));continue
  if d.get('schema')!='b2arx-fresh-expert-v1':continue
  kind='reproduction' if d.get('replication_of') else 'acquisition'
  reason='success'
  if not d['success']:
   text=d.get('error','')
   if not text and 'frames' not in d:reason='interrupted_before_result'
   elif 'replicator graph' in text:reason='camera_initialization'
   elif d.get('whole_body_acceptance',{}).get('failures'):reason=','.join(d['whole_body_acceptance']['failures'])
   elif d.get('physical_metrics',{}).get('knob_hold_first_loss_frame') is not None:reason='lost_knob_contact'
   elif 'establish both knob contacts' in text or 'grasp gate' in text:reason='grasp_not_established'
   else:reason=text[:100]
   if reason!='interrupted_before_result':failures[kind+':'+reason]+=1
  attempts.append(dict(path=str(file.parent),kind=kind,success=d['success'],frames=d.get('frames',0),reason=reason,
                       nominal_pose=d.get('pose'),
                       reproduction_mode=('recorded_commands' if 'actuator-command replay' in d.get('provenance','') else 'controller_reexecution') if kind=='reproduction' else None,
                       full_coverage=len(d.get('contact_sensor_bodies',{}).get('leg_contact',[]))==24))
 for file in root.glob('geometric_candidates*.json'):
  for candidate in json.loads(file.read_text()).get('candidates',[]):
   key=tuple(candidate['position'])+(candidate['yaw_deg'],);geometric[key]=candidate['geometric_pass']
stats=dict(selected_usable_sources=len(rows),by_owner=dict(counts),acquisition_attempts=sum(x['kind']=='acquisition' for x in attempts),
 reproduction_attempts=sum(x['kind']=='reproduction' for x in attempts),failure_distribution=dict(failures),
 successful_acquisitions=sum(x['kind']=='acquisition' and x['success'] for x in attempts),
 successful_reproductions=sum(x['kind']=='reproduction' and x['success'] for x in attempts),
 interrupted_attempts=sum(x['reason']=='interrupted_before_result' for x in attempts),
 physically_attempted_placements=len({tuple(x['nominal_pose']) for x in attempts if x['kind']=='acquisition' and x['nominal_pose']}),
 geometric_candidates=len(geometric),geometric_passes=sum(geometric.values()),standing_probes=standing,attempts=attempts,
 official_augmented_episodes=0,vla_training_runs=0)
(a.output/'attempts.json').write_text(json.dumps(stats,indent=2))
lines=[f'# B2＋ARX 开门专家种子：{len(rows)} 条可用源','',
 '每条都来自完整浮动机器人真实物理执行，并在新 reset 后独立执行原始全身子步命令复现。复现不计为新种子。',
 '门、抓握、连续接触及全身稳定性门槛均保持严格；官方 DataGenInfoPool 已逐源加载并核对动作解码和坐标变换。','',
 '| 编号 | 狗身 (x,y) m | 门距 m | 安装座→把手 m | yaw ° | 站高 m | 帧数 | 门角 ° | 倾斜 ° / 漂移 mm / 足端位移 mm | 物理/回放/Pool | 文件 |',
 '|---|---|---:|---:|---:|---:|---:|---:|---|---|---|']
for r in rows:
 rel=os.path.relpath(r['source'],a.output)
 lines.append(f"| {r['id']} | ({r['x_m']:.3f}, {r['y_m']:.3f}) | {r['door_distance_m']:.3f} | {r['mount_handle_distance_m']:.3f} | {r['yaw_deg']:.2f} | {r['standing_height_m']:.3f} | {r['frames']} | {r['door_angle_deg']:.2f} | {r['max_tilt_deg']:.3f} / {r['max_drift_m']*1000:.2f} / {r['max_slip_m']*1000:.2f} | 通过/通过/通过 | [HDF5]({rel}/generated.hdf5) · [录像]({rel}/overview.mp4) · [验收]({rel}/result.json) |")
lines+=['','C = Codex；G = Gemini w6:p19。坐标为站稳后的实测值；门距为狗身参考点到闭门平面 y=-0.3082 m 的有符号距离。足端位移是相对于站稳位置的平面位移指标。',
 '',f"几何候选 {len(geometric)}，几何通过 {sum(geometric.values())}（仅 IK 筛选）；专家采集尝试 {stats['acquisition_attempts']} 次，复现尝试 {stats['reproduction_attempts']} 次，最终选入 {len(rows)} 条源。初始化失败、控制器基线和失败轨迹均保留，详见 [全部尝试](attempts.json)。",'',
 '失败分布：'+json.dumps(dict(failures),ensure_ascii=False)+f"；另外中止 {stats['interrupted_attempts']} 次，不归作物理失败。",'',
 '原始 HDF5 保存两路 RGB、8D 实际位姿动作、120 Hz 全身命令/状态/接触及阶段标签。AutoData 输入中的动作仅做固定初始安装座坐标重表达；原始动作另存，未截断成 7D，也未用下一帧状态替代动作。',
 '本次官方扩增 0 条、VLA 训练 0 次。输入池验证与完整官方生成质量验证分开；后续生成必须使用固定参考系到实时安装座的动态坐标桥接，不能直接沿用旧柱式 generate.sh。',
 '',f"[官方输入池报告]({os.path.relpath(a.pool_report,a.output)}) · [完整数值 CSV](seed_table.csv) · [完整数值 JSON](seed_table.json)"]
(a.output/'DELIVERY.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in stats.items() if k not in ('attempts','standing_probes')},ensure_ascii=False))
