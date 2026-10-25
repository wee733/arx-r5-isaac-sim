#!/usr/bin/env python3
"""Capture a real GR00T prediction with blank RGB, then replay a bounded slow copy.

Diagnostic only: replay uses fresh execution timestamps, not a live policy loop.
Translation <=10 mm, rotation <=0.02 rad, single-finger change <=3 mm.
"""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation, Slerp
from std_msgs.msg import String

from arx_door.real.contract import observation, from_policy_reply
from run_groot import PolicyConnection
from smoke import Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'execute'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path('configs/real/groot.json').read_text())
    connection = None
    if args.mode == 'capture':
        connection = PolicyConnection(config)
        connection.connect()
    rclpy.init(); node = Client(); armed = False; stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    result = {}
    try:
        node.wait(lambda: node.observation and node.status and node.watchdog)
        if node.status['enabled'] or node.watchdog['enabled']:
            raise RuntimeError('Control already enabled; do not take over another controller')
        if args.mode == 'capture':
            state = node.observation
            rgb = {k: np.zeros((480,640,3),np.uint8) for k in ('wrist','ego')}
            reply, elapsed = connection.infer(observation(state, rgb, config['language']))
            from_policy_reply(reply, state, 1, execute_steps=16)
            result = {'scope':'real GR00T prediction from measured joints/EEF plus blank RGB',
                      'checkpoint':config['checkpoint'], 'inference_s':elapsed,
                      'observation':state, 'eef':reply['eef'][0].tolist(),
                      'gripper':reply['gripper'][0,:,0].tolist()}
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            delta = np.asarray(result['eef'])[:,:3]-np.asarray(state['state']['eef'])[:3]
            print(json.dumps({'inference_s':elapsed,'predicted_displacement_m':delta.tolist(),
                              'predicted_gripper_m':result['gripper']}),flush=True)
            return
        captured = json.loads(args.output.read_text())
        anchor = np.asarray(node.observation['state']['eef'])
        previous = np.asarray(captured['observation']['state']['eef'])
        rotation = Rotation.from_rotvec(anchor[3:])
        if np.linalg.norm(anchor[:3]-previous[:3]) > .002 or (rotation.inv()*Rotation.from_rotvec(previous[3:])).magnitude() > .02:
            raise RuntimeError('Robot moved since capture; capture a fresh prediction')
        raw = np.asarray(captured['eef'])
        if raw.shape != (16,6) or not np.isfinite(raw).all():
            raise ValueError('Invalid recorded model output')
        delta = raw[:,:3]-previous[:3]
        xyz_scale = min(1., .010/max(np.linalg.norm(delta,axis=1).max(),1e-12))
        relative_rot = (Rotation.from_rotvec(raw[:,3:])*Rotation.from_rotvec(previous[3:]).inv()).as_rotvec()
        rot_scale = min(1., .020/max(np.linalg.norm(relative_rot,axis=1).max(),1e-12))
        poses = np.concatenate((anchor[:3]+delta*xyz_scale,
                 (Rotation.from_rotvec(relative_rot*rot_scale)*rotation).as_rotvec()),axis=1)
        grip = node.observation['state']['gripper'][0]
        grippers = np.clip(np.asarray(captured['gripper']), max(0.,grip-.003), min(.04,grip+.003))
        knots = np.r_[0.,np.linspace(2,6,16),9.,13.,16.]
        xyz = np.vstack((anchor[:3],poses[:,:3],poses[-1,:3],anchor[:3],anchor[:3]))
        rotations = Slerp(knots,Rotation.from_rotvec(np.vstack((anchor[3:],poses[:,3:],poses[-1,3:],anchor[3:],anchor[3:]))))
        grips = np.r_[grip,grippers,grippers[-1],grip,grip]
        result = {'scope':'bounded time-stretched replay of captured model output; blank RGB; synthetic return to start',
                  'translation_scale':xyz_scale,'rotation_scale':rot_scale,'duration_s':16,
                  'bounded_eef':poses.tolist(),'bounded_gripper':grippers.tolist(),
                  'initial_observation':node.observation,'passed':False}
        if not node.status['execute'] or node.count_publishers('/vla/chunk') != 1:
            raise RuntimeError('Need real executor and exclusive chunk publisher')
        armed = True
        session = node.enable('/vla/enable',True); node.enable('/vla/watchdog/enable',True)
        node.wait(lambda:node.observation['session_id']==session and node.status['enabled'] and node.watchdog['enabled'])
        start = time.monotonic(); next_send=start; seq=0
        print('MOVING: 16 seconds; bounded model trajectory then return and hold.',flush=True)
        while time.monotonic()-start < 16:
            rclpy.spin_once(node,timeout_sec=.005)
            if stopping:raise RuntimeError('interrupted')
            if not node.status['enabled'] or not node.watchdog['enabled']:
                raise RuntimeError(f'control stopped: {node.status}/{node.watchdog}')
            if time.monotonic() >= next_send:
                state=node.observation
                t=np.clip(state['sample_monotonic_s']+np.arange(16)/30-start,0,16)
                eef=np.column_stack([np.interp(t,knots,xyz[:,i]) for i in range(3)]+[rotations(t).as_rotvec()])
                reply={'eef':eef[None],'gripper':np.interp(t,knots,grips)[None,:,None]}
                seq+=1
                msg=from_policy_reply(reply,state,seq,execute_steps=16)
                node.publisher.publish(String(data=json.dumps(msg,allow_nan=False)))
                next_send=time.monotonic()+.08
        result.update(passed=True,final_observation=node.observation,chunks_sent=seq)
    except BaseException as exc:
        result['error']=str(exc)
        raise
    finally:
        if armed:
            for service in ('/vla/enable','/vla/watchdog/enable'):
                try:node.enable(service,False)
                except Exception as exc:result.setdefault('cleanup_errors',[]).append(str(exc))
        if args.mode=='execute':
            result['records']=node.records
            args.output.with_suffix('.execution.json').write_text(json.dumps(result)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('records','bounded_eef','bounded_gripper','initial_observation','final_observation')}),flush=True)
        if connection is not None:connection.close()
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
