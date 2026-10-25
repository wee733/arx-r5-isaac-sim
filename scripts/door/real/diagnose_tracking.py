#!/usr/bin/env python3
"""Ramp/hold/return with ROS and passive SocketCAN capture; same guarded executor."""
import argparse
import json
import math
import os
import socket
import struct
import time
from pathlib import Path
import numpy as np

import rclpy
from arx5_arm_msg.msg import RobotCmd, RobotStatus
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from smoke import Client
from arx_door.real.contract import from_policy_reply


def profile(t, ramp=2., hold=4.):
    if t < ramp:
        return (1-math.cos(math.pi*max(0,t)/ramp))/2
    if t < ramp+hold:
        return 1.
    if t < 2*ramp+hold:
        return (1+math.cos(math.pi*(t-ramp-hold)/ramp))/2
    return 0.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amplitude',type=float,default=.005)
    parser.add_argument('--axis',type=int,choices=(0,1,2),default=2)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frame',choices=('sdk','model'),default='sdk')
    parser.add_argument('--gripper-metres',type=float,default=0)
    parser.add_argument('--ramp-s',type=float,default=2.)
    parser.add_argument('--hold-s',type=float,default=4.)
    args=parser.parse_args()
    if not 0 < args.amplitude <= .01: parser.error('amplitude must be 0..10 mm')
    if not 0<=args.gripper_metres<=.044:parser.error('gripper must be 0..0.044 m')
    if not 2<=args.ramp_s<=10 or not 4<=args.hold_s<=15:parser.error('ramp must be 2..10 s, hold 4..15 s')
    total=2*(args.ramp_s+args.hold_s)
    def trajectory(t):return profile(t,args.ramp_s,args.hold_s)
    rclpy.init();node=Client();records=[];wire=[];status=None;command=None
    def feedback(m):
        nonlocal status
        status={'q':list(m.joint_pos),'vel':list(m.joint_vel),'current':list(m.joint_cur)}
    def sent(m):
        nonlocal command
        command={'q':list(m.joint_pos),'gripper':m.gripper}
    node.create_subscription(RobotStatus,'/vla_r5/arm_status',feedback,qos_profile_sensor_data)
    node.create_subscription(RobotCmd,'/vla_r5/arm_cmd',sent,1)
    can=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW);can.bind((os.environ.get('VLA_CAN','slcan1'),));can.setblocking(False)
    result={'scope':'hardware_ramp_hold_return','amplitude_m':args.amplitude,'axis':args.axis,'frame':args.frame,
            'gripper_metres':args.gripper_metres,'ramp_s':args.ramp_s,'hold_s':args.hold_s,
            'measurement':'encoder FK; no external metrology'}
    try:
        node.wait(lambda:node.observation and node.status and node.watchdog and status)
        session=node.enable('/vla/enable',True);node.enable('/vla/watchdog/enable',True)
        node.wait(lambda:node.observation['session_id']==session and node.status['enabled'] and node.watchdog['enabled'])
        anchor=(node.observation['sdk_eef'] if args.frame=='sdk' else node.observation['state']['eef'])[:]
        grip_anchor=node.observation['state']['gripper'][0]
        start=time.monotonic();next_send=start;sequence=0;next_record=start
        while time.monotonic()-start < total:
            rclpy.spin_once(node,timeout_sec=.001)
            t=time.monotonic()
            if not node.status['enabled'] or not node.watchdog['enabled']:
                raise RuntimeError(f'control stopped: {node.status}/{node.watchdog}')
            if t>=next_send:
                obs=node.observation;poses=[];grippers=[]
                for i in range(16):
                    p=anchor[:];p[args.axis]+=args.amplitude*trajectory(obs['sample_monotonic_s']+i/30-start);poses.append(p)
                    grippers.append(grip_anchor+(args.gripper_metres-grip_anchor)*trajectory(obs['sample_monotonic_s']+i/30-start))
                sequence+=1
                msg={'schema':'arx_vla_chunk_v1','session_id':session,'observation_id':obs['observation_id'],
                     'sequence':sequence,'semantics':'absolute_xyz_rotvec','frame':'sdk_base_tool',
                     'dt_s':1/30,'control_gripper':False,'eef':poses,'gripper':[0]*16}
                if args.frame=='model':
                    reply={'eef':np.asarray(poses)[None], 'gripper':np.asarray(grippers)[None,:,None]}
                    msg=from_policy_reply(reply,obs,sequence,execute_steps=16,control_gripper=args.gripper_metres>0)
                node.publisher.publish(String(data=json.dumps(msg)));next_send=t+.08
            if t>=next_record:
                records.append({'t':t-start,'target_offset':args.amplitude*trajectory(t-start),
                                'observation':node.observation,'feedback':status,'command':command,'executor':node.status})
                next_record=t+.02
            # Bounded passive receive; never send CAN frames from this diagnostic.
            for _ in range(100):
                try:
                    data=can.recv(16)
                except BlockingIOError:
                    break
                cid,dlc,payload=struct.unpack('=IB3x8s',data)
                wire.append([t-start,cid,dlc,payload.hex()])
        result['complete']=True
    except BaseException as e:
        result['error']=str(e);raise
    finally:
        for service in ('/vla/enable','/vla/watchdog/enable'):
            try:node.enable(service,False)
            except Exception as e:result.setdefault('cleanup_errors',[]).append(str(e))
        result.update(records=records,wire=wire)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','wire')}))
        can.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
