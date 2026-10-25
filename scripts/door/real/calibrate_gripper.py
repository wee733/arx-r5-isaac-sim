#!/usr/bin/env python3
"""Measure SDK command/encoder response; physical opening uses a separate reference."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import rclpy
from std_msgs.msg import String
from smoke import Client


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rclpy.init();n=Client();samples=[];records=[];result={"physical_measurement":False,"opening_reference":"manufacturer 80 mm total; not caliper measured"}
    try:
        n.wait(lambda:n.observation and n.status and n.watchdog)
        session=n.enable('/vla/enable',True);n.enable('/vla/watchdog/enable',True)
        n.wait(lambda:n.observation['session_id']==session and n.status['enabled'] and n.watchdog['enabled'])
        anchor=n.observation['sdk_eef'][:];sequence=0;previous=n.status['gripper_command']
        for target in (0.,1.,2.5,4.,5.,4.,2.5,1.,0.):
            started=time.monotonic();duration=abs(target-previous)/.45+2;next_send=0;settled=[]
            while time.monotonic()-started<duration:
                rclpy.spin_once(n,timeout_sec=.005);t=time.monotonic()
                if not n.status['enabled'] or not n.watchdog['enabled']:raise RuntimeError(f'{n.status}/{n.watchdog}')
                if t>=next_send:
                    obs=n.observation;sequence+=1
                    message={"schema":"arx_vla_chunk_v1","session_id":session,"observation_id":obs['observation_id'],
                             "sequence":sequence,"semantics":"absolute_xyz_rotvec","frame":"sdk_base_tool",
                             "dt_s":1/30,"control_gripper":True,"gripper_units":"sdk_command",
                             "eef":[anchor]*16,"gripper":[target]*16}
                    n.publisher.publish(String(data=json.dumps(message)));next_send=t+.08
                    records.append({"target":target,"elapsed":t-started,"raw":obs['sdk_gripper_raw'],"command":n.status['gripper_command']})
                    if t-started>duration-1: settled.append(obs['sdk_gripper_raw'])
            sample={"command":target,"previous":previous,"raw_median":float(np.median(settled)),
                    "raw_min":min(settled),"raw_max":max(settled)}
            samples.append(sample);print(json.dumps(sample),flush=True);previous=target
        closed=float(np.mean([s['raw_median'] for s in samples if s['command']==0]))
        opened=next(s['raw_median'] for s in samples if s['command']==5)
        if opened-closed<4:raise RuntimeError('gripper response range too small; do not apply calibration')
        response=[[float(np.mean([s['raw_median'] for s in samples if s['command']==c])),c] for c in (0.,1.,2.5,4.,5.)]
        if not all(response[i][0]<response[i+1][0] for i in range(4)):raise RuntimeError('nonmonotonic response')
        result.update(passed=True,feedback_closed=closed,feedback_open=opened,response_raw_command=response,
                      maximum_hysteresis_raw=max(max(s['raw_median'] for s in samples if s['command']==c)-min(s['raw_median'] for s in samples if s['command']==c) for c in (0.,1.,2.5,4.)))
    except BaseException as e:
        result.update(passed=False,error=str(e));raise
    finally:
        for name in ('/vla/enable','/vla/watchdog/enable'):
            try:n.enable(name,False)
            except Exception as e:result.setdefault('cleanup_errors',[]).append(str(e))
        result.update(samples=samples,records=records);a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('samples','records')}))
        n.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
