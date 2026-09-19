import json, os, subprocess, time, urllib.request
from pathlib import Path
root=Path('/home/workspace/arx-r5-isaac-sim')
python='/home/lbz/miniforge3/envs/isaaclab/bin/python'
env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None);env['PYTHONPATH']=str(root/'arx_r5_isaac_sim_bringup')
starts=[[0,1,1.5,0,0,0],[.03,1.04,1.48,0,0,0],[-.03,.97,1.52,0,0,0]]
results=[]
for number,start in enumerate(starts,2):
 output=root/f'generated/door_teaching/pilot{number:02d}';output.mkdir(exist_ok=False)
 name=f'candidate_{number:03d}'
 with (output/'simulator.log').open('w') as simlog:
  sim=subprocess.Popen([python,str(root/'scripts/door_teach_session.py'),'--scene',str(root/'generated/door/scene.usd'),'--output',str(output),'--initial-positions',json.dumps(start),'--gripper-force','60','--gripper-stiffness','3000'],cwd=root,env=env,stdout=simlog,stderr=simlog)
  try:
   deadline=time.monotonic()+120
   while 'TEACHING_READY' not in (output/'simulator.log').read_text(errors='replace'):
    if sim.poll() is not None:raise RuntimeError('simulator exited at boot')
    if time.monotonic()>deadline:raise TimeoutError('simulator boot timeout')
    time.sleep(.5)
   print('START',number,flush=True)
   with (output/'collector.log').open('w') as log:
    collected=subprocess.run([python,str(root/'scripts/collect_door_pilot.py'),'--session-output',str(output),'--name',name],cwd=root,env=env,stdout=log,stderr=log)
   results.append({'pilot':number,'collector_exit':collected.returncode})
  finally:
   if sim.poll() is None:
    try:
     req=urllib.request.Request('http://127.0.0.1:8877',data=b'{"op":"shutdown"}',headers={'Content-Type':'application/json'})
     urllib.request.urlopen(req,timeout=10).read()
     sim.wait(timeout=30)
    except Exception:sim.terminate();sim.wait(timeout=15)
  if (output/name/'episode.json').exists():
   subprocess.run([python,str(root/'scripts/encode_door_attempt.py'),str(output/name)],cwd=root,env=env,check=True)
   results[-1]['success']=json.loads((output/name/'episode.json').read_text())['success']
  print('FINISHED',json.dumps(results[-1]),flush=True)
  (root/'generated/door_teaching/batch_results.json').write_text(json.dumps(results,indent=2))
