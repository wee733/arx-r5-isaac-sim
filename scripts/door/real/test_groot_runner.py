#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""One no-CAN integration check: official RPC + ROS runner + guarded executor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def fake_server(config):
    import numpy as np
    from gr00t.data.types import ModalityConfig
    from gr00t.policy.server_client import PolicyServer

    class Policy:
        calls = 0

        def get_modality_config(self):
            return {key: ModalityConfig(modality_keys=keys, delta_indices=list(range(16)) if key == 'action' else [0])
                    for key, keys in {'video': ['wrist', 'ego'], 'state': ['eef', 'gripper', 'joints'],
                                      'action': ['eef', 'gripper'],
                                      'language': ['annotation.human.task_description']}.items()}

        def reset(self, options=None):
            self.calls = 0
            return {}

        def get_action(self, observation, options=None):
            self.calls += 1
            time.sleep(0.4 if self.calls >= 3 else 0.04)
            return {key: np.repeat(observation['state'][key], 16, axis=1)
                    for key in ('eef', 'gripper')}, {}

    with PolicyServer(Policy(), host='127.0.0.1', port=config['port']) as server:
        server.run()


def main():
    os.environ['ROS_DOMAIN_ID'] = '72'
    config = json.loads((ROOT/'configs/real/groot.json').read_text())
    config['port'] = 15568
    if '--server' in sys.argv:
        fake_server(config)
        return
    import rclpy
    from sensor_msgs.msg import Image
    from integration_test import Rig
    # integration_test sets 71 at import; this test uses 72 for all children.
    os.environ['ROS_DOMAIN_ID'] = '72'
    output = ROOT/'outputs/real-validation/groot_runner'
    output.mkdir(parents=True, exist_ok=True)
    config_path = output/'config.json'
    config_path.write_text(json.dumps(config))
    rclpy.init()
    rig = Rig()
    rig.destroy_publisher(rig.publisher)  # The runner is the sole chunk writer.
    images_enabled = False
    pubs = [rig.create_publisher(Image, config[key+'_topic'], 1) for key in ('wrist', 'ego')]

    def images():
        if images_enabled:
            msg = Image(height=480, width=640, encoding='rgb8', step=1920, data=bytes(480*1920))
            msg.header.stamp = rig.get_clock().now().to_msg()
            for pub in pubs:
                pub.publish(msg)
    rig.create_timer(1/30, images)
    children, logs = [], []

    def start(name, args):
        log = (output/(name+'.log')).open('w')
        logs.append(log)
        child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        return child

    try:
        start('server', [sys.executable, __file__, '--server'])
        for name in ('executor', 'watchdog'):
            args = [str(ROOT/f'install/real/arx_vla_control/lib/arx_vla_control/{name}'),
                    '--ros-args', '-p', f'config:={ROOT / "configs/real/arx_r5.json"}', '-p',
                    'execute:=true' if name == 'executor' else 'driver_node_name:=vla_smoke_client']
            start(name, args)
        rig.wait(lambda: rig.observation and rig.status and rig.watchdog)
        base = [sys.executable, str(ROOT/'scripts/door/real/run_groot.py'), 'run', '--config', str(config_path)]
        missing = start('no_images', base+['--duration', '1'])
        rig.wait(lambda: missing.poll() is not None, timeout=30)
        assert missing.returncode == 0, (output/'no_images.log').read_text()
        assert not rig.commands and not rig.status['enabled'] and not rig.watchdog['enabled']
        images_enabled = True
        run = start('late_reply', base+['--duration', '15'])
        rig.wait(lambda: rig.status.get('last_sequence', 0) >= 2, timeout=30)
        rig.wait(lambda: run.poll() is not None, timeout=15)
        rig.wait(lambda: not rig.status['enabled'] and not rig.watchdog['enabled'])
        log = (output/'late_reply.log').read_text()
        assert run.returncode != 0 and 'inference_too_slow' in log, log
        assert rig.commands
        print('PASS: missing RGB stays disabled; official RPC actions execute; late inference stops both control stages. Domain 72, fake hardware, no CAN.')
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        for log in logs:
            log.close()
        rig.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
