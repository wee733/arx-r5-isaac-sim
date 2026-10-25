#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""ROS feedback -> official PolicyClient -> guarded B2ARX EEF executor.

RGB topics are an input boundary only; this program never starts camera drivers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from arx_door.policy.config import load_config
from arx_door.real.contract import observation, from_policy_reply


class PolicyConnection:
    """All ZeroMQ operations stay on one worker; inference never blocks ROS."""
    def __init__(self, config):
        self.config = config
        self.client = None

    def connect(self):
        from gr00t.policy.server_client import PolicyClient
        c = self.config
        self.client = PolicyClient(c['host'], c['port'], timeout_ms=c['rpc_timeout_ms'])
        modes = self.client.get_modality_config()
        expected = {'video': ['wrist', 'ego'], 'state': ['eef', 'gripper', 'joints'],
                    'action': ['eef', 'gripper'], 'language': ['annotation.human.task_description']}
        if any(modes[k].modality_keys != keys for k, keys in expected.items()):
            raise ValueError('Policy server is not the B2ARX EEF model')
        if modes['action'].delta_indices != list(range(16)):
            raise ValueError('Policy server must return 16 action steps')
        self.client.reset()

    def infer(self, obs):
        start = time.monotonic()
        reply, _ = self.client.get_action(obs)
        return reply, time.monotonic()-start

    def close(self):
        if self.client is not None:
            # Upstream replaces REQ sockets on timeout. Destroy also closes the
            # abandoned socket, so context termination cannot hang on shutdown.
            self.client.context.destroy(linger=0)
            self.client.close()


class Runner(Node):
    def __init__(self, config, preview):
        super().__init__('arx_groot_runner')
        self.config, self.preview = config, preview
        self.state = None
        self.control, self.watchdog = {}, {}
        self.status_received = {}
        self.images = {}
        self.image_pair = {}
        self.stopping = False
        self.armed = False
        self.session = None
        self.last_report = None
        self.chunk_pub = self.create_publisher(String, '/vla/policy_preview' if preview else '/vla/chunk', 1)
        self.report_pub = self.create_publisher(String, '/vla/policy/status', 1)
        self.create_subscription(String, '/vla/observation', self.on_state, 1)
        self.create_subscription(String, '/vla/status', lambda m: self.on_status('executor', m), 1)
        self.create_subscription(String, '/vla/watchdog/status', lambda m: self.on_status('watchdog', m), 1)
        for key in ('wrist', 'ego'):
            self.create_subscription(Image, config[key+'_topic'],
                                     lambda m, key=key: self.on_image(key, m), qos_profile_sensor_data)
        self.service_clients = {name: self.create_client(SetBool, name)
                        for name in ('/vla/enable', '/vla/watchdog/enable')}
        self.create_service(Trigger, '/vla/policy/stop', self.on_stop)

    def on_stop(self, request, response):
        self.stopping = True
        response.success, response.message = True, 'stopping'
        return response

    def on_state(self, message):
        self.state = json.loads(message.data)

    def on_status(self, key, message):
        setattr(self, 'control' if key == 'executor' else key, json.loads(message.data))
        self.status_received[key] = time.monotonic()

    def on_image(self, key, msg):
        # No cv_bridge / resizing / camera-specific driver dependency.
        if msg.encoding not in ('rgb8', 'bgr8') or (msg.height, msg.width) != (480, 640):
            self.images.pop(key, None)
            return
        if msg.step < 640*3 or len(msg.data) != msg.height*msg.step:
            self.images.pop(key, None)
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        rgb = np.frombuffer(msg.data, np.uint8).reshape(480, msg.step)[:, :1920].reshape(480, 640, 3)
        if msg.encoding == 'bgr8':
            rgb = rgb[:, :, ::-1]
        self.images[key] = (time.monotonic(), stamp, rgb.copy())
        if set(self.images) == {'wrist', 'ego'} and abs(self.images['wrist'][1]-self.images['ego'][1]) <= self.config['image_pair_tolerance_s']:
            # Keep the last matched pair while the next two frames arrive separately.
            self.image_pair = self.images.copy()

    def unavailable(self):
        now = time.monotonic()
        if not self.state or now-self.state['received_monotonic_s'] > 0.2:
            return 'waiting_for_robot_feedback'
        if not self.state.get('calibrated'):
            return 'waiting_for_control_calibration'
        if any(now-self.status_received.get(key, 0) > 0.5 for key in ('executor', 'watchdog')):
            return 'waiting_for_executor_and_watchdog'
        if set(self.image_pair) != {'wrist', 'ego'}:
            return 'waiting_for_wrist_and_ego_RGB'
        wall = self.get_clock().now().nanoseconds*1e-9
        age = self.config['image_max_age_s']
        if any(now-rx > age or not 0 <= wall-stamp <= age for rx, stamp, _ in self.image_pair.values()):
            return 'waiting_for_fresh_RGB'
        return None

    def report(self, state, **extra):
        data = {'state': state, 'preview': self.preview, **extra}
        self.report_pub.publish(String(data=json.dumps(data)))
        if state != self.last_report:
            self.get_logger().info(json.dumps(data))
            self.last_report = state

    def call(self, name, value):
        client = self.service_clients[name]
        if not client.wait_for_service(timeout_sec=1):
            raise RuntimeError(f'Missing service {name}')
        request = SetBool.Request()
        request.data = value
        future = client.call_async(request)
        deadline = time.monotonic()+2
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
        if not future.done():
            raise TimeoutError(name)
        response = future.result()
        if not response.success:
            raise RuntimeError(f'{name}: {response.message}')
        return response.message

    def disable(self):
        for name in self.service_clients:
            try:
                self.call(name, False)
            except Exception as exc:
                self.get_logger().error(str(exc))
        self.armed = False

    def run(self, duration):
        connection = PolicyConnection(self.config)
        pool = ThreadPoolExecutor(max_workers=1)
        pending = pool.submit(connection.connect)
        connected, started = False, time.monotonic()
        captured, sequence, next_request = None, 0, 0.
        try:
            while not self.stopping and (not duration or time.monotonic()-started < duration):
                rclpy.spin_once(self, timeout_sec=0.005)
                now = time.monotonic()
                missing = self.unavailable()
                if self.armed:
                    if missing:
                        raise RuntimeError(missing)
                    if self.state['session_id'] != self.session:
                        raise RuntimeError('executor_session_changed')
                    if not self.control.get('enabled') or not self.watchdog.get('enabled'):
                        raise RuntimeError(f'control_disabled: {self.control.get("state")} / {self.watchdog.get("state")}')
                if pending is not None and pending.done():
                    result = pending.result()
                    pending = None
                    if not connected:
                        connected = True
                    else:
                        reply, elapsed = result
                        sequence += 1
                        chunk = from_policy_reply(reply, captured, sequence,
                                                  execute_steps=self.config['execute_steps'])
                        age = now-captured['sample_monotonic_s']
                        # Do not retimestamp late predictions to make them look fresh.
                        if age >= min(0.3, self.config['execute_steps']/30)-0.03:
                            if not self.preview:
                                raise RuntimeError(f'inference_too_slow: observation age {age:.3f}s')
                            self.report('preview_late_result', inference_s=elapsed, observation_age_s=age)
                        else:
                            self.report('preview' if self.preview else 'running', inference_s=elapsed,
                                        observation_age_s=age, sequence=sequence)
                        self.chunk_pub.publish(String(data=json.dumps(chunk, allow_nan=False)))
                        next_request = captured['sample_monotonic_s']+self.config['replan_steps']/30
                if missing:
                    self.report(missing)
                    continue
                if not connected:
                    self.report('connecting_policy')
                    continue
                if not self.armed and not self.preview:
                    if not self.control.get('execute'):
                        raise RuntimeError('executor is preview-only')
                    if self.control.get('enabled') or self.watchdog.get('enabled'):
                        raise RuntimeError('control is already armed; stop it before starting policy')
                    if self.count_publishers('/vla/chunk') != 1:
                        raise RuntimeError('another chunk publisher is active')
                    # Mark ownership before arming so partial startup is cleaned up.
                    self.armed = True
                    self.session = self.call('/vla/enable', True)
                    self.call('/vla/watchdog/enable', True)
                    deadline = time.monotonic()+1
                    while not (self.state['session_id'] == self.session and self.control.get('enabled') and self.watchdog.get('enabled')):
                        if self.stopping or time.monotonic() > deadline:
                            raise RuntimeError('control_start_interrupted')
                        rclpy.spin_once(self, timeout_sec=0.01)
                    continue
                if pending is None and now >= next_request:
                    captured = self.state
                    rgb = {key: value[2] for key, value in self.image_pair.items()}
                    obs = observation(captured, rgb, self.config['language'])
                    pending = pool.submit(connection.infer, obs)
        finally:
            if self.armed:
                self.disable()
            self.report('stopped')
            pool.submit(connection.close)
            pool.shutdown(wait=True)


def stop_control():
    node = rclpy.create_node('arx_groot_stop')
    try:
        for name, kind in (('/vla/policy/stop', Trigger), ('/vla/enable', SetBool), ('/vla/watchdog/enable', SetBool)):
            client = node.create_client(kind, name)
            if not client.wait_for_service(timeout_sec=1):
                continue
            request = kind.Request()
            if kind is SetBool:
                request.data = False
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=2)
            if not future.done() or not future.result().success:
                raise RuntimeError(f'Failed to stop {name}')
            print(f'{name}: stopped')
    finally:
        node.destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'preview', 'stop'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--language')
    parser.add_argument('--duration', type=float, default=0, help='0 runs until stopped')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.language:
        config['language'] = args.language
    if not 1 <= config['replan_steps'] <= config['execute_steps'] <= 16 or args.duration < 0:
        parser.error('Require 1 <= replan_steps <= execute_steps <= 16 and nonnegative duration')
    rclpy.init()
    node = None
    try:
        if args.mode == 'stop':
            stop_control()
            return
        with open(Path(os.environ.get('XDG_RUNTIME_DIR', '/tmp'))/'arx-groot-runner.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            node = Runner(config, args.mode == 'preview')
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: setattr(node, 'stopping', True))
            node.run(args.duration)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
