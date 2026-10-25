#!/usr/bin/env python3
"""Bounded hardware smoke test through the same chunk/executor/watchdog path.

Default: hold measured joints only. --motion-m enables a <=10 mm SDK-frame
out-and-back sinusoid. Optional gripper motion uses explicit SDK units. No cameras/model.
"""
import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool


class Client(Node):
    def __init__(self):
        super().__init__("vla_smoke_client")
        self.observation = None
        self.status = {}
        self.watchdog = {}
        self.records = []
        self.last_observation_record = 0
        self.create_subscription(String, "/vla/observation", self.on_observation, 1)
        self.create_subscription(String, "/vla/status", self.on_status, 1)
        self.create_subscription(String, "/vla/watchdog/status", self.on_watchdog, 1)
        self.publisher = self.create_publisher(String, "/vla/chunk", 1)
        self.service_clients = {name: self.create_client(SetBool, name)
                         for name in ("/vla/enable", "/vla/watchdog/enable")}

    def on_observation(self, m):
        self.observation = json.loads(m.data)
        if time.monotonic() - self.last_observation_record > 0.05:
            self.last_observation_record = time.monotonic()
            self.records.append({"time": time.monotonic(), "observation": self.observation})

    def on_status(self, m):
        self.status = json.loads(m.data)
        self.records.append({"time": time.monotonic(), "executor": self.status})

    def on_watchdog(self, m):
        self.watchdog = json.loads(m.data)
        self.records.append({"time": time.monotonic(), "watchdog": self.watchdog})

    def wait(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError("ROS condition timed out")
            rclpy.spin_once(self, timeout_sec=0.02)

    def enable(self, name, value):
        client = self.service_clients[name]
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError(f"missing service {name}")
        request = SetBool.Request()
        request.data = value
        future = client.call_async(request)
        self.wait(future.done)
        result = future.result()
        if not result.success:
            raise RuntimeError(f"{name}: {result.message}")
        return result.message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-m", type=float, default=0)
    parser.add_argument("--gripper-delta", type=float, default=0,
                        help="SDK command out-and-back offset, 0..0.2; physical metre calibration is not implied")
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--duration", type=float, default=4)
    parser.add_argument("--position-tolerance-m", type=float, default=0.003,
                        help="task-level peak/return tolerance; default 3 mm, not a GR00T requirement")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.motion_m <= 0.01 or not 2 <= args.duration <= 15 or not 0 <= args.gripper_delta <= 0.2:
        parser.error("motion must be 0..0.01 m, duration 2..15 s, gripper delta 0..0.2 SDK units")
    if not 0.0005<=args.position_tolerance_m<=0.005:
        parser.error("position tolerance must be 0.5..5 mm")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Client()
    receipt = {"scope": "hardware_control_only", "motion_m": args.motion_m,
               "gripper_delta_sdk": args.gripper_delta,
               "position_tolerance_m": args.position_tolerance_m,
               "axis": args.axis, "camera_or_model_used": False, "passed": False}
    try:
        node.wait(lambda: node.observation is not None and bool(node.status) and bool(node.watchdog))
        if not node.status["execute"]:
            raise RuntimeError("this is a real-control smoke test, but executor is preview-only")
        session = node.enable("/vla/enable", True)
        node.enable("/vla/watchdog/enable", True)
        node.wait(lambda: node.observation["session_id"] == session and
                  node.status.get("enabled") and node.watchdog.get("enabled"))
        anchor = node.observation["sdk_eef"][:]
        grip_anchor = node.status["gripper_command"]
        if grip_anchor + args.gripper_delta > 5:
            raise ValueError("gripper test would exceed command range")
        receipt["initial_observation"] = node.observation
        start = time.monotonic()
        next_send, sequence = start, 0
        while time.monotonic() - start < args.duration + 0.6:
            rclpy.spin_once(node, timeout_sec=0.005)
            if not node.status["enabled"] or not node.watchdog["enabled"]:
                raise RuntimeError(f"control stopped: {node.status} / {node.watchdog}")
            if (args.motion_m or args.gripper_delta) and time.monotonic() >= next_send:
                obs = node.observation
                poses = []
                grippers = []
                for i in range(16):
                    elapsed = obs["sample_monotonic_s"] + i / 30 - start
                    phase = min(1, max(0, elapsed / args.duration))
                    target = anchor[:]
                    target[args.axis] += args.motion_m * (1 - math.cos(2 * math.pi * phase)) / 2
                    poses.append(target)
                    grippers.append(grip_anchor + args.gripper_delta * (1 - math.cos(2 * math.pi * phase)) / 2)
                sequence += 1
                message = {"schema": "arx_vla_chunk_v1", "session_id": session,
                           "observation_id": obs["observation_id"], "sequence": sequence,
                           "semantics": "absolute_xyz_rotvec", "frame": "sdk_base_tool",
                           "dt_s": 1 / 30, "control_gripper": args.gripper_delta > 0,
                           "gripper_units": "sdk_command", "eef": poses, "gripper": grippers}
                node.publisher.publish(String(data=json.dumps(message, allow_nan=False)))
                next_send = time.monotonic() + 0.08
        receipt["control_path_passed"] = True
        measured = [r["observation"] for r in node.records
                    if "observation" in r and r["time"] >= start]
        if args.motion_m:
            peak = max(o["sdk_eef"][args.axis] - anchor[args.axis] for o in measured)
            final_error = abs(node.observation["sdk_eef"][args.axis] - anchor[args.axis])
            receipt.update(measured_peak_displacement_m=peak, return_error_m=final_error)
            receipt["tracking_passed"] = (abs(peak - args.motion_m) <= args.position_tolerance_m and final_error <= args.position_tolerance_m)
            if peak < args.motion_m * 0.25 or peak > args.motion_m + args.position_tolerance_m or final_error > args.position_tolerance_m:
                raise RuntimeError("measured motion did not follow the bounded test trajectory")
        if args.gripper_delta:
            values = [o["sdk_gripper_raw"] for o in measured]
            receipt["gripper_feedback_range"] = [min(values), max(values)]
            if max(values) - min(values) < args.gripper_delta * 0.25:
                raise RuntimeError("gripper feedback did not move enough to validate execution")
        receipt.update(passed=True, final_observation=node.observation,
                       final_executor=node.status, final_watchdog=node.watchdog,
                       chunks_sent=sequence)
        # A functioning command path is not proof of millimetre trajectory tracking.
        if args.motion_m and not receipt["tracking_passed"]:
            receipt["passed"] = False
            raise RuntimeError("control path responded, but configured motion tracking/return tolerance failed")
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup_errors = []
        for service in ("/vla/enable", "/vla/watchdog/enable"):
            try:
                node.enable(service, False)
            except Exception as exc:
                cleanup_errors.append(f"{service}: {exc}")
        receipt["cleanup_errors"] = cleanup_errors
        receipt["records"] = node.records
        if cleanup_errors:
            receipt["passed"] = False
        args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k not in
                          ("records", "initial_observation", "final_observation")}, ensure_ascii=False))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
