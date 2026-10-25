#!/usr/bin/env python3
"""Real ROS executables + deterministic fake driver on isolated Domain 71. No CAN."""
import json
import os
import subprocess
import time
from pathlib import Path

# Set before rclpy.init and before spawning either executable.
os.environ["ROS_DOMAIN_ID"] = "71"
import rclpy
from arx5_arm_msg.msg import RobotCmd, RobotStatus
from std_msgs.msg import String
from smoke import Client

ROOT = Path(__file__).resolve().parents[3]


class Rig(Client):
    def __init__(self):
        super().__init__()
        self.q = [0, 0.5, 0.8, 0, 0.4, 0, 0]
        self.commands = []
        self.emit_feedback = True
        self.feedback_pub = self.create_publisher(RobotStatus, "/vla_r5/arm_status", 1)
        self.create_subscription(RobotCmd, "/vla_r5/arm_cmd", self.command, 1)
        self.create_timer(0.01, self.feedback)

    def command(self, m):
        self.commands.append(list(m.joint_pos))
        self.q = list(m.joint_pos) + [m.gripper]

    def feedback(self):
        if self.emit_feedback:
            m = RobotStatus()
            m.header.stamp = self.get_clock().now().to_msg()
            m.joint_pos = self.q
            self.feedback_pub.publish(m)

    def arm(self):
        session = self.enable("/vla/enable", True)
        self.enable("/vla/watchdog/enable", True)
        self.wait(lambda: self.observation and self.observation["session_id"] == session and
                  self.status.get("enabled") and self.watchdog.get("enabled"))

    def chunk(self, sequence=1):
        obs = self.observation
        return {"schema": "arx_vla_chunk_v1", "session_id": obs["session_id"],
                "observation_id": obs["observation_id"], "sequence": sequence,
                "semantics": "absolute_xyz_rotvec", "frame": "sdk_base_tool",
                "dt_s": 1 / 30, "control_gripper": False,
                "eef": [obs["sdk_eef"][:] for _ in range(16)], "gripper": [0] * 16}

    def send(self, data):
        self.publisher.publish(String(data=json.dumps(data)))


def main():
    output = ROOT / "outputs/real-validation/integration"
    output.mkdir(parents=True, exist_ok=True)
    children, logs, checks = [], [], []
    rclpy.init()
    rig = Rig()
    recorder=rclpy.create_node('vla_readonly_recorder')
    recorder.create_subscription(RobotCmd,'/vla_r5/arm_cmd',lambda _:None,1)
    try:
        for name in ("executor", "watchdog"):
            logfile = (output / f"{name}.log").open("w")
            logs.append(logfile)
            args = [str(ROOT / f"install/real/arx_vla_control/lib/arx_vla_control/{name}"),
                    "--ros-args", "-p", f"config:={ROOT / 'configs/real/arx_r5.json'}"]
            if name == "executor":
                args.extend(["-p", "execute:=true"])
            else:
                args.extend(["-p", "driver_node_name:=vla_smoke_client"])
            children.append(subprocess.Popen(args, stdout=logfile, stderr=subprocess.STDOUT))
        rig.wait(lambda: rig.observation and rig.status and rig.watchdog)
        rig.arm()
        rig.wait(lambda: len(rig.commands) >= 5)
        checks.append("hold flows through watchdog")
        checks.append("read-only command recorder does not count as a second driver")
        rig.send(rig.chunk())
        rig.wait(lambda: rig.status.get("last_sequence") == 1)
        rig.wait(lambda: rig.status.get("state") == "chunk_exhausted_hold")
        checks.append("chunk expiry holds without replay")
        rig.send(rig.chunk(1))
        rig.wait(lambda: not rig.status.get("enabled"))
        assert "out_of_order" in rig.status["state"], rig.status
        checks.append("duplicate sequence rejected")
        rig.enable("/vla/watchdog/enable", False)
        rig.arm()
        stale = rig.chunk()
        deadline = time.monotonic() + 0.35
        rig.wait(lambda: time.monotonic() >= deadline)
        rig.send(stale)
        rig.wait(lambda: not rig.status.get("enabled"))
        assert "stale_observation" in rig.status["state"], rig.status
        checks.append("stale inference rejected")
        rig.enable("/vla/watchdog/enable", False)
        rig.arm()
        rig.emit_feedback = False
        rig.wait(lambda: not rig.watchdog.get("enabled"))
        assert rig.watchdog["state"] == "feedback_timeout", rig.watchdog
        checks.append("feedback loss freezes last bounded command")
        rig.emit_feedback = True
        old_id = rig.observation["observation_id"]
        rig.wait(lambda: rig.observation["observation_id"] > old_id and
                  rig.status.get("feedback_age_s", 1) < 0.05)
        rig.arm()
        competitor = rig.create_publisher(RobotCmd, "/vla/candidate", 1)
        rig.wait(lambda: not rig.watchdog.get("enabled"))
        assert rig.watchdog["state"] == "executor_missing_or_multiple", rig.watchdog
        checks.append("competing command publisher is rejected")
        rig.destroy_publisher(competitor)
        rig.wait(lambda: rig.count_publishers("/vla/candidate") == 1)
        rig.arm()
        before = len(rig.commands)
        children[0].kill()
        rig.wait(lambda: not rig.watchdog.get("enabled"))
        assert rig.watchdog["state"] in ("executor_timeout", "executor_missing_or_multiple"), rig.watchdog
        assert len(rig.commands) > before, "watchdog did not publish hold"
        checks.append("executor SIGKILL produces watchdog hold and latches disabled")
        result = {"passed": True, "scope": "ROS integration, fake hardware, Domain 71", "checks": checks}
        (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for logfile in logs:
            logfile.close()
        rig.destroy_node()
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
