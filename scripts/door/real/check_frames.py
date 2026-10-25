#!/usr/bin/env python3
"""Compare native SDK FK with a supplied R5a URDF; never touch CAN or set calibration flags."""
import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def transform(v):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(v[3:]).as_matrix()
    t[:3, 3] = v[:3]
    return t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    tree = ET.parse(args.urdf)
    joints = [tree.find(f".//joint[@name='R5a_joint{i}']") for i in range(1, 7)]
    origins, axes = [], []
    for j in joints:
        origin = j.find("origin")
        t = np.eye(4)
        t[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
        t[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
        origins.append(t)
        axes.append(np.fromstring(j.find("axis").get("xyz"), sep=" "))
    samples = np.random.default_rng(42).uniform(-0.7, 0.7, (40, 6))
    samples[0] = 0
    result = subprocess.run([str(root / "install/real/arx_vla_control/lib/arx_vla_control/kinematics_probe")],
                            input="\n".join(json.dumps(q.tolist()) for q in samples) + "\n",
                            capture_output=True, text=True, check=True)
    native = [transform(json.loads(line)["eef"]) for line in result.stdout.splitlines() if line.startswith("{")]
    if len(native) != len(samples):
        raise RuntimeError(result.stdout + result.stderr)
    model = []
    for q in samples:
        t = np.eye(4)
        for origin, axis, angle in zip(origins, axes, q):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            t = t @ origin @ rotation
        model.append(t)

    def residual(x, indices):
        values = []
        for i in indices:
            difference = np.linalg.inv(native[i]) @ transform(x[:6]) @ model[i] @ transform(x[6:])
            values.extend(difference[:3, 3])
            values.extend(Rotation.from_matrix(difference[:3, :3]).as_rotvec())
        return values

    fit = least_squares(residual, np.zeros(12), args=(range(25),), max_nfev=300)
    errors = np.asarray(residual(fit.x, range(25, 40))).reshape(-1, 6)
    report = {"urdf": str(args.urdf.resolve()), "sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
              "sample_count": 40, "fitted_samples": 25, "held_out_samples": 15,
              "sdk_base_from_model_mount": fit.x[:6].tolist(),
              "model_link6_from_sdk_tool": fit.x[6:].tolist(),
              "held_out_max_position_error_m": float(np.linalg.norm(errors[:, :3], axis=1).max()),
              "held_out_max_rotation_error_rad": float(np.linalg.norm(errors[:, 3:], axis=1).max()),
              "model_zero_eef": model[0].tolist(), "sdk_zero_eef": native[0].tolist(),
              "calibration_applied": False,
              "note": "Numerical FK consistency only. This does not establish that this URDF is the exact training asset or verify hardware joint zero conventions."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
