# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""URDF kinematics for bounded, local Cartesian teaching motions.

This solver does not perform collision checking; use cuMotion for free-space
transfers and simulator contact feedback for slow manipulation segments.
"""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class ArmKinematics:
    """Six-joint ARX chain ending at link6, parsed from the selected URDF."""

    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        self.origins, self.axes, lower, upper = [], [], [], []
        for index in range(1, 7):
            joint = root.find(f"joint[@name='joint{index}']")
            origin = joint.find('origin')
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.get('xyz'), sep=' ')
            transform[:3, :3] = Rotation.from_euler(
                'xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            self.origins.append(transform)
            self.axes.append(np.fromstring(joint.find('axis').get('xyz'), sep=' '))
            lower.append(float(joint.find('limit').get('lower')))
            upper.append(float(joint.find('limit').get('upper')))
        self.lower, self.upper = np.array(lower), np.array(upper)

    def fk(self, joints):
        """Return base_link to link6 homogeneous transform."""
        result = np.eye(4)
        for angle, origin, axis in zip(joints, self.origins, self.axes):
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            result = result @ origin @ motion
        return result

    def ik(self, target, seed, attempts=12):
        """Solve within joint limits, preferring the current branch."""
        target = np.asarray(target, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError('IK target must be a finite 4x4 matrix')
        def residual(q):
            actual = self.fk(q)
            return np.r_[10 * (actual[:3, 3] - target[:3, 3]),
                         Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()]
        rng = np.random.default_rng(0)
        best = None
        for index in range(attempts):
            initial = seed if index == 0 else rng.uniform(self.lower, self.upper)
            result = least_squares(residual, np.clip(initial, self.lower, self.upper),
                                   bounds=(self.lower, self.upper), max_nfev=150)
            position_error = np.linalg.norm(result.fun[:3]) / 10
            angle_error = np.linalg.norm(result.fun[3:])
            if best is None or np.linalg.norm(result.fun) < best[0]:
                best = (np.linalg.norm(result.fun), position_error, angle_error)
            if position_error < 0.0005 and angle_error < 0.005:
                return result.x
        raise ValueError(f'IK failed: position error {best[1]:.6f} m, angle {best[2]:.6f} rad')
