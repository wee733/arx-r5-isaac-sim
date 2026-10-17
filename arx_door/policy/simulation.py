# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Runtime helpers for camera synchronization, official client IPC, and video."""

from __future__ import annotations

import json
import selectors
import subprocess
import time
from pathlib import Path

import numpy as np

from arx_door.policy.config import ROOT, policy_environment, resource


class RemotePolicy:
    """Own an official PolicyClient subprocess without importing GR00T into Kit."""

    def __init__(self, config: dict, config_path: Path, output: Path):
        self.output = output
        self.config = config
        self.log = (output / 'client.log').open('w')
        self.process = subprocess.Popen(
            [str(resource(config, 'policy_python')),
             str(ROOT / 'scripts/door/policy/client_bridge.py'),
             '--config', str(config_path.absolute())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
            bufsize=1, env=policy_environment(config))
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            if not self._response().get('ready'):
                raise RuntimeError('official policy client could not reach its server')
        except BaseException:
            self.close()
            raise

    def _response(self) -> dict:
        if not self.selector.select(timeout=90):
            raise TimeoutError('policy bridge timeout; simulation has not advanced')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'policy client exited; inspect {self.output / "client.log"}')
        return json.loads(line)

    def request(self, request: dict) -> dict:
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        response = self._response()
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'policy client request failed'))
        return response

    def reset(self) -> None:
        self.request({'op': 'reset'})

    def infer(self, rgb, state: np.ndarray, language: str | None = None) -> tuple[np.ndarray, float]:
        """``rgb`` is the wrist frame or a ``{view: frame}`` mapping."""
        incoming = self.output / 'current_observation.npz'
        outgoing = self.output / 'current_action.npy'
        views = rgb if isinstance(rgb, dict) else {'wrist': rgb}
        np.savez(incoming, state=state, **{f'rgb_{name}': frame for name, frame in views.items()})
        tick = time.monotonic()
        self.request({'op': 'infer', 'input': str(incoming), 'output': str(outgoing),
                      'language': language or self.config['language']})
        return np.load(outgoing, allow_pickle=False), time.monotonic() - tick

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        self.selector.close()
        self.log.close()


class VideoWriter:
    """Stream RGB at the actual control sampling rate to a complete MP4."""

    def __init__(self, path: Path, width: int, height: int):
        self.log = path.with_suffix('.ffmpeg.log').open('w')
        self.process = subprocess.Popen(
            ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
             '-video_size', f'{width}x{height}', '-framerate', '30', '-i', '-',
             '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast',
             '-crf', '20', '-pix_fmt', 'yuv420p', str(path)],
            stdin=subprocess.PIPE, stderr=self.log)
        self.frames = 0

    def append(self, rgb: np.ndarray) -> None:
        self.process.stdin.write(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        self.log.close()
        if code:
            raise RuntimeError(f'video encoder exited {code}')


class MountedCamera:
    """Preserve the trained link6 mount and render from actual articulation poses."""

    def __init__(self, env):
        import omni.usd
        from pxr import UsdGeom
        self.env = env
        prim = omni.usd.get_context().get_stage().GetPrimAtPath('/R5a/link6/TeachingWristCamera')
        self.mount = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T.copy()

    def render(self) -> dict:
        import torch
        from scipy.spatial.transform import Rotation

        from arx_door.skillgen.compat import as_torch, quaternion_from_xyzw, quaternion_to_xyzw

        env = self.env
        expected = env.get_robot_eef_pose('link6')[0].cpu().numpy() @ self.mount
        quat = quaternion_from_xyzw(Rotation.from_matrix(expected[:3, :3]).as_quat())
        sensor = env.scene.sensors['wrist_camera']
        sensor.set_world_poses(
            torch.as_tensor(expected[:3, 3][None], dtype=torch.float32, device=env.device),
            torch.as_tensor(quat[None], dtype=torch.float32, device=env.device), convention='opengl')
        env.sim.render()
        sensor.update(0., force_recompute=True)
        env.scene.sensors['overview_camera'].update(0., force_recompute=True)
        if 'ego_camera' in env.scene.sensors:
            env.scene.sensors['ego_camera'].update(0., force_recompute=True)
        position = as_torch(sensor.data.pos_w)[0].cpu().numpy()
        actual = Rotation.from_quat(quaternion_to_xyzw(
            as_torch(sensor.data.quat_w_opengl)[0].cpu().numpy())).as_matrix()
        error = {'position_error_m': float(np.linalg.norm(position - expected[:3, 3])),
                 'rotation_error_rad': float(Rotation.from_matrix(
                     actual.T @ expected[:3, :3]).magnitude())}
        if error['position_error_m'] > 1e-4 or error['rotation_error_rad'] > 1e-3:
            raise RuntimeError(f'live wrist camera mount mismatch: {error}')
        return error

    def rgb(self, name: str) -> np.ndarray:
        from arx_door.skillgen.compat import as_torch
        return as_torch(self.env.scene.sensors[name].data.output['rgb'])[0, ..., :3].cpu().numpy().copy()
