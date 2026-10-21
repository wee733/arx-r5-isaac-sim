# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Optional global-camera recording of live, parallel AutoData generation.

Pass this module's register and recorder_cfg callbacks to generate.sh after --.
The camera observes all environments in one scene; it does not alter actions,
resets, physics, or generation acceptance.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path


def register():
    import gymnasium as gym
    from isaaclab.sensors import CameraCfg
    from pxr import Gf
    from . import register as base

    remaining = base.register()
    cfg = base._REGISTERED['cfg']
    # Keep every camera referenced by Arena's observation terms. Use smaller
    # per-environment images for this visualization-only batch.
    for name in ('wrist_camera', 'ego_camera', 'overview_camera'):
        camera = getattr(cfg.scene, name, None)
        if camera is not None:
            camera.width, camera.height = 320, 240
            # RTX normally culls all other envs from a camera's image. This
            # visualization batch needs the complete grid visible in one view.
            camera.renderer_cfg.enable_scene_partitioning = False
            camera.renderer_cfg.global_settings.show_all_partitions_by_default = True
    eye, target = Gf.Vec3d(7.5, -11.0, 9.0), Gf.Vec3d(0.0, 0.0, 1.2)
    matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
    quat = matrix.ExtractRotationQuat()
    overview = cfg.scene.overview_camera
    overview.width, overview.height = 1280, 800
    overview.spawn.focal_length = 40.0
    overview.spawn.horizontal_aperture = 36.0
    overview.spawn.vertical_aperture = 22.5
    overview.spawn.clipping_range = (0.1, 100.0)
    overview.offset = CameraCfg.OffsetCfg(pos=tuple(eye),
        rot=(*tuple(quat.GetImaginary()), float(quat.GetReal())), convention='opengl')
    cfg.viewer.eye, cfg.viewer.lookat = tuple(eye), tuple(target)
    gym.spec(base._REGISTERED['env_id']).kwargs['env_cfg_entry_point'] = cfg
    return remaining


@lru_cache(maxsize=1)
def _classes():
    from isaaclab.managers import RecorderTerm, RecorderTermCfg
    from isaaclab.utils import configclass

    class ArenaOverviewRecorder(RecorderTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.output = Path(env.cfg.recorders.dataset_export_dir_path)
            self.writer = None
            self.events = None
            self.frames = 0
            self.resets = [0] * env.num_envs
            # Kit shutdown can bypass Python finalizers. Finalize the MP4 and
            # telemetry during the explicit env.close() used by AutoData.
            close_environment = env.close

            def close_with_video():
                self.close('')
                close_environment()

            env.close = close_with_video

        def record_post_reset(self, env_ids):
            ids = (range(self._env.num_envs) if env_ids is None else
                   env_ids.tolist() if hasattr(env_ids, 'tolist') else env_ids)
            for i in ids:
                self.resets[i] += 1
            return None, None

        def record_post_step(self):
            import numpy as np
            from arx_door.envs.acceptance import as_numpy
            from arx_door.policy.simulation import VideoWriter
            from arx_door.envs.recorders import _joint_state, DOOR_STATE_NAMES

            env = self._env
            if self.writer is None:
                import torch
                self.output.mkdir(parents=True, exist_ok=True)
                camera = env.scene.sensors['overview_camera']
                # Camera 0 looks over the entire clone grid. Wrist/ego cameras
                # retain their task mounts and observations in every environment.
                camera.set_world_poses(
                    torch.tensor([env.cfg.viewer.eye], device=env.device),
                    torch.tensor([camera.cfg.offset.rot], device=env.device),
                    env_ids=[0], convention='opengl')
                for _ in range(12):
                    env.sim.render()
                self.writer = VideoWriter(self.output / 'arena_parallel_raw.mp4', 1280, 800)
                self.events = (self.output / 'arena_parallel_frames.jsonl').open('w')
                metadata = dict(schema='arx-arena-live-generation-video-v1',
                    kind='global camera during official AutoData generation',
                    num_envs=env.num_envs, fps=30, camera='/World/envs/env_0/OverviewCamera',
                    camera_eye=list(env.cfg.viewer.eye), camera_target=list(env.cfg.viewer.lookat),
                    image_size=[1280, 800], horizontal_aperture_mm=36, focal_length_mm=40,
                    env_origins=as_numpy(env.scene.env_origins).tolist(),
                    wrist_ego_resolution=[320, 240], visualization_only=True,
                    policy_checkpoint_used=False, physics=env.cfg.physics_profile)
                (self.output / 'arena_parallel_video.json').write_text(json.dumps(metadata, indent=2)+'\n')
            # Render without stepping physics or advancing the generation clock.
            env.sim.render()
            camera = env.scene.sensors['overview_camera']
            camera.update(0.0, force_recompute=True)
            rgb = as_numpy(camera.data.output['rgb'])[0, ..., :3]
            if rgb.shape != (800, 1280, 3):
                raise RuntimeError(f'Global camera produced an invalid frame: {rgb.shape}')
            self.writer.append(rgb)
            angles = _joint_state(env, 'door', DOOR_STATE_NAMES)
            self.events.write(json.dumps(dict(frame=self.frames,
                time_s=self.frames / 30, resets=self.resets,
                door_deg=np.rad2deg(angles[:, 0]).tolist(),
                handle_deg=np.rad2deg(angles[:, 1]).tolist()))+'\n')
            self.events.flush()
            self.frames += 1
            if self.frames == 1 or self.frames % 150 == 0:
                self.events.flush()
                from PIL import Image
                Image.fromarray(rgb).save(self.output / 'arena_parallel_preview.jpg')
                print(f'[Arena video] frames={self.frames}, door_deg={np.rad2deg(angles[:, 0]).round(1).tolist()}', flush=True)
            return None, None

        def close(self, file_path):
            if self.writer is not None:
                self.writer.close()
                self.writer = None
                self.events.close()
                (self.output / 'arena_parallel_complete.json').write_text(
                    json.dumps(dict(frames=self.frames, fps=30, duration_s=self.frames/30), indent=2)+'\n')

    @configclass
    class ArenaOverviewRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = ArenaOverviewRecorder

    for cls in (ArenaOverviewRecorder, ArenaOverviewRecorderCfg):
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls
    return ArenaOverviewRecorderCfg


def recorder_cfg():
    from .register import recorder_cfg as base_recorder_cfg
    cfg = base_recorder_cfg()
    cfg.record_arena_global_video = _classes()()
    return cfg


def __getattr__(name):
    if name in ('ArenaOverviewRecorder', 'ArenaOverviewRecorderCfg'):
        _classes()
        return globals()[name]
    raise AttributeError(name)
