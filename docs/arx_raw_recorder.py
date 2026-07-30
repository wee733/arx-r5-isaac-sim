#!/usr/bin/env python3
"""Portable two-camera ARX raw-v1 recorder for integration on the 5090 host.

This module deliberately has no Isaac Sim dependency. Call ``append`` from the
Isaac/Isaac Lab step where you have the current RGB observations, current 7-D
feedback state, and the final 7-D absolute command target that will be applied
to the articulation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import numpy as np


JOINT_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
UNITS = ["rad"] * 6 + ["m"]
CAMERAS = ("front", "wrist")
H264_ENCODERS = ("libx264", "libopenh264")
RESERVED_EPISODE_METADATA = {
    "episode_index",
    "task",
    "success",
    "length",
    "notes",
    "timestamp_policy",
    "source_timestamp_origin",
    "video_encoder",
}


class ArxRawRecorder:
    def __init__(
        self,
        root: str | Path,
        *,
        fps: int = 30,
        action_source: str = "isaac_joint_commands",
        ffmpeg: str = "ffmpeg",
        encoder: str = "auto",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.fps = int(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        self.action_source = action_source
        self.ffmpeg = ffmpeg
        self.encoder = self._resolve_encoder(encoder)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "episodes").mkdir(exist_ok=True)
        self._write_or_check_dataset_json()
        self._active: dict[str, Any] | None = None

    def _write_or_check_dataset_json(self) -> None:
        metadata = {
            "schema_version": "arx-raw-v1",
            "robot_type": "arx_r5",
            "fps": self.fps,
            "joint_names": JOINT_NAMES,
            "state_units": UNITS,
            "action_units": UNITS,
            "action_source": self.action_source,
            "action_semantics": "absolute_command_target",
            "camera_color_order": "RGB",
            "camera_keys": list(CAMERAS),
            "video_codec": "h264",
            "video_encoder": self.encoder,
            "video_pix_fmt": "yuv420p",
        }
        path = self.root / "dataset.json"
        if path.exists():
            current = json.loads(path.read_text())
            if current != metadata:
                raise ValueError(f"existing {path} does not match this recorder configuration")
            return
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(tmp, path)

    def _resolve_encoder(self, requested: str) -> str:
        if requested != "auto" and requested not in H264_ENCODERS:
            raise ValueError(
                f"encoder must be 'auto' or an H.264 encoder in {H264_ENCODERS}, "
                f"got {requested!r}"
            )
        try:
            result = subprocess.run(
                [self.ffmpeg, "-hide_banner", "-encoders"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError as exc:
            raise ValueError(f"ffmpeg executable was not found: {self.ffmpeg}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"ffmpeg encoder probe timed out: {self.ffmpeg}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "unknown ffmpeg error").strip()
            raise ValueError(f"cannot query ffmpeg encoders: {detail}") from exc
        available = {
            match.group(1)
            for line in (result.stdout + "\n" + result.stderr).splitlines()
            if (match := re.match(r"^\s*[A-Z.]{6}\s+(\S+)", line))
        }
        candidates = H264_ENCODERS if requested == "auto" else (requested,)
        for encoder in candidates:
            if encoder in available:
                return encoder
        raise ValueError(
            f"{self.ffmpeg} has no supported H.264 encoder; install FFmpeg with "
            f"one of {H264_ENCODERS}"
        )

    def _next_episode_index(self) -> int:
        indices = []
        for path in (self.root / "episodes").glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"):
            if path.is_dir():
                indices.append(int(path.name.removeprefix("episode_")))
        return max(indices, default=-1) + 1

    def start_episode(self, task: str, *, metadata: dict[str, Any] | None = None) -> int:
        if self._active is not None:
            raise RuntimeError("an episode is already active")
        if not task.strip():
            raise ValueError("task must be a non-empty natural-language instruction")
        episode_metadata = dict(metadata or {})
        overlap = RESERVED_EPISODE_METADATA.intersection(episode_metadata)
        if overlap:
            raise ValueError(
                f"metadata cannot override reserved episode fields: {sorted(overlap)}"
            )
        index = self._next_episode_index()
        final_dir = self.root / "episodes" / f"episode_{index:06d}"
        temp_dir = final_dir.with_name(f".{final_dir.name}.recording")
        if final_dir.exists() or temp_dir.exists():
            raise FileExistsError(final_dir)
        temp_dir.mkdir()
        self._active = {
            "index": index,
            "task": task.strip(),
            "metadata": episode_metadata,
            "temp_dir": temp_dir,
            "final_dir": final_dir,
            "shapes": {},
            "processes": {},
            "timestamp": [],
            "front_timestamp": [],
            "wrist_timestamp": [],
            "state": [],
            "action": [],
            "source_timestamp_origin": None,
        }
        return index

    def _start_encoder(self, camera: str, height: int, width: int) -> subprocess.Popen:
        assert self._active is not None
        output = self._active["temp_dir"] / f"{camera}.mp4"
        codec_args = ["-c:v", self.encoder]
        if self.encoder == "libx264":
            codec_args += ["-preset", "veryfast", "-crf", "18"]
        elif self.encoder == "libopenh264":
            codec_args += ["-b:v", "8M"]
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            *codec_args,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE)

    @staticmethod
    def _rgb(frame: np.ndarray, label: str) -> np.ndarray:
        value = np.asarray(frame)
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"{label} must be HWC RGB with 3 channels, got {value.shape}")
        if value.dtype != np.uint8:
            raise ValueError(f"{label} must be uint8 RGB 0..255, got {value.dtype}")
        return np.ascontiguousarray(value)

    def append(
        self,
        *,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        timestamp: float | None = None,
        front_timestamp: float | None = None,
        wrist_timestamp: float | None = None,
    ) -> None:
        if self._active is None:
            raise RuntimeError("call start_episode before append")
        front = self._rgb(front_rgb, "front_rgb")
        wrist = self._rgb(wrist_rgb, "wrist_rgb")
        for camera, frame in (("front", front), ("wrist", wrist)):
            shape = frame.shape
            if camera not in self._active["shapes"]:
                self._active["shapes"][camera] = shape
                self._active["processes"][camera] = self._start_encoder(
                    camera, shape[0], shape[1]
                )
            elif self._active["shapes"][camera] != shape:
                raise ValueError(f"{camera} resolution changed within an episode")

        state_value = np.asarray(state, dtype=np.float32)
        action_value = np.asarray(action, dtype=np.float32)
        if state_value.shape != (7,) or action_value.shape != (7,):
            raise ValueError("state and action must both be float-compatible shape (7,)")
        if not np.isfinite(state_value).all() or not np.isfinite(action_value).all():
            raise ValueError("state/action contains NaN or Inf")
        frame_index = len(self._active["timestamp"])
        raw_timestamp = frame_index / self.fps if timestamp is None else float(timestamp)
        raw_front_timestamp = raw_timestamp if front_timestamp is None else float(front_timestamp)
        raw_wrist_timestamp = raw_timestamp if wrist_timestamp is None else float(wrist_timestamp)
        if not all(
            math.isfinite(value)
            for value in (raw_timestamp, raw_front_timestamp, raw_wrist_timestamp)
        ):
            raise ValueError("timestamp/front_timestamp/wrist_timestamp must be finite")
        if self._active["source_timestamp_origin"] is None:
            self._active["source_timestamp_origin"] = raw_timestamp
        origin = float(self._active["source_timestamp_origin"])
        timestamp_value = raw_timestamp - origin
        front_timestamp_value = raw_front_timestamp - origin
        wrist_timestamp_value = raw_wrist_timestamp - origin
        if self._active["timestamp"] and timestamp_value <= self._active["timestamp"][-1]:
            raise ValueError("timestamps must be strictly increasing")
        if (
            self._active["front_timestamp"]
            and front_timestamp_value <= self._active["front_timestamp"][-1]
        ):
            raise ValueError("front timestamps must be strictly increasing")
        if (
            self._active["wrist_timestamp"]
            and wrist_timestamp_value <= self._active["wrist_timestamp"][-1]
        ):
            raise ValueError("wrist timestamps must be strictly increasing")

        for camera, frame in (("front", front), ("wrist", wrist)):
            process = self._active["processes"][camera]
            if process.stdin is None:
                raise RuntimeError(f"{camera} encoder stdin is unavailable")
            process.stdin.write(frame.tobytes())
        self._active["timestamp"].append(timestamp_value)
        self._active["front_timestamp"].append(front_timestamp_value)
        self._active["wrist_timestamp"].append(wrist_timestamp_value)
        self._active["state"].append(state_value.copy())
        self._active["action"].append(action_value.copy())

    def _finish_encoders(self) -> None:
        assert self._active is not None
        failures = []
        for camera, process in self._active["processes"].items():
            try:
                if process.stdin is not None:
                    process.stdin.close()
                return_code = process.wait(timeout=300)
            except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
                process.kill()
                process.wait(timeout=30)
                failures.append(f"{camera} encoder failed: {exc}")
                continue
            if return_code != 0:
                failures.append(f"{camera} encoder exit={return_code}")
        if failures:
            raise RuntimeError(", ".join(failures))

    def end_episode(self, *, success: bool, notes: str = "") -> int | None:
        if self._active is None:
            raise RuntimeError("no active episode")
        active = self._active
        try:
            self._finish_encoders()
            if not success:
                shutil.rmtree(active["temp_dir"])
                return None
            length = len(active["timestamp"])
            if length < 16:
                raise ValueError("a successful episode must contain at least 16 frames")
            np.savez_compressed(
                active["temp_dir"] / "trajectory.npz",
                timestamp=np.asarray(active["timestamp"], dtype=np.float64),
                front_timestamp=np.asarray(active["front_timestamp"], dtype=np.float64),
                wrist_timestamp=np.asarray(active["wrist_timestamp"], dtype=np.float64),
                state=np.asarray(active["state"], dtype=np.float32),
                action=np.asarray(active["action"], dtype=np.float32),
            )
            episode_json = {
                **active["metadata"],
                "episode_index": active["index"],
                "task": active["task"],
                "success": True,
                "length": length,
                "notes": notes,
                "timestamp_policy": "episode_relative_master_clock",
                "source_timestamp_origin": active["source_timestamp_origin"],
                "video_encoder": self.encoder,
            }
            (active["temp_dir"] / "episode.json").write_text(
                json.dumps(episode_json, indent=2, ensure_ascii=False) + "\n"
            )
            os.replace(active["temp_dir"], active["final_dir"])
            return int(active["index"])
        except BaseException:
            self._stop_encoders()
            shutil.rmtree(active["temp_dir"], ignore_errors=True)
            raise
        finally:
            self._active = None

    def _stop_encoders(self) -> None:
        if self._active is None:
            return
        for process in self._active["processes"].values():
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)

    def abort_episode(self) -> None:
        if self._active is None:
            return
        active = self._active
        self._stop_encoders()
        shutil.rmtree(active["temp_dir"], ignore_errors=True)
        self._active = None

    def __enter__(self) -> "ArxRawRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._active is not None:
            self.abort_episode()
