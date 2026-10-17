# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Keyframe selection, prompt and strict response parsing for VLM episode review.

Transport is any OpenAI-compatible ``/chat/completions`` endpoint (Gemini
exposes one); the model name is always caller-supplied.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request

import numpy as np

from ..dataset.stages import STAGE_NAMES

VERDICTS = ('pass', 'fail', 'uncertain')
# Closed-loop failure taxonomy, ordered along the task.
FAILURE_MODES = (
    'none', 'approach_misaligned', 'gripper_closed_early', 'missed_grasp', 'grasp_slipped',
    'knob_not_turned', 'door_not_pulled', 'collision_with_door', 'joint_limit_or_singularity',
    'camera_view_blocked', 'rendering_artifact', 'other',
)
REVIEW_SCHEMA = 'arx-door-vlm-review-v1'

PROMPT = """You review a simulated robot-arm episode. Task: {task}
The images are keyframes in time order; each tile is labelled with its frame
index, camera ({cameras}) and the stage the data pipeline assigned
({stages}). The wrist camera is on the gripper; other views are fixed.

Judge only what is visible. Report, as one JSON object and nothing else:
{{"verdict": one of {verdicts},
  "stages": {{stage name: {{"visible_success": true|false|null, "note": short text}}}},
  "failure_mode": one of {modes},
  "first_bad_frame": frame index or null,
  "visual_defects": [short texts: interpenetration, flicker, occlusion, ...],
  "summary": one sentence}}
Use "uncertain" when the keyframes cannot show the outcome."""


def keyframes(stage: np.ndarray | None, count: int, *, per_stage: int = 2,
              uniform: int = 8) -> list[int]:
    """Pick first/last frames of every stage, else uniform samples, plus the end."""
    if stage is None or not len(stage):
        return sorted({int(x) for x in np.linspace(0, count - 1, uniform)})
    stage = np.asarray(stage)
    frames = {0, count - 1}
    for value in np.unique(stage):
        indices = np.flatnonzero(stage == value)
        frames.update(int(x) for x in np.linspace(indices[0], indices[-1], per_stage))
    return sorted(frames)


def contact_sheet(tiles: list[tuple[str, np.ndarray]], *, columns: int = 4,
                  tile_width: int = 320) -> bytes:
    """Compose labelled RGB tiles into one JPEG (the endpoint sees one image)."""
    from PIL import Image, ImageDraw

    images = []
    for label, frame in tiles:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        image = image.resize((tile_width, int(image.height * tile_width / image.width)))
        ImageDraw.Draw(image).rectangle((0, 0, tile_width, 16), fill=(0, 0, 0))
        ImageDraw.Draw(image).text((4, 2), label, fill=(255, 255, 0))
        images.append(image)
    height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * tile_width, rows * height))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * tile_width, (index // columns) * height))
    buffer = io.BytesIO()
    sheet.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


def build_prompt(task: str, cameras: list[str]) -> str:
    return PROMPT.format(task=task, cameras=', '.join(cameras), stages=', '.join(STAGE_NAMES),
                         verdicts=list(VERDICTS), modes=list(FAILURE_MODES))


def parse_review(text: str) -> dict:
    """Extract and validate the JSON verdict; reject anything outside the schema.

    Raises ``ValueError`` for missing JSON or unknown labels, ``TypeError`` for
    wrongly typed fields.
    """
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('VLM reply contains no JSON object')
    review = json.loads(text[start:end + 1])
    if review.get('verdict') not in VERDICTS:
        raise ValueError(f'invalid verdict {review.get("verdict")!r}')
    if review.get('failure_mode') not in FAILURE_MODES:
        raise ValueError(f'invalid failure_mode {review.get("failure_mode")!r}')
    if not isinstance(review.get('stages', {}), dict) or not isinstance(review.get('visual_defects', []), list):
        raise TypeError('stages must be an object and visual_defects a list')
    return review


def triage(review: dict, physics_success: bool | None) -> str:
    """Combine without overriding: disagreements go to a human."""
    if physics_success is None:
        return 'vlm_only'
    if review['verdict'] == 'uncertain' or review.get('visual_defects'):
        return 'human_review'
    agree = (review['verdict'] == 'pass') == bool(physics_success)
    return 'consistent' if agree else 'human_review'


class OpenAICompatibleClient:
    """Minimal chat-completions client; reads the key from an environment variable."""

    def __init__(self, base_url: str, model: str, *, api_key_env: str = 'DOOR_VLM_API_KEY',
                 timeout_s: float = 120.0):
        if not base_url or not model:
            raise ValueError('a VLM base URL and model name are required')
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.model, self.api_key_env, self.timeout_s = model, api_key_env, timeout_s

    def complete(self, prompt: str, jpeg: bytes) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f'set {self.api_key_env} to the endpoint API key')
        image = 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()
        body = {'model': self.model, 'temperature': 0,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': image}}]}]}
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), method='POST',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            reply = json.loads(response.read())
        return reply['choices'][0]['message']['content']
