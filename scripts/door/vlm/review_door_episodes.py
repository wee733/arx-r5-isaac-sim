#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Ask a vision-language model (e.g. Gemini) to review door episodes.

Works on SkillGen trial directories and policy rollout directories. Writes
``vlm_review.json`` and the reviewed ``vlm_contact_sheet.jpg`` beside the
episode; ``result.json`` and every physics verdict stay untouched. The
endpoint is OpenAI-compatible (``--base-url``), the model is always explicit
and the key is read from ``DOOR_VLM_API_KEY``. ``--dry-run`` only renders the
sheet and prompt.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.files import write_json
from arx_door.vlm.review import (
    REVIEW_SCHEMA,
    OpenAICompatibleClient,
    build_prompt,
    contact_sheet,
    keyframes,
    parse_review,
    triage,
)

CAMERAS = ('wrist', 'ego', 'overview')
TASK = ('Grasp the round doorknob, turn it counterclockwise 45 degrees, and keep holding '
        'the knob while pulling the door open.')
STAGE_NAMES = ('approach', 'grasp', 'unlock', 'open')


def read_frames(video: Path, frames: list[int]) -> dict[int, np.ndarray]:
    """Decode exactly the requested frame indices as RGB arrays."""
    probe = json.loads(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height', '-of', 'json', str(video)]))['streams'][0]
    width, height = probe['width'], probe['height']
    expression = '+'.join(f'eq(n\\,{frame})' for frame in frames)
    raw = subprocess.check_output(
        ['ffmpeg', '-v', 'error', '-i', str(video), '-vf', f'select={expression}',
         '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    images = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    if len(images) != len(frames):
        raise ValueError(f'{video}: decoded {len(images)} of {len(frames)} requested frames')
    return dict(zip(frames, images))


def episode_stages(path: Path, result: dict):
    """Stage labels: a rollout records its prompter stages; a trial is aligned to its seed."""
    trajectory = path / 'rollout/trajectory.npz' if (path / 'rollout').is_dir() else path / 'trajectory.npz'
    if trajectory.is_file():
        with np.load(trajectory) as data:
            if 'stage' in data.files and len(data['stage']):
                return data['stage']
        return None
    if result.get('schema') == 'arx-door-skillgen-result-v1' and result.get('source_seed'):
        import h5py

        from arx_door.dataset.stages import annotate_generated, load_seed_skill
        try:
            with h5py.File(result['artifacts']['hdf5'], 'r') as file:
                return annotate_generated(file['data/demo_0'], load_seed_skill(result['source_seed']))['stage']
        except (KeyError, ValueError, OSError):
            return None
    return None


def review(path: Path, client, *, dry_run: bool) -> dict:
    episode = path / 'rollout' if (path / 'rollout/result.json').is_file() else path
    result = json.loads((episode / 'result.json').read_text())
    videos = {camera: episode / f'{camera}.mp4' for camera in CAMERAS
              if (episode / f'{camera}.mp4').is_file()}
    if 'wrist' not in videos:
        raise ValueError(f'{episode}: wrist.mp4 is required')
    count = int(result.get('frames') or result.get('steps') or 0)
    stage = episode_stages(path, result)
    if stage is not None:
        count = len(stage)
    if count < 2:
        raise ValueError(f'{episode}: episode has no frames to review')
    frames = keyframes(stage, count)
    decoded = {camera: read_frames(video, frames) for camera, video in videos.items()}
    tiles = []
    for frame in frames:
        label = STAGE_NAMES[int(stage[frame])] if stage is not None else '-'
        for camera in videos:
            tiles.append((f'f{frame} {camera} {label}', decoded[camera][frame]))
    jpeg = contact_sheet(tiles, columns=len(videos) * 2)
    (episode / 'vlm_contact_sheet.jpg').write_bytes(jpeg)
    prompt = build_prompt(TASK, list(videos))
    record = {'schema': REVIEW_SCHEMA, 'episode': str(episode), 'keyframes': frames,
              'cameras': list(videos), 'stage_source': None if stage is None else 'labels',
              'physics_success': result.get('success'), 'prompt': prompt}
    if dry_run:
        record['status'] = 'dry_run'
    else:
        reply = client.complete(prompt, jpeg)
        record.update(model=client.model, endpoint=client.url, raw_reply=reply)
        try:
            record['review'] = parse_review(reply)
            record['triage'] = triage(record['review'], result.get('success'))
            record['status'] = 'reviewed'
        except (ValueError, TypeError) as error:
            record.update(status='unparseable', error=str(error), triage='human_review')
    write_json(episode / 'vlm_review.json', record)
    return record


def expand(paths: list[Path], every: int) -> list[Path]:
    """Trial/rollout directories as given; batch_results.json -> every Nth successful trial."""
    episodes = []
    for path in paths:
        if path.is_file() and path.name == 'batch_results.json':
            report = json.loads(path.read_text())
            successful = [Path(item['output_dir']) for item in report['trials']
                          if item.get('result', {}).get('success')]
            episodes += successful[::every]
        else:
            episodes.append(path)
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', type=Path, nargs='+',
                        help='trial or rollout directories, or batch_results.json files')
    parser.add_argument('--every', type=int, default=1,
                        help='from batch_results.json, review every Nth successful trial')
    parser.add_argument('--base-url', help='OpenAI-compatible endpoint, e.g. the Gemini OpenAI API')
    parser.add_argument('--model', help='VLM model name; never defaulted')
    parser.add_argument('--dry-run', action='store_true', help='render sheets and prompts only')
    args = parser.parse_args()
    client = None if args.dry_run else OpenAICompatibleClient(args.base_url, args.model)
    summary = {}
    if args.every < 1:
        parser.error('--every must be positive')
    for path in expand(args.episodes, args.every):
        try:
            record = review(path.resolve(), client, dry_run=args.dry_run)
            summary[str(path)] = record.get('triage', record['status'])
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            summary[str(path)] = f'error: {error}'
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if all(not value.startswith('error') for value in summary.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
