#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Encode inspectable attempt videos without relabeling failed attempts as demos."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def review_video(episode, metadata, data):
    """Make a labeled review copy; original policy camera videos stay separate."""
    def timecode(seconds):
        milliseconds = round(seconds*1000)
        hours, milliseconds = divmod(milliseconds, 3600000)
        minutes, milliseconds = divmod(milliseconds, 60000)
        seconds, milliseconds = divmod(milliseconds, 1000)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'

    phases = data['phase']
    count = len(phases)
    boundaries = [0, *(np.flatnonzero(phases[1:] != phases[:-1])+1).tolist(), count]
    subtitles = []
    for number, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        subtitles.append(f'{number}\n{timecode(start/metadata["fps"])} --> '
                         f'{timecode(end/metadata["fps"])}\n{phases[start]}\n')
    caption = episode / 'review_phases.srt'
    caption.write_text('\n'.join(subtitles))
    # Use cwd and a fixed subtitle basename so arbitrary absolute paths never
    # become filter-graph syntax. FFmpeg is invoked without a shell.
    status = 'PHYSICAL CHECKS PASSED - REVIEW PENDING' if metadata['success'] else 'FAILED DIAGNOSTIC'
    filters = (
        "[0:v]scale=640:480,pad=640:540:0:35:black,"
        "drawtext=text='Wrist - policy input':x=10:y=8:fontsize=18:fontcolor=white[left];"
        "[1:v]scale=640:480,pad=640:540:0:35:black,"
        f"drawtext=text='Overview - {status}':x=10:y=8:fontsize=15:fontcolor=white[right];"
        "[left][right]hstack=inputs=2,subtitles=review_phases.srt:"
        "force_style='FontSize=12,MarginV=3'[review]")
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', 'wrist.mp4', '-i', 'overview.mp4', '-filter_complex', filters,
                    '-map', '[review]', '-c:v', 'libx264', '-crf', '20',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', 'review.mp4'],
                   cwd=episode, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.episode / 'episode.json').read_text())
    data = np.load(args.episode / 'trajectory.npz')
    count = len(data['timestamp'])
    if count < 2 or data['state'].shape != (count, 7) or data['action'].shape != (count, 7):
        raise ValueError('invalid or empty seven-dimensional trajectory')
    if not np.all(np.isfinite(data['state'])) or not np.all(np.isfinite(data['action'])):
        raise ValueError('nonfinite trajectory')
    if not np.allclose(np.diff(data['timestamp']), 1/metadata['fps'], atol=1e-7):
        raise ValueError('recording is not uniformly sampled at the declared FPS')
    for camera in ['wrist', 'overview']:
        if len(list((args.episode / camera).glob('*.png'))) != count:
            raise ValueError(f'{camera} image count differs from trajectory')
        output = args.episode / f'{camera}.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-framerate', str(metadata['fps']), '-i',
                        str(args.episode / camera / '%06d.png'), '-c:v', 'libx264',
                        '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(output)], check=True)
        probe = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=nb_frames', '-of', 'json', str(output)])
        if int(json.loads(probe)['streams'][0]['nb_frames']) != count:
            raise ValueError(f'{camera} encoded frame count mismatch')
    review_video(args.episode.resolve(), metadata, data)
    print(json.dumps({'episode': str(args.episode), 'frames': count,
                      'success': metadata['success'], 'video_checked': True}))


if __name__ == '__main__':
    main()
