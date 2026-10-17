#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Audit recorded episodes and create a reviewable closed-loop evaluation report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.files import write_json
from arx_door.runtime import video_tool


def review_video(path: Path, expected_frames: int) -> tuple[dict, list[Image.Image]]:
    ffprobe, environment = video_tool('ffprobe')
    probe = json.loads(subprocess.check_output(
        [ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames',
         '-of', 'json', str(path)], env=environment))['streams'][0]
    if int(probe['nb_read_frames']) != expected_frames:
        raise ValueError(f'{path}: decoded frame count differs from executed steps')
    ffmpeg, environment = video_tool('ffmpeg')
    frames = []
    for index in (0, expected_frames // 2, expected_frames - 1):
        data = subprocess.check_output(
            [ffmpeg, '-v', 'error', '-ss', str(index / 30), '-i', str(path),
             '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], env=environment)
        rgb = np.frombuffer(data, np.uint8).reshape(probe['height'], probe['width'], 3)
        if float(rgb.std()) < 2:
            raise ValueError(f'{path}: near-uniform frame {index}')
        frames.append(Image.fromarray(rgb).resize((320, 240)))
    return probe, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    args = parser.parse_args()
    root = args.evaluation.absolute()
    summary = json.loads((root / 'summary.json').read_text())
    rows, audit = [], []
    for item in summary['episodes']:
        name = item['scenario']['name']
        result = item['result']
        directory = root / name / 'rollout'
        entry = {'scenario': name, 'status': result['status']}
        if result['status'] == 'completed' and result['steps'] == 0:
            entry.update(frames=0, artifact_status='no_executed_steps')
        elif result['status'] == 'completed':
            with np.load(directory / 'trajectory.npz', allow_pickle=False) as arrays:
                count = len(arrays['state'])
                if count != result['steps'] or arrays['command'].shape != (count, 7):
                    raise ValueError(f'{name}: trajectory does not match executed steps')
                if not all(np.isfinite(arrays[key]).all() for key in arrays.files):
                    raise ValueError(f'{name}: trajectory contains nonfinite values')
                entry['frames'] = count
                entry['inference_seconds_median'] = float(np.median(arrays['inference'][:, 1]))
                entry['maximum_door_angle_deg'] = float(np.rad2deg(arrays['door_joint_pos'][:, 0]).max())
                entry['command_tracking_mae'] = np.abs(arrays['command'] - arrays['actual']).mean(0).tolist()
            sheet = Image.new('RGB', (960, 520), 'white')
            draw = ImageDraw.Draw(sheet)
            entry['videos'] = {}
            for row, camera in enumerate(('overview', 'wrist')):
                metadata, frames = review_video(directory / f'{camera}.mp4', count)
                entry['videos'][camera] = metadata
                for column, frame in enumerate(frames):
                    sheet.paste(frame, (column * 320, row * 260 + 20))
                    draw.text((column * 320 + 5, row * 260 + 3),
                              f'{name} / {camera} / {("start", "middle", "end")[column]}', fill='black')
            sheet.save(directory / 'review.png')
        audit.append(entry)
        metrics = result.get('physical_metrics', {})
        angle = metrics.get('final_door_angle_deg')
        angle_text = '-' if angle is None else f'{angle:.3f}'
        rows.append(f'| {name} | {result["status"]} | {result.get("steps", 0)} | '
                    f'{angle_text} | {result.get("stop_reason", "-")} | '
                    f'{"成功" if result["success"] else "未成功"} |')
    write_json(root / 'artifact_audit.json', audit)
    lines = [
        '# ARX GR00T N1.7 闭环评估', '',
        (f'完成 {summary["completed"]}/{summary["requested"]} 个 episode；'
        f'物理成功 {summary["successes"]} 个。评估完成：{summary["evaluation_complete"]}。'), '',
        ('这是小规模仿真试验，不是实机验证或总体泛化成功率。动作来自微调模型，'
        '物理等待推理时暂停；没有用专家轨迹、IK 或规划器补动作。'), '',
        '| 场景 | 运行状态 | 执行步数 | 最终门角 ° | 停止原因 | 物理结果 |',
        '|---|---|---:|---:|---|---|', *rows, '',
        ('采用训练时的 source_cpu 物理参数，120Hz 物理 / 30Hz 控制。'
        '每次预测 16 步、执行前 8 步，再根据实际腕图像和关节状态推理。'), '',
        ('成功门槛包括门角 >29°、双指接触与相反力方向、解锁/门缝顺序、持续握持、'
        '非手指接触 ≤2N 和相机间隙 ≥1cm，持续 30 步。'), '',
        ('有执行步的已完成 episode 已核对数值轨迹形状和有限值，两个视频已全量解码计帧；'
        '开始/中间/结束画面汇成 review.png。零步终止只记录实际失败，不生成执行录像验收。'), '',
        '## 证据入口', '',
    ]
    for item in summary['episodes']:
        name = item['scenario']['name']
        if item['result']['status'] != 'completed' or item['result']['steps'] == 0:
            lines += [(f'- **{name}**：[指标]({name}/rollout/result.json)；'
                      '无已完成的执行轨迹/录像验收。')]
            continue
        lines += [(f'- **{name}**：[指标]({name}/rollout/result.json)、'
                  f'[总览录像]({name}/rollout/overview.mp4)、'
                  f'[腕部录像]({name}/rollout/wrist.mp4)、'
                  f'[轨迹]({name}/rollout/trajectory.npz)。'),
                  f'\n![{name} 开始/中间/结束]({name}/rollout/review.png)\n']
    lines += ['配置见 `policy_config.json` / `evaluation_config.json`，实际采样见 `scenarios.json`。',
              '原始汇总见 `summary.json`；独立数组与视频检查见 `artifact_audit.json`。', '']
    (root / '报告.md').write_text('\n'.join(lines))
    print(root / '报告.md')


if __name__ == '__main__':
    main()
