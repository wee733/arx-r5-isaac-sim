#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""检查克隆后的源码、关键场景、样本和 LFS 实体，不启动仿真。"""

from pathlib import Path
import json
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """报告未入库源码、缺失文件和未下载的 LFS 指针。"""
    tracked = set(subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '-z'], text=True,
    ).rstrip('\0').split('\0'))
    errors = []
    required = {
        'generated/door_teaching/upright02/live_scene.usd',
        'generated/door_teaching/upright02/candidate_006/episode.json',
        'generated/door_teaching/upright02/candidate_006/trajectory.npz',
        'generated/door_skillgen/seed/candidate_006.hdf5',
        'dependencies.repos', 'dependencies-isaac-ros.repos',
        'dependencies-skillgen.repos', 'configs/door_migration_samples.json',
        'patches/isaac_ros_manipulation.patch',
        'patches/isaac_ros_manipulation_arx_r5a.patch',
        'docs/migration.md',
    }
    samples_path = ROOT / 'configs/door_migration_samples.json'
    if samples_path.is_file():
        samples = json.loads(samples_path.read_text())
        for attempt in samples['generated_attempts']:
            required.update(f'{attempt}/{name}' for name in (
                'generated.hdf5', 'result.json', 'scene.json', 'cuboids.json',
                'wrist.mp4', 'overview.mp4',
            ))
        required.add(samples['validated_dataset'] + '/meta/n17_validation.json')
    for path in sorted(required):
        if path not in tracked:
            errors.append(f'必要文件尚未入库：{path}')
    for relative in sorted(tracked):
        path = ROOT / relative
        if not path.is_file():
            errors.append(f'文件缺失：{relative}')
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(ROOT):
            errors.append(f'软链接指向仓库外：{relative}')
        with path.open('rb') as stream:
            if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                errors.append(f'LFS 实体未下载，请执行 git lfs pull：{relative}')
    extensions = {'.py', '.sh', '.yaml', '.yml', '.xacro', '.md', '.scene'}
    for directory in ('scripts', 'configs', 'docs', 'arx_r5_isaac_sim_bringup'):
        for path in (ROOT / directory).rglob('*'):
            relative = path.relative_to(ROOT).as_posix()
            if path.suffix not in extensions or not path.is_file():
                continue
            if path.name == 'my_tabletop.yaml' and path.parent.name == 'config':
                continue
            if relative not in tracked:
                errors.append(f'源码或文档未入库：{relative}')
    if errors:
        print('\n'.join(errors))
        return 1
    print(f'通过：{len(tracked)} 个跟踪文件存在，必要样本齐全，未发现 LFS 指针或遗漏源码。')
    print('此检查不验证系统依赖、GPU、ROS 运行时或仿真结果；安装步骤见 docs/migration.md。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
