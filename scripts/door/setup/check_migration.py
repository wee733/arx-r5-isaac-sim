#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""检查克隆后的源码、关键场景、样本和 LFS 实体，不启动仿真。"""

from pathlib import Path
import json
import subprocess


ROOT = Path(__file__).resolve().parents[3]
BASE_REQUIRED = {
    'generated/door_teaching/upright02/live_scene.usd',
    'generated/door_teaching/upright02/candidate_006/episode.json',
    'generated/door_teaching/upright02/candidate_006/trajectory.npz',
    'generated/door_skillgen/seed/candidate_006.hdf5',
    'dependencies/ros.repos', 'dependencies/isaac_ros.repos',
    'dependencies/skillgen.repos', 'configs/runtime/door_migration_samples.json',
    'dependencies/patches/isaac_ros_manipulation.patch',
    'dependencies/patches/isaac_ros_manipulation_arx_r5a.patch',
    # Local-only (gitignored); restore from the migration backup, not git.
    'docs/migration.md',
    'arx_r5_isaac_sim_bringup/assets/door/door.usda',
    'arx_r5_isaac_sim_bringup/assets/door/door_visuals.usdc',
}


def inspect_migration(root: Path) -> dict:
    """Inspect files on disk, including restored or newly written untracked files."""
    root = root.resolve()
    tracked = set(subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '-z'], text=True,
    ).rstrip('\0').split('\0'))
    tracked.discard('')
    errors = []
    warnings = []
    required = set(BASE_REQUIRED)
    samples_path = root / 'configs/runtime/door_migration_samples.json'
    if samples_path.is_file():
        samples = json.loads(samples_path.read_text())
        for attempt in samples['generated_attempts']:
            required.update(f'{attempt}/{name}' for name in (
                'generated.hdf5', 'result.json', 'scene.json', 'cuboids.json',
                'wrist.mp4', 'overview.mp4',
            ))
        required.add(samples['validated_dataset'] + '/meta/n17_validation.json')
    discovered = set()
    extensions = {'.py', '.sh', '.yaml', '.yml', '.xacro', '.md', '.scene'}
    for directory in ('arx_door', 'scripts', 'configs', 'docs', 'tests',
                      'arx_r5_isaac_sim_bringup'):
        for path in (root / directory).rglob('*'):
            if path.suffix in extensions and path.is_file():
                if any(part in ('build', 'install', '__pycache__') for part in path.parts):
                    continue
                if path.name == 'my_tabletop.yaml' and path.parent.name == 'config':
                    continue
                discovered.add(path.relative_to(root).as_posix())
    inspected = tracked | required | discovered
    for relative in sorted(inspected):
        path = root / relative
        if not path.is_file():
            errors.append(f'文件缺失：{relative}')
            continue
        if relative not in tracked:
            warnings.append(f'文件存在但未跟踪（请随迁移备份，可能被 gitignore 忽略）：{relative}')
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            errors.append(f'软链接指向仓库外：{relative}')
        with path.open('rb') as stream:
            if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                errors.append(f'LFS 实体未下载，请执行 git lfs pull：{relative}')
    return {'errors': errors, 'warnings': warnings, 'inspected_files': len(inspected),
            'tracked_files': len(tracked)}


def main() -> int:
    """报告文件缺失；未提交文件仅提醒备份，不误判为丢失。"""
    report = inspect_migration(ROOT)
    for warning in report['warnings']:
        print(f'提醒：{warning}')
    errors = report['errors']
    if errors:
        print('\n'.join(errors))
        return 1
    print(f"通过：检查 {report['inspected_files']} 个文件，必要样本齐全，未发现 LFS 指针。")
    print('此检查不验证系统依赖、GPU、ROS 运行时或仿真结果；安装步骤见 docs/migration.md。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
