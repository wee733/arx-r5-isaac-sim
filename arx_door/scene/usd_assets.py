# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Copy USD assets with exact project relocation and explicit missing-file errors."""

import os
from pathlib import Path


# Scenes recorded before the move reference arx_r5_isaac_sim_bringup/assets/<suffix>;
# the same suffix now resolves under the project assets/ directory.
from ..paths import ASSETS as PROJECT_ASSETS


def _migrated_project_asset(value: str) -> Path | None:
    """Resolve only an unambiguous, unchanged package/assets suffix."""
    parts = Path(value).parts
    markers = [index for index in range(len(parts) - 1)
               if parts[index:index + 2] == ('arx_r5_isaac_sim_bringup', 'assets')]
    if len(markers) != 1:
        return None
    suffix = parts[markers[0] + 2:]
    if not suffix or '..' in suffix:
        return None
    candidate = PROJECT_ASSETS.joinpath(*suffix)
    return candidate.resolve() if candidate.is_file() else None


def rebase_local_assets(layer, source: Path, destination: Path) -> dict:
    """Update a copied layer, preserving valid absolute and remote/SDK paths.

    Relative paths are rebased for the destination. A missing project asset
    may move only by its exact package/assets suffix. Unknown missing local
    resources fail before callers publish or render the copied scene.
    """
    from pxr import UsdUtils

    source = source.resolve()
    destination = destination.resolve()
    report = {'schema': 'arx-usd-local-assets-v1', 'source_layer': str(source),
              'destination_layer': str(destination), 'remapped_assets': [],
              'rebased_assets': [], 'validated_local_assets': 0}
    validated = set()
    missing = set()
    recorded = set()

    def relocate(value: str) -> str:
        # 内置材质由 Isaac Sim 解析；远程资源保持原始 URI。
        if not value or '://' in value or value.lower().endswith('.mdl'):
            return value
        asset = Path(value)
        if not asset.is_absolute():
            asset = source.parent / asset
        asset = asset.resolve()
        if asset.is_file():
            validated.add(str(asset))
            if Path(value).is_absolute():
                return value
            authored = os.path.relpath(asset, destination.parent)
            if authored != value and value not in recorded:
                report['rebased_assets'].append({'source_asset': value,
                    'resolved_asset': str(asset), 'authored_asset': authored})
                recorded.add(value)
            return authored
        migrated = _migrated_project_asset(value)
        if migrated is None:
            missing.add((value, str(asset)))
            return value
        authored = os.path.relpath(migrated, destination.parent)
        validated.add(str(migrated))
        if value not in recorded:
            report['remapped_assets'].append({'source_asset': value,
                'resolved_source': str(asset), 'resolved_asset': str(migrated),
                'authored_asset': authored, 'reason': 'exact_project_assets_suffix'})
            recorded.add(value)
        return authored

    UsdUtils.ModifyAssetPaths(layer, relocate)
    if missing:
        details = '; '.join(f'{value} (resolved {resolved})' for value, resolved in sorted(missing))
        raise FileNotFoundError(f'missing local USD assets; no exact project relocation: {details}')
    report['validated_local_assets'] = len(validated)
    return report


def copy_flattened_scene(source: Path, destination: Path) -> dict:
    """Flatten a disposable copy after repairing/checking its authored assets."""
    from pxr import Sdf, Usd

    source, destination = source.resolve(), destination.resolve()
    original = Sdf.Layer.FindOrOpen(str(source))
    if original is None:
        raise ValueError(f'could not open source USD: {source}')
    if destination.exists():
        raise FileExistsError(f'copied USD already exists: {destination}')
    copied = Sdf.Layer.CreateAnonymous('expert_asset_copy.usda')
    copied.TransferContent(original)
    report = rebase_local_assets(copied, source, destination)
    temporary = destination.with_name(f'.{destination.stem}.flatten{destination.suffix}')
    try:
        if not copied.Export(str(temporary)):
            raise RuntimeError('could not export disposable USD copy')
        stage = Usd.Stage.Open(str(temporary))
        if stage is None:
            raise RuntimeError('could not open disposable USD copy')
        report['composed_layer_checks'] = []
        for used in stage.GetUsedLayers():
            if not used.realPath or Path(used.realPath).resolve() == temporary:
                continue
            checked = Sdf.Layer.CreateAnonymous('asset_preflight.usda')
            checked.TransferContent(used)
            result = rebase_local_assets(checked, Path(used.realPath), destination)
            # Flattening preserves texture attributes, but an unresolved nested
            # composition arc could already have omitted model geometry.
            if any(Path(item['resolved_asset']).suffix.lower() in ('.usd', '.usda', '.usdc')
                   for item in result['remapped_assets']):
                raise ValueError(f'nested USD composition needs explicit relocation: {used.realPath}')
            report['composed_layer_checks'].append({
                'source_layer': used.realPath,
                'validated_local_assets': result['validated_local_assets'],
            })
        flattened = stage.Flatten()
        after = rebase_local_assets(flattened, temporary, destination)
        # Root paths were repaired before composition; composed material paths
        # are checked again before export. Source layers are never edited.
        report['remapped_assets'].extend(after['remapped_assets'])
        report['rebased_assets'].extend(after['rebased_assets'])
        report['flattened_validated_local_assets'] = after['validated_local_assets']
        if not flattened.Export(str(destination)):
            raise RuntimeError('flattened USD export failed')
    finally:
        temporary.unlink(missing_ok=True)
    return report
