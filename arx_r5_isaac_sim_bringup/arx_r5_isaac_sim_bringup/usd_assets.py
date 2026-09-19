# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""复制 USD 图层时保持本地资源引用可解析。"""

import os
from pathlib import Path


def rebase_local_assets(layer, source: Path, destination: Path) -> None:
    """将源图层的本地资源路径改写为相对目标图层的路径。"""
    from pxr import UsdUtils

    source = source.resolve()
    destination = destination.resolve()

    def relocate(value: str) -> str:
        # 内置材质由 Isaac Sim 解析；远程资源保持原始 URI。
        if not value or '://' in value or value.endswith('.mdl'):
            return value
        asset = Path(value)
        if not asset.is_absolute():
            asset = source.parent / asset
        return os.path.relpath(asset.resolve(), destination.parent)

    UsdUtils.ModifyAssetPaths(layer, relocate)
