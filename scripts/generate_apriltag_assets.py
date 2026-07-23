#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the exact tag36h11 textures used by the tabletop demo."""

from pathlib import Path

import cv2
import numpy as np


def generate_tag(destination: Path, tag_id: int) -> None:
    """Write an 800-pixel tag with one-module white quiet zone."""
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    marker = cv2.aruco.drawMarker(dictionary, tag_id, 640)
    canvas = np.full((800, 800), 255, dtype=np.uint8)
    canvas[80:720, 80:720] = marker
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f'failed to write {destination}')


def main() -> None:
    """Generate source and destination textures beside the package."""
    repository_root = Path(__file__).resolve().parents[1]
    asset_directory = (
        repository_root / 'arx_r5_isaac_sim_bringup' / 'assets' / 'apriltag'
    )
    for tag_id in (0, 1):
        destination = asset_directory / f'tag36h11_{tag_id}.png'
        generate_tag(destination, tag_id)
        print(destination)


if __name__ == '__main__':
    main()
