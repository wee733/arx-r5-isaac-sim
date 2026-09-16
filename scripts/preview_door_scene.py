#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Open the authored door scene in Isaac Sim, optionally saving a smoke-test image."""

import argparse
import json
from pathlib import Path


def main():
    """Run a standalone preview without the tabletop task controller or ROS graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'generated/door/scene.usd')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=0, help='0 runs until the window closes')
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    if args.headless and args.frames <= 0:
        parser.error('headless mode requires --frames > 0')
    if not args.scene.is_file():
        parser.error('scene is missing; run scripts/run_author_door_scene.sh first')

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 960})
    try:
        import numpy as np
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.viewports import set_camera_view
        # Leave remote sensor payloads unloaded for an offline geometry preview.
        context = omni.usd.get_context()
        if not context.open_stage(str(args.scene.resolve()),
                                  load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE):
            raise RuntimeError('failed to open the door stage')
        for _ in range(10):
            app.update()
        world = World(stage_units_in_meters=1.0, physics_prim_path='/physicsScene')
        robot = world.scene.add(SingleArticulation('/R5a', name='arx_door_robot'))
        world.reset()
        set_camera_view(eye=np.array([2.2, -3.1, 2.1]),
                        target=np.array([0.0, -0.35, 0.95]))
        before, _ = robot.get_world_pose()
        before = before.copy()
        annotator = None
        if args.screenshot:
            import omni.replicator.core as rep
            product = rep.create.render_product('/World/OverviewCamera', (1280, 960))
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach(product)
        frame = 0
        while app.is_running() and (args.frames == 0 or frame < args.frames):
            world.step(render=True)
            frame += 1
        after, _ = robot.get_world_pose()
        expected = json.loads(args.scene.with_suffix('.json').read_text())['placement']['position']
        error = float(np.linalg.norm(after - np.asarray(expected)))
        check = {
            'frames': frame, 'base_before': before.tolist(), 'base_after': after.tolist(),
            'expected_base': expected, 'base_position_error_m': error,
            'passed': error <= 1e-3,
        }
        args.scene.with_suffix('.check.json').write_text(json.dumps(check, indent=2) + '\n')
        print('DOOR_SCENE_CHECK ' + json.dumps(check), flush=True)
        if error > 1e-3:
            raise RuntimeError(f'base departed from the sampled pose by {error:.6f} m')
        if annotator:
            from PIL import Image
            rgb = annotator.get_data()
            if not isinstance(rgb, np.ndarray) or rgb.size == 0:
                raise RuntimeError('renderer did not return an image')
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(args.screenshot)
    finally:
        app.close()


if __name__ == '__main__':
    main()
