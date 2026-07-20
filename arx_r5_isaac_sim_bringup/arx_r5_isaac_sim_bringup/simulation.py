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

"""Build and run the fixed-base ARX R5A scene in Isaac Sim 5.1."""

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Optional, Sequence

from arx_r5_isaac_sim_bringup.contracts import (
    ALL_JOINTS,
    CLOCK_TOPIC,
    DEFAULT_JOINT_POSITIONS,
    DESCRIPTION_PACKAGE,
    INDEPENDENT_JOINTS,
    JOINT_COMMANDS_TOPIC,
    JOINT_POSITION_LIMITS,
    JOINT_STATES_TOPIC,
    positions_from_sequence,
    resolve_description_mesh_uris,
)


def _parse_position_vector(value: str) -> dict[str, float]:
    values = [float(item.strip()) for item in value.split(',') if item.strip()]
    positions = dict(positions_from_sequence(values))
    invalid = {
        name: position
        for name, position in positions.items()
        if not math.isfinite(position)
        or not (
            JOINT_POSITION_LIMITS[name][0]
            <= position
            <= JOINT_POSITION_LIMITS[name][1]
        )
    }
    if invalid:
        raise ValueError(f'initial joint positions exceed URDF limits: {invalid}')
    if positions['joint8'] != positions['joint7']:
        raise ValueError('joint8 must equal its source mimic joint7 at startup')
    return positions


def _description_share_from_ament() -> Optional[Path]:
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return None

    try:
        return Path(get_package_share_directory(DESCRIPTION_PACKAGE))
    except Exception:  # ament raises different lookup errors across ROS releases
        return None


def _source_tree_candidates() -> list[Path]:
    module_path = Path(__file__).resolve()
    candidates = []
    for parent in module_path.parents:
        candidates.extend(
            [
                parent / 'arx-r5-moveit' / DESCRIPTION_PACKAGE,
                parent / 'isaac_ros_manipulation' / 'arx-r5-moveit' /
                DESCRIPTION_PACKAGE,
            ]
        )
    return candidates


def find_description_share(explicit_path: Optional[str]) -> Path:
    """Find the installed or source-tree ARX description package."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.environ.get('ARX_R5_DESCRIPTION_SHARE'):
        candidates.append(Path(os.environ['ARX_R5_DESCRIPTION_SHARE']))

    ament_share = _description_share_from_ament()
    if ament_share is not None:
        candidates.append(ament_share)
    candidates.extend(_source_tree_candidates())

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / 'urdf' / 'r5a_cumotion.urdf').is_file():
            return candidate

    raise FileNotFoundError(
        'ARX R5A description package was not found. Source the '
        'arx-r5-moveit install space, set ARX_R5_DESCRIPTION_SHARE, or pass '
        '--description-share.'
    )


def materialize_isaac_urdf(
    source_urdf: Path,
    description_share: Path,
    destination: Path,
) -> Path:
    """Create an importer-safe URDF with resolved package mesh paths."""
    source_text = source_urdf.read_text(encoding='utf-8')
    resolved_text = resolve_description_mesh_uris(
        source_text,
        description_share,
    )
    unresolved_prefix = f'package://{DESCRIPTION_PACKAGE}/'
    if unresolved_prefix in resolved_text:
        raise ValueError(f'failed to resolve mesh URI prefix {unresolved_prefix}')
    destination.write_text(resolved_text, encoding='utf-8')
    return destination


def _build_argument_parser() -> argparse.ArgumentParser:
    default_positions = ','.join(
        str(DEFAULT_JOINT_POSITIONS[name]) for name in ALL_JOINTS
    )
    parser = argparse.ArgumentParser(
        description='Run the ARX R5A cuMotion execution scene in Isaac Sim.',
    )
    parser.add_argument(
        '--description-share',
        help='Path to the ARX robot-description package share/source directory.',
    )
    parser.add_argument(
        '--urdf',
        help=(
            'Plain URDF to import; defaults to '
            '<description-share>/urdf/r5a_cumotion.urdf.'
        ),
    )
    parser.add_argument('--headless', action='store_true')
    parser.add_argument(
        '--duration',
        type=float,
        default=0.0,
        help='Simulation seconds to run; zero runs until the app closes.',
    )
    parser.add_argument('--physics-dt', type=float, default=0.01)
    parser.add_argument('--rendering-dt', type=float, default=1.0 / 60.0)
    parser.add_argument(
        '--convex-decomposition',
        action='store_true',
        help='Use convex decomposition instead of one convex hull per collider.',
    )
    parser.add_argument('--drive-stiffness', type=float, default=625.0)
    parser.add_argument('--drive-damping', type=float, default=50.0)
    parser.add_argument(
        '--initial-positions',
        default=default_positions,
        help='Comma-separated joint1..joint8 positions.',
    )
    parser.add_argument(
        '--ros-domain-id',
        type=int,
        help='Set ROS_DOMAIN_ID before the ROS 2 bridge is loaded.',
    )
    parser.add_argument(
        '--no-ros',
        action='store_true',
        help='Build the scene without the ROS 2 Action Graph.',
    )
    parser.add_argument(
        '--save-usd',
        help='Export the generated scene to this .usd/.usda/.usdc path.',
    )
    parser.add_argument(
        '--exit-after-build',
        action='store_true',
        help='Exit after import, graph creation, initialization, and optional export.',
    )
    return parser


def _find_articulation_root(stage, robot_path: str) -> str:
    from pxr import Usd, UsdPhysics

    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f'imported robot prim does not exist: {robot_path}')

    articulation_roots = [
        prim
        for prim in Usd.PrimRange(robot_prim)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(articulation_roots) != 1:
        paths = [str(prim.GetPath()) for prim in articulation_roots]
        raise RuntimeError(
            f'expected exactly one articulation root below {robot_path}, '
            f'found {paths}'
        )
    return str(articulation_roots[0].GetPath())


def _configure_joint_drives(
    stage,
    robot_path: str,
    stiffness: float,
    damping: float,
    target_positions: dict[str, float],
) -> None:
    from pxr import Usd, UsdPhysics

    robot_prim = stage.GetPrimAtPath(robot_path)
    joint_prims = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(robot_prim)
        if prim.IsA(UsdPhysics.RevoluteJoint)
        or prim.IsA(UsdPhysics.PrismaticJoint)
    }
    missing = sorted(set(INDEPENDENT_JOINTS) - set(joint_prims))
    if missing:
        raise RuntimeError(f'cannot configure missing joint drives: {missing}')

    for joint_name in INDEPENDENT_JOINTS:
        joint_prim = joint_prims[joint_name]
        drive_name = (
            'linear' if joint_prim.IsA(UsdPhysics.PrismaticJoint) else 'angular'
        )
        drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_name)
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_name)
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        target = target_positions[joint_name]
        if drive_name == 'angular':
            target = math.degrees(target)
        drive.CreateTargetPositionAttr(target)


def _create_ros_action_graph(robot_path: str) -> None:
    import omni.graph.core as og
    import usdrt.Sdf

    target = [usdrt.Sdf.Path(robot_path)]
    og.Controller.edit(
        {
            'graph_path': '/World/ROS2Graph',
            'evaluator_name': 'execution',
            'pipeline_stage': (
                og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND
            ),
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ('OnPhysicsStep', 'isaacsim.core.nodes.OnPhysicsStep'),
                ('ReadSimTime', 'isaacsim.core.nodes.IsaacReadSimulationTime'),
                ('Context', 'isaacsim.ros2.bridge.ROS2Context'),
                ('PublishClock', 'isaacsim.ros2.bridge.ROS2PublishClock'),
                (
                    'PublishJointState',
                    'isaacsim.ros2.bridge.ROS2PublishJointState',
                ),
                (
                    'SubscribeJointState',
                    'isaacsim.ros2.bridge.ROS2SubscribeJointState',
                ),
                (
                    'ArticulationController',
                    'isaacsim.core.nodes.IsaacArticulationController',
                ),
            ],
            og.Controller.Keys.CONNECT: [
                ('OnPhysicsStep.outputs:step', 'PublishClock.inputs:execIn'),
                ('OnPhysicsStep.outputs:step', 'PublishJointState.inputs:execIn'),
                ('OnPhysicsStep.outputs:step', 'SubscribeJointState.inputs:execIn'),
                (
                    'OnPhysicsStep.outputs:step',
                    'ArticulationController.inputs:execIn',
                ),
                ('Context.outputs:context', 'PublishClock.inputs:context'),
                ('Context.outputs:context', 'PublishJointState.inputs:context'),
                ('Context.outputs:context', 'SubscribeJointState.inputs:context'),
                ('ReadSimTime.outputs:simulationTime', 'PublishClock.inputs:timeStamp'),
                (
                    'ReadSimTime.outputs:simulationTime',
                    'PublishJointState.inputs:timeStamp',
                ),
                (
                    'SubscribeJointState.outputs:jointNames',
                    'ArticulationController.inputs:jointNames',
                ),
                (
                    'SubscribeJointState.outputs:positionCommand',
                    'ArticulationController.inputs:positionCommand',
                ),
                (
                    'SubscribeJointState.outputs:velocityCommand',
                    'ArticulationController.inputs:velocityCommand',
                ),
                (
                    'SubscribeJointState.outputs:effortCommand',
                    'ArticulationController.inputs:effortCommand',
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ('PublishClock.inputs:topicName', CLOCK_TOPIC),
                ('PublishJointState.inputs:topicName', JOINT_STATES_TOPIC),
                ('PublishJointState.inputs:targetPrim', target),
                ('SubscribeJointState.inputs:topicName', JOINT_COMMANDS_TOPIC),
                ('ArticulationController.inputs:targetPrim', target),
            ],
        },
    )


def _set_initial_positions(robot, requested_positions: dict[str, float]) -> None:
    from isaacsim.core.utils.types import ArticulationAction
    import numpy as np

    dof_names = tuple(robot.dof_names)
    missing = sorted(set(ALL_JOINTS) - set(dof_names))
    if missing:
        raise RuntimeError(
            f'Isaac Sim articulation is missing expected joints: {missing}; '
            f'found {dof_names}'
        )

    positions = robot.get_joint_positions()
    if positions is None:
        positions = np.zeros(len(dof_names), dtype=np.float32)
    else:
        positions = np.asarray(positions, dtype=np.float32)
    for joint_name, value in requested_positions.items():
        positions[dof_names.index(joint_name)] = value

    robot.set_joint_positions(positions)
    robot.set_joint_velocities(np.zeros_like(positions))
    independent_indices = np.asarray(
        [dof_names.index(name) for name in INDEPENDENT_JOINTS],
        dtype=np.int32,
    )
    robot.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=positions[independent_indices],
            joint_indices=independent_indices,
        )
    )


def _export_stage(stage, destination: str) -> None:
    from isaacsim.core.utils.stage import save_stage

    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_stage(
        str(destination_path),
        save_and_reload_in_place=False,
    ):
        raise RuntimeError(f'failed to export USD stage to {destination_path}')
    print(f'[arx-r5-sim] exported scene: {destination_path}', flush=True)


def _run_simulation(args: argparse.Namespace) -> None:
    if args.physics_dt <= 0.0 or args.rendering_dt <= 0.0:
        raise ValueError('physics and rendering time steps must be positive')
    if args.duration < 0.0:
        raise ValueError('duration must be non-negative')
    if args.drive_stiffness <= 0.0 or args.drive_damping < 0.0:
        raise ValueError('drive stiffness must be positive and damping non-negative')

    if args.ros_domain_id is not None:
        os.environ['ROS_DOMAIN_ID'] = str(args.ros_domain_id)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            'headless': args.headless,
            'width': 1280,
            'height': 720,
        }
    )

    world = None
    failure = None
    try:
        from isaacsim.asset.importer.urdf import _urdf
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.viewports import set_camera_view
        import omni.kit.commands
        import omni.usd
        from pxr import PhysxSchema, Sdf, UsdLux

        enable_extension('isaacsim.asset.importer.urdf')
        if not args.no_ros:
            enable_extension('isaacsim.ros2.bridge')
        simulation_app.update()

        description_share = find_description_share(args.description_share)
        source_urdf = (
            Path(args.urdf).expanduser().resolve()
            if args.urdf
            else description_share / 'urdf' / 'r5a_cumotion.urdf'
        )
        if not source_urdf.is_file():
            raise FileNotFoundError(f'URDF does not exist: {source_urdf}')

        requested_positions = _parse_position_vector(args.initial_positions)
        print(f'[arx-r5-sim] description: {description_share}', flush=True)
        print(f'[arx-r5-sim] importing: {source_urdf}', flush=True)
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=args.physics_dt,
            rendering_dt=args.rendering_dt,
        )
        world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()

        distant_light = UsdLux.DistantLight.Define(
            stage,
            Sdf.Path('/World/DistantLight'),
        )
        distant_light.CreateIntensityAttr(1000.0)

        with tempfile.TemporaryDirectory(prefix='arx_r5a_isaac_sim_') as temp_dir:
            resolved_urdf = materialize_isaac_urdf(
                source_urdf,
                description_share,
                Path(temp_dir) / 'r5a_isaac_sim.urdf',
            )
            status, import_config = omni.kit.commands.execute(
                'URDFCreateImportConfig'
            )
            if not status:
                raise RuntimeError('failed to create the Isaac Sim URDF import config')
            import_config.merge_fixed_joints = False
            import_config.convex_decomp = args.convex_decomposition
            import_config.fix_base = True
            import_config.make_default_prim = False
            import_config.create_physics_scene = False
            import_config.import_inertia_tensor = True
            import_config.parse_mimic = True
            import_config.self_collision = False
            import_config.density = 0.0
            import_config.distance_scale = 1.0
            import_config.collision_from_visuals = False
            import_config.default_drive_type = (
                _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
            )
            import_config.default_drive_strength = args.drive_stiffness
            import_config.default_position_drive_damping = args.drive_damping

            status, robot_path = omni.kit.commands.execute(
                'URDFParseAndImportFile',
                urdf_path=str(resolved_urdf),
                import_config=import_config,
                get_articulation_root=False,
            )
            if not status or not robot_path:
                raise RuntimeError(f'failed to import URDF: {resolved_urdf}')
            robot_path = str(robot_path)

        articulation_path = _find_articulation_root(stage, robot_path)
        _configure_joint_drives(
            stage,
            robot_path,
            stiffness=args.drive_stiffness,
            damping=args.drive_damping,
            target_positions=requested_positions,
        )
        articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(
            stage.GetPrimAtPath(articulation_path)
        )
        articulation_api.CreateSolverPositionIterationCountAttr(64)
        articulation_api.CreateSolverVelocityIterationCountAttr(16)

        robot = world.scene.add(
            SingleArticulation(prim_path=articulation_path, name='arx_r5a')
        )

        set_camera_view(
            eye=[1.1, 1.1, 0.8],
            target=[0.0, 0.0, 0.3],
            camera_prim_path='/OmniverseKit_Persp',
        )
        world.reset()
        _set_initial_positions(robot, requested_positions)
        if not args.no_ros:
            _create_ros_action_graph(articulation_path)
        simulation_app.update()

        print(f'[arx-r5-sim] robot prim: {robot_path}', flush=True)
        print(f'[arx-r5-sim] articulation root: {articulation_path}', flush=True)
        print(f'[arx-r5-sim] joints: {tuple(robot.dof_names)}', flush=True)
        if not args.no_ros:
            print(
                '[arx-r5-sim] ROS topics: '
                f'/{JOINT_STATES_TOPIC} -> ros2_control, '
                f'ros2_control -> /{JOINT_COMMANDS_TOPIC}',
                flush=True,
            )

        if args.save_usd:
            _export_stage(stage, args.save_usd)
        if args.exit_after_build:
            return

        world.play()
        step_count = 0
        max_steps = (
            int(args.duration / args.physics_dt) if args.duration > 0.0 else None
        )
        while simulation_app.is_running():
            world.step(render=True)
            step_count += 1
            if max_steps is not None and step_count >= max_steps:
                break
    except BaseException:
        failure = sys.exc_info()
        traceback.print_exc()
    finally:
        if world is not None:
            try:
                world.stop()
            except Exception:
                traceback.print_exc()
        simulation_app.close()
    if failure is not None:
        _, error, error_traceback = failure
        raise error.with_traceback(error_traceback)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and run the Isaac Sim application."""
    args = _build_argument_parser().parse_args(argv)
    _run_simulation(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
