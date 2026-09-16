#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Paused-between-commands Isaac Sim session for inspectable robot teaching."""
import argparse
import json
from pathlib import Path
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--initial-positions', type=json.loads)
    parser.add_argument('--initial-gripper', type=float, default=.028)
    parser.add_argument('--gripper-force', type=float, default=40.0)
    parser.add_argument('--gripper-stiffness', type=float, default=2000.0)
    parser.add_argument('--mimic-frequency', type=float, default=0.0,
                        help='Passive gear compliance frequency; 0 is a rigid mimic constraint.')
    parser.add_argument('--camera-clearance', type=float, default=.01,
                        help='Minimum virtual viewpoint clearance from the door panel, metres.')
    parser.add_argument('--arm-contact-limit', type=float, default=2.0,
                        help='Abort above this net contact force on any non-finger arm link, N.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 960, 'height': 720})
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, RigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    from PIL import Image
    from arx_r5_isaac_sim_bringup.door_teaching_kinematics import ArmKinematics
    from arx_r5_isaac_sim_bringup.door_view_guard import point_box_clearance

    root = Path(__file__).resolve().parents[1]
    description = root.parent / 'arx-r5-moveit/isaac_ros_manipulation_arx_r5a_robot_description'
    kin = ArmKinematics(description / 'urdf/r5a_cumotion.urdf')
    omni.usd.get_context().open_stage(str(args.scene.resolve()),
                                    load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE)
    stage = omni.usd.get_context().get_stage()
    # Session-only changes: camera, correct front-view knob rotation, drive setup.
    stage.SetEditTarget(stage.GetSessionLayer())
    initial_cache = UsdGeom.XformCache()
    panel_zero = np.asarray(initial_cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath('/World/Door/door_panel'))).T
    panel_cube = stage.GetPrimAtPath('/World/Door/door_panel/Colliders/panel')
    box_zero = np.asarray(initial_cache.GetLocalToWorldTransform(panel_cube)).T
    box_scale = np.linalg.norm(box_zero[:3, :3], axis=0)
    box_half = box_scale * float(UsdGeom.Cube(panel_cube).GetSizeAttr().Get()) / 2
    box_zero[:3, :3] /= box_scale
    camera_offset = np.array([.1044276157, -.0115000179, .0767273606])
    for sensor in ['/R5a/base_link/ZED_X', '/R5a/link6/rsd455']:
        if stage.GetPrimAtPath(sensor):
            stage.GetPrimAtPath(sensor).SetActive(False)
    handle_joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/World/Door/handle_joint'))
    # Both local frames rotate Y to -Y: positive 45 deg is CCW seen from door front.
    handle_joint.GetLocalRot0Attr().Set(Gf.Quatf(0, 0, 0, 1))
    handle_joint.GetLocalRot1Attr().Set(Gf.Quatf(0, 0, 0, 1))
    # A single convex hull bridges the concave finger/rack geometry and fills
    # the grasp opening. Decompose the actual mesh; do not remove collisions.
    for finger in ('link7', 'link8'):
        stage.GetPrimAtPath(f'/R5a/{finger}/collisions').SetInstanceable(False)
        collider = stage.GetPrimAtPath(f'/R5a/{finger}/collisions/{finger}/node_STL_BINARY_')
        UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Set('convexDecomposition')
        decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(collider)
        decomposition.CreateMaxConvexHullsAttr(32)
        decomposition.CreateHullVertexLimitAttr(64)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(collider)
        collision.CreateContactOffsetAttr(.001)
        collision.CreateRestOffsetAttr(0)
    wrist = UsdGeom.Camera.Define(stage, '/R5a/link6/TeachingWristCamera')
    wrist.AddTranslateOp().Set(Gf.Vec3d(0.1044276157, -0.0115000179, 0.0767273606))
    # USD camera looks along local -Z, with +Y up. Tool +X down 30 deg.
    cam_rotation = np.array([[0, .5, -np.sqrt(3)/2], [-1, 0, 0],
                             [0, np.sqrt(3)/2, .5]])
    quat = Rotation.from_matrix(cam_rotation).as_quat()
    wrist.AddOrientOp().Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*quat[:3])))
    wrist.CreateFocalLengthAttr(1.93)
    wrist.CreateHorizontalApertureAttr(3.896)
    wrist.CreateVerticalApertureAttr(2.922)
    wrist.CreateClippingRangeAttr(Gf.Vec2f(.01, 10))
    for prim in stage.Traverse():
        if prim.GetTypeName() in ('PhysicsRevoluteJoint', 'PhysicsPrismaticJoint'):
            if str(prim.GetPath()).startswith('/R5a/'):
                drive = UsdPhysics.DriveAPI(prim, 'linear' if prim.GetTypeName() ==
                                           'PhysicsPrismaticJoint' else 'angular')
                if drive:
                    drive.GetStiffnessAttr().Set(args.gripper_stiffness if prim.GetTypeName() ==
                                                 'PhysicsPrismaticJoint' else 2000)
                    drive.GetDampingAttr().Set(50 if prim.GetTypeName() ==
                                              'PhysicsPrismaticJoint' else 100)
                    drive.GetMaxForceAttr().Set(args.gripper_force if prim.GetTypeName() ==
                                               'PhysicsPrismaticJoint' else 100)
                    if prim.GetTypeName() == 'PhysicsPrismaticJoint':
                        drive.GetTypeAttr().Set('force')
    mimic = stage.GetPrimAtPath('/R5a/joints/joint8')
    mimic.GetAttribute('physxMimicJoint:rotY:dampingRatio').Set(1.0)
    mimic.GetAttribute('physxMimicJoint:rotY:naturalFrequency').Set(args.mimic_frequency)
    for _ in range(5):
        app.update()
    world = World(stage_units_in_meters=1.0, physics_prim_path='/physicsScene',
                  physics_dt=1/120, rendering_dt=1/120)
    robot = world.scene.add(SingleArticulation('/R5a', name='robot'))
    door = world.scene.add(SingleArticulation('/World/Door', name='door'))
    fingers = world.scene.add(RigidPrim(
        prim_paths_expr=['/R5a/link7', '/R5a/link8'], name='finger_contacts',
        contact_filter_prim_paths_expr=[
            ['/World/Door/door_handle', '/World/Door/door_panel'],
            ['/World/Door/door_handle', '/World/Door/door_panel']],
        track_contact_forces=True, reset_xform_properties=False))
    arm_contacts = world.scene.add(RigidPrim(
        prim_paths_expr=[f'/R5a/link{i}' for i in range(1, 7)], name='arm_contacts',
        contact_filter_prim_paths_expr=[[] for _ in range(6)],
        track_contact_forces=True, reset_xform_properties=False))
    world.reset()
    names = list(robot.dof_names)
    arm_indices = [names.index(f'joint{i}') for i in range(1, 7)]
    finger_indices = [names.index(n) for n in ('joint7', 'joint8') if n in names]
    initial = robot.get_joint_positions().copy()
    initial[arm_indices] = args.initial_positions or [0, 1, 1.5, 0, 0, 0]
    initial[finger_indices] = args.initial_gripper
    robot.set_joint_positions(initial)
    robot.set_joint_velocities(np.zeros_like(initial))
    robot.apply_action(ArticulationAction(joint_positions=initial))
    manifest = json.loads(args.scene.with_suffix('.json').read_text())
    base = np.eye(4)
    base[:3, 3] = manifest['placement']['position']
    base[:3, :3] = Rotation.from_quat(manifest['placement']['quaternion_xyzw']).as_matrix()
    inverse_base = np.linalg.inv(base)
    annotators = {}
    for key, path, resolution in [('wrist', str(wrist.GetPath()), (640, 480)),
                                  ('overview', '/World/OverviewCamera', (960, 720))]:
        product = rep.create.render_product(path, resolution)
        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach(product)
        annotators[key] = annotator
    for _ in range(30):
        world.step(render=True)
    sequence = 0
    tick = 0
    recording = None
    commands = []
    last_grip_target = args.initial_gripper
    safety_events = []

    def camera_clearance(arm=None, hinge_angle=None):
        if arm is None:
            arm = robot.get_joint_positions()[arm_indices]
        if hinge_angle is None:
            hinge_angle = float(door.get_joint_positions()[0])
        tool = base @ kin.fk(arm)
        camera_world = tool[:3, :3] @ camera_offset + tool[:3, 3]
        rotation = Rotation.from_euler('z', -hinge_angle).as_matrix()
        box = box_zero.copy()
        box[:3, 3] = panel_zero[:3, 3] + rotation @ (box_zero[:3, 3]-panel_zero[:3, 3])
        box[:3, :3] = rotation @ box_zero[:3, :3]
        return point_box_clearance(camera_world, box, box_half)

    def check_runtime_safety():
        clearance = camera_clearance()
        forces = np.linalg.norm(arm_contacts.get_net_contact_forces(dt=1/120), axis=1)
        reason = None
        if clearance < args.camera_clearance:
            reason = f'virtual camera clearance {clearance:.6f} m below {args.camera_clearance}'
        elif np.max(forces) > args.arm_contact_limit:
            reason = f'non-finger link{int(np.argmax(forces))+1} contact {np.max(forces):.3f} N'
        if reason:
            safety_events.append({'sim_time': tick/120, 'reason': reason})
            raise RuntimeError(reason)

    def rgba(key):
        data = annotators[key].get_data()
        if not isinstance(data, np.ndarray) or data.size == 0:
            raise RuntimeError(f'{key} camera has no rendered data')
        return np.ascontiguousarray(data[:, :, :3])

    def transform(path):
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        return np.array(matrix).T

    def observe():
        nonlocal sequence
        sequence += 1
        result = {'sequence': sequence, 'sim_time': tick/120,
                  'joint_names': names, 'joints': robot.get_joint_positions().tolist(),
                  'door_joint_names': list(door.dof_names),
                  'door_joints': door.get_joint_positions().tolist(),
                  'finger_contact_forces': fingers.get_contact_force_matrix(dt=1/120).tolist(),
                  'finger_contact_filter_order': ['knob', 'panel'],
                  'camera_panel_clearance_m': camera_clearance(),
                  'arm_contact_forces': arm_contacts.get_net_contact_forces(dt=1/120).tolist(),
                  'safety_events': safety_events.copy(),
                  'world_to_base': base.tolist(), 'images': {}}
        for name, path in [('link6', '/R5a/link6'),
                           ('knob', '/World/Door/door_handle/grasp_target'),
                           ('panel', '/World/Door/door_panel')]:
            result[name] = transform(path).tolist()
        for key in annotators:
            path = args.output / f'{sequence:04d}_{key}.png'
            Image.fromarray(rgba(key)).save(path)
            result['images'][key] = str(path.resolve())
        result['fk_error_m'] = float(np.linalg.norm(
            (base @ kin.fk(robot.get_joint_positions()[arm_indices]))[:3, 3]
            - np.asarray(result['link6'])[:3, 3]))
        (args.output / 'latest.json').write_text(json.dumps(result, indent=2))
        return result

    def step_target(arm, grip, phase):
        nonlocal tick, last_grip_target
        target = robot.get_joint_positions().copy()
        target[arm_indices] = arm
        target[finger_indices] = grip
        check_runtime_safety()
        if recording is not None and tick % 4 == 0:
            number = len(recording['state'])
            current = robot.get_joint_positions()
            recording['state'].append(np.r_[current[arm_indices], current[finger_indices[0]]])
            recording['action'].append(np.r_[arm, grip])
            recording['timestamp'].append(tick/120 - recording['start_time'])
            recording['phase'].append(phase)
            recording['door'].append(door.get_joint_positions().copy())
            recording['contacts'].append(fingers.get_contact_force_matrix(dt=1/120).copy())
            recording['camera_clearance'].append(camera_clearance())
            recording['arm_contacts'].append(arm_contacts.get_net_contact_forces(dt=1/120).copy())
            for key in annotators:
                Image.fromarray(rgba(key)).save(recording['path'] / key / f'{number:06d}.png',
                                                compress_level=1)
        robot.apply_action(ArticulationAction(joint_positions=target))
        last_grip_target = float(grip)
        world.step(render=(tick % 4 == 3))
        tick += 1
        check_runtime_safety()

    def execute(request):
        nonlocal recording
        operation = request.get('op', 'observe')
        current = robot.get_joint_positions()
        seed = current[arm_indices]
        # A blocked finger's measured position differs from its closing target.
        # Starting each segment at the measurement would release the grasp load.
        grip_start = last_grip_target
        grip = float(request.get('gripper', grip_start))
        if not 0 <= grip <= .044:
            raise ValueError('gripper must be between 0 and 0.044 m')
        duration = float(request.get('duration', 2))
        if not .03 <= duration <= 60:
            raise ValueError('duration must be between .03 and 60 seconds')
        count = max(1, int(duration * 120))
        phase = request.get('phase', operation)
        if operation == 'begin':
            if recording:
                raise ValueError('episode is already recording')
            path = args.output / request.get('name', f'attempt_{sequence:04d}')
            path.mkdir(exist_ok=False)
            for key in annotators:
                (path / key).mkdir()
            recording = {'path': path, 'state': [], 'action': [], 'timestamp': [],
                         'phase': [], 'door': [], 'contacts': [], 'camera_clearance': [],
                         'arm_contacts': [], 'start_time': tick/120}
        elif operation == 'end':
            if not recording:
                raise ValueError('no recording episode')
            path = recording['path']
            if recording['timestamp']:
                origin = recording['timestamp'][0]
                recording['timestamp'] = [t-origin for t in recording['timestamp']]
            np.savez_compressed(path / 'trajectory.npz',
                                **{key: np.asarray(recording[key]) for key in
                                   ['state', 'action', 'timestamp', 'phase', 'door', 'contacts',
                                    'camera_clearance', 'arm_contacts']})
            metadata = {'schema': 'arx-door-teaching-attempt-v1',
                        'success': bool(request.get('success', False)) and not safety_events,
                        'review_required': True, 'fps': 30,
                        'frames': len(recording['state']), 'door_joint_names': list(door.dof_names),
                        'camera': 'link6 +X downward 30 degrees, provisional D455 intrinsics',
                        'finger_collision': 'mesh convex decomposition, 32 hulls, 1mm contact offset',
                        'gripper_drive': {'type': 'force', 'stiffness': args.gripper_stiffness,
                                          'damping': 50, 'max_force_N': args.gripper_force,
                                          'mimic_frequency': args.mimic_frequency,
                                          'real_hardware_calibrated': False},
                        'scene_placement': manifest, 'notes': request.get('notes', ''),
                        'viewpoint_guard': {'minimum_clearance_m': args.camera_clearance,
                                            'physical_camera_housing_modeled': False},
                        'arm_contact_limit_N': args.arm_contact_limit,
                        'safety_events': safety_events.copy(),
                        'commands': commands.copy()}
            (path / 'episode.json').write_text(json.dumps(metadata, indent=2))
            recording = None
        elif operation in ('joints', 'hold'):
            goal = np.asarray(request.get('positions', seed), dtype=float)
            if goal.shape != (6,) or not np.all(np.isfinite(goal)):
                raise ValueError('positions must be six finite joint values')
            if np.any(goal < kin.lower) or np.any(goal > kin.upper):
                raise ValueError('joint target outside URDF limits')
            if np.max(np.abs(goal-seed))/duration > 0.6:
                raise ValueError('joint motion exceeds teaching speed limit')
            for fraction in np.linspace(1/count, 1, count):
                step_target(seed + (goal-seed)*fraction,
                            grip_start + (grip-grip_start)*fraction, phase)
        elif operation == 'pose':
            goal = np.eye(4)
            goal[:3, 3] = request['position']
            goal[:3, :3] = Rotation.from_quat(request['quaternion_xyzw']).as_matrix()
            if request.get('frame', 'world') == 'world':
                goal = inverse_base @ goal
            start = kin.fk(seed)
            slerp = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], goal[:3, :3]]))
            # Preflight whole segment before applying it; local solver has no collision model.
            joints = [seed]
            samples = max(2, int(duration * 30))
            for fraction in np.linspace(0, 1, samples)[1:]:
                pose = np.eye(4)
                pose[:3, 3] = start[:3, 3] + fraction * (goal[:3, 3]-start[:3, 3])
                pose[:3, :3] = slerp(fraction).as_matrix()
                joints.append(kin.ik(pose, joints[-1], attempts=2))
            joints = np.asarray(joints)
            if np.max(np.abs(np.diff(joints, axis=0))) * (samples-1)/duration > 0.6:
                raise ValueError('IK path jumps or exceeds teaching speed limit')
            for fraction in np.linspace(1/count, 1, count):
                q = np.array([np.interp(fraction, np.linspace(0, 1, samples), joints[:,j])
                              for j in range(6)])
                step_target(q, grip_start+(grip-grip_start)*fraction, phase)
        elif operation == 'trajectory':
            trajectory = json.loads(Path(request['path']).read_text())
            if not trajectory.get('success'):
                raise ValueError('cannot execute unsuccessful planner output')
            positions = np.asarray(trajectory['positions'])
            times = np.asarray(trajectory['timestamps'])
            if positions.shape != (len(times), 6) or len(times) < 2:
                raise ValueError('invalid planned trajectory')
            if np.max(np.abs(positions[0]-seed)) > .05:
                raise ValueError('planned start differs from live robot state')
            if not np.all(np.isfinite(positions)) or not np.all(np.diff(times)>0):
                raise ValueError('nonfinite trajectory or nonmonotonic timestamps')
            if np.any(positions<kin.lower-.001) or np.any(positions>kin.upper+.001):
                raise ValueError('planned trajectory exceeds joint limits')
            stretch = max(1, float(request.get('time_scale', 1)))
            duration = float(times[-1]-times[0])*stretch
            # Reject a planned free-space trajectory before any physics step if
            # its virtual camera would enter the panel. The scene is paused.
            for q in positions:
                if camera_clearance(q) < args.camera_clearance:
                    raise ValueError('planned trajectory violates virtual camera clearance')
            for instant in np.arange(1/120, duration+1/240, 1/120):
                q = np.array([np.interp(times[0]+instant/stretch, times, positions[:,j])
                              for j in range(6)])
                step_target(q, grip, phase)
        elif operation == 'snapshot':
            path = args.output / 'live_scene.usd'
            stage.Flatten().Export(str(path))
            path.with_suffix('.json').write_text(json.dumps(manifest))
        elif operation not in ('observe', 'shutdown'):
            raise ValueError(f'unknown operation {operation}')
        commands.append(request)
        return observe()

    work = queue.Queue()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 1000000:
                    raise ValueError('invalid request size')
                request = json.loads(self.rfile.read(length))
                response_queue = queue.Queue()
                work.put((request, response_queue))
                response = response_queue.get(timeout=300)
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(data)
            except Exception as error:
                self.send_error(500, str(error))
        def log_message(self, *unused):
            pass
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print('TEACHING_READY ' + json.dumps(observe()), flush=True)
    try:
        while True:
            request, response_queue = work.get()
            try:
                response = {'ok': True, **execute(request)}
            except Exception as error:
                response = {'ok': False, 'error': str(error), **observe()}
            response_queue.put(response)
            if request.get('op') == 'shutdown':
                break
    finally:
        server.shutdown()
        app.close()


if __name__ == '__main__':
    main()
