"""Read-only collision diagnosis after a failed transit; never used to accept a plan."""
import json
import pickle
from pathlib import Path


def report_start_collisions(planner):
    import torch
    from curobo.geom.sdf.world import CollisionQueryBuffer
    checker = planner.motion_gen.world_coll_checker
    q = planner._get_current_joint_state_for_curobo().position
    spheres = planner.motion_gen.kinematics.get_state(q).get_link_spheres().unsqueeze(1)
    buffer = CollisionQueryBuffer.initialize_from_shape(spheres.shape, planner.tensor_args, checker.collision_types)
    weight = torch.ones(1, device=spheres.device)
    activation = torch.zeros(1, device=spheres.device)
    objects = planner._static_world_config.objects
    rows = []
    all_rows = []
    # Temporarily isolate obstacles only for diagnosis, then restore every one.
    # The failed plan is never retried against this diagnostic world.
    try:
        for obj in objects:
            checker.enable_obstacle(obj.name, False)
        for obj in objects:
            checker.enable_obstacle(obj.name, True)
            distances = checker.get_sphere_distance(spheres, buffer, weight, activation,
                                                   compute_esdf=True).reshape(-1).clone()
            ids = torch.where((distances > 0) & (spheres.reshape(-1, 4)[:, 3] > 0))[0]
            valid_ids = torch.where(spheres.reshape(-1, 4)[:, 3] > 0)[0]
            worst = valid_ids[torch.argmax(distances[valid_ids])]
            feasible = planner.motion_gen.rollout_fn.rollout_constraint(q.unsqueeze(1), use_batch_env=False).feasible.item()
            all_rows.append({'obstacle': obj.name, 'max_signed_distance': float(distances[worst]),
                             'worst_sphere': int(worst), 'rollout_constraint_feasible': bool(feasible)})
            if len(ids):
                rows.append({'obstacle': obj.name, 'pose_wxyz': obj.pose,
                             'sphere_ids': ids.cpu().tolist(),
                             'penetration_m': distances[ids].cpu().tolist(),
                             'spheres_base_xyzr': spheres.reshape(-1, 4)[ids].cpu().tolist()})
            checker.enable_obstacle(obj.name, False)
    finally:
        for obj in objects:
            checker.enable_obstacle(obj.name, True)
    cfg = planner.motion_gen.kinematics.kinematics_config
    report = {'joint_state': q.cpu().tolist(), 'collisions': rows, 'all_obstacles': all_rows,
              'link_names': cfg.link_names,
              'link_sphere_idx_map': cfg.link_sphere_idx_map.cpu().tolist()}
    output = Path(planner.env.cfg.recorders.dataset_export_dir_path)
    (output / 'planning_failure.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'planning_world.pkl').open('wb') as file:
        pickle.dump(planner._static_world_config, file)
    import shutil
    shutil.copy(planner.config.robot_config_file, output / 'planning_robot.yaml')
    from PIL import Image
    from arx_door.envs.acceptance import as_numpy
    for name in ('wrist', 'ego', 'overview'):
        sensor = planner.env.scene.sensors.get(name + '_camera')
        if sensor is not None:
            Image.fromarray(as_numpy(sensor.data.output['rgb'])[0, ..., :3]).save(output / f'failure_{name}.png')
    print('ARX_START_COLLISIONS', json.dumps(rows), flush=True)
