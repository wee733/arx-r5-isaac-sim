# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Native SkillGen with one-time re-anchoring for a freely moving door."""

import torch

from isaaclab_mimic.datagen.data_generator import DataGenerator


class DoorDataGenerator(DataGenerator):
    """Refresh a released panel after the long free-space regrasp transition.

    Native SkillGen anchors a whole skill before planning its transition.
    A hinged door can keep moving during that transition. Refresh the anchor
    once, then request a collision-checked correction before contact. Never
    follow the panel frame continuously: that would cancel the opening motion.
    """

    def __init__(self, *args, motion_planner, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_planner = motion_planner
        self._panel_before_transition = {}

    def generate_eef_subtask_trajectory(
        self, env_id, eef_name, subtask_ind, *args, **kwargs,
    ):
        """Remember the actual panel frame used by the native transform."""
        panel = self.env_cfg.subtask_configs[eef_name][subtask_ind].object_ref == 'panel'
        # Do not finish a long transition at a stale, nearly touching pose:
        # the released panel can swing into that pose while the arm travels.
        self.motion_planner.approach_standoff_m = 0.0
        self.motion_planner.transition_goal_override = None
        if panel:
            self._panel_before_transition[(env_id, eef_name, subtask_ind)] = (
                self.env.get_object_poses(env_ids=[env_id])['panel'][0].clone()
            )
        trajectory = super().generate_eef_subtask_trajectory(
            env_id, eef_name, subtask_ind, *args, **kwargs,
        )
        if panel:
            reference = self._panel_before_transition[(env_id, eef_name, subtask_ind)]
            intermediate = trajectory[0].pose.clone()
            # Temporarily face the panel during transfer. Keeping the nearly
            # singular side-grasp orientation while retreating is unreachable
            # on ARX. The actual contact skill retains its original rotation.
            intermediate[:3, 3] -= 0.04 * reference[:3, 1]
            intermediate[:3, :3] = torch.stack((
                reference[:3, 1], -reference[:3, 0], reference[:3, 2],
            ), dim=1)
            self.motion_planner.transition_goal_override = intermediate.cpu().numpy()
        return trajectory

    def merge_eef_subtask_trajectory(
        self, env_id, eef_name, subtask_index, prev_executed_traj, subtask_trajectory,
    ):
        """Re-anchor once and prepend a native cuMotion correction path."""
        previous = self._panel_before_transition.pop(
            (env_id, eef_name, subtask_index), None,
        )
        correction = []
        if previous is not None:
            current = self.env.get_object_poses(env_ids=[env_id])['panel'][0]
            delta = current @ torch.linalg.inv(previous)
            entrance_before = subtask_trajectory[0].pose.clone()
            for sequence in subtask_trajectory.waypoint_sequences:
                for waypoint in sequence.sequence:
                    waypoint.pose = delta @ waypoint.pose
            displacement = float(torch.linalg.vector_norm(
                subtask_trajectory[0].pose[:3, 3] - entrance_before[:3, 3],
            ))
            print(f'Re-anchor panel after transition: entrance moved {displacement:.6f} m',
                  flush=True)
            self.motion_planner.approach_standoff_m = 0.0
            self.motion_planner.transition_goal_override = None
            if not self.motion_planner.update_world_and_plan_motion(
                target_pose=subtask_trajectory[0].pose,
                expected_attached_object=None, env_id=env_id,
            ):
                raise RuntimeError('cuMotion could not connect the refreshed panel entrance')
            correction = self._convert_planned_trajectory_to_waypoints(
                self.motion_planner, subtask_trajectory[0].gripper_action,
            )
            prev_executed_traj = correction
        return correction + super().merge_eef_subtask_trajectory(
            env_id, eef_name, subtask_index, prev_executed_traj, subtask_trajectory,
        )
