# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Small metadata hooks around the official AutoData SkillGen algorithm."""

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from autodata_core.algorithms import SkillGen as SkillGenAlgorithm
from autodata_core.waypoint import Waypoint

from arx_door.autodata.adapter import PLAN_GRIPPER_KEY, PLAN_Q_KEY, TERMINAL_KEY, require_zero_noise


class DoorSkillGen(SkillGenAlgorithm):
    """Retain official segment generation, merging, scheduling and execution."""

    name = 'arx_skillgen'

    def validate_setup(self, datastream) -> None:
        super().validate_setup(datastream)
        if list(datastream.get_eef_names()) != ['link6']:
            raise ValueError('ARX DoorSkillGen requires exactly link6')
        for subtask in datastream.get_subtasks('link6'):
            require_zero_noise(subtask.action_noise)
        for planner in self.motion_planners.values():
            require_zero_noise(getattr(planner.config, 'motion_noise_scale', 0.))

    @staticmethod
    def _convert_planned_trajectory_to_waypoints(
        motion_planner: Any,
        passthrough_action: dict[str, torch.Tensor],
    ) -> list[Waypoint]:
        """Pair every real planner pose with its precise six joint targets."""
        require_zero_noise(getattr(motion_planner.config, 'motion_noise_scale', 0.))
        poses = motion_planner.get_planned_poses()
        joints = motion_planner.get_planned_joint_positions()
        gripper = motion_planner.get_planned_gripper_joint_positions()
        if not poses or len(poses) != len(joints):
            raise ValueError('planner pose/joint sequences must be nonempty and aligned')
        if gripper.shape != (2,) or not torch.isfinite(gripper).all():
            raise ValueError('planner must provide measured joint7/joint8 for the fixed gripper model')
        waypoints = []
        for pose, joint_positions in zip(poses, joints):
            if joint_positions.shape != (6,) or not torch.isfinite(joint_positions).all():
                raise ValueError('planner metadata must contain six finite joints')
            metadata = dict(passthrough_action)
            # The worker held both fingers fixed at their measured state. The
            # source skill's first q7 may differ and is restored by the native
            # skill merge only after this collision-checked transit ends.
            metadata['link6'] = gripper[:1].clone()
            metadata[PLAN_Q_KEY] = joint_positions.clone()
            metadata[PLAN_GRIPPER_KEY] = gripper.clone()
            metadata[TERMINAL_KEY] = torch.tensor(False, device=pose.device)
            # MultiWaypoint.execute forwards passthrough metadata; it does not
            # forward Waypoint.joint_seed, so joint_seed is deliberately unused.
            waypoints.append(Waypoint(pose=pose.clone(), passthrough_action=metadata, noise=0.))
        return waypoints

    def plan_subtask_trajectory(
        self,
        *,
        data_generator,
        env_id: int,
        eef_name: str,
        eef_state,
        all_randomized_subtask_boundaries: dict,
        runtime_subtask_constraints_dict: dict,
        selected_src_demo_inds: dict,
    ) -> tuple[list[Waypoint], bool] | None:
        result = super().plan_subtask_trajectory(
            data_generator=data_generator,
            env_id=env_id,
            eef_name=eef_name,
            eef_state=eef_state,
            all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
            selected_src_demo_inds=selected_src_demo_inds,
        )
        if result is None:
            return None
        trajectory, is_motion_plan = result
        is_final_skill = not is_motion_plan and (
            eef_state.current_subtask_index == data_generator.datastream.num_subtasks(eef_name) - 1
        )
        annotated = []
        for index, original in enumerate(trajectory):
            waypoint = copy(original)
            waypoint.passthrough_action = dict(original.passthrough_action)
            if not is_motion_plan:
                # Merging can carry the final transit waypoint into an
                # interpolated skill segment. That segment must use seeded IK.
                waypoint.passthrough_action.pop(PLAN_Q_KEY, None)
                waypoint.passthrough_action.pop(PLAN_GRIPPER_KEY, None)
            waypoint.passthrough_action[TERMINAL_KEY] = torch.tensor(
                is_final_skill and index == len(trajectory) - 1, device=waypoint.pose.device,
            )
            require_zero_noise(waypoint.noise)
            annotated.append(waypoint)
        return annotated, is_motion_plan
