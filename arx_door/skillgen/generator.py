# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Native SkillGen with one-time re-anchoring for a freely moving door."""

import torch

import numpy as np

from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.waypoint import WaypointSequence, WaypointTrajectory

from .compat import validate_mimic_api
from .entrance import (
    EntranceSelectionError,
    load_source_approach_masks,
    record_connection_audit,
    select_reachable_entrance,
    trim_trajectory_prefix,
)
from .recovery import knob_approach_mask, perturbed_entrance, rejoin_index


class DoorDataGenerator(DataGenerator):
    """Refresh a released panel after the long free-space regrasp transition.

    Native SkillGen anchors a whole skill before planning its transition.
    A hinged door can keep moving during that transition. Refresh the anchor
    once, then request a collision-checked correction before contact. Never
    follow the panel frame continuously: that would cancel the opening motion.
    """

    def __init__(self, *args, motion_planner, source_dataset_path=None, perturbation=None, **kwargs):
        validate_mimic_api(DataGenerator)
        super().__init__(*args, **kwargs)
        self.motion_planner = motion_planner
        self._panel_before_transition = {}
        self._panel_source_approach = {}
        self.entrance_adjustments = []
        # One sampled knob-entrance perturbation for this trial, or None.
        self.perturbation = perturbation
        self.recovery_audit = {} if perturbation is None else {'perturbation': perturbation}
        self._pending_recovery = {}
        self._transition_gripper = None
        self._knob_approach_mask = None
        if perturbation is not None:
            if source_dataset_path is None:
                raise ValueError('a recovery perturbation requires the source dataset')
            self._knob_approach_mask = knob_approach_mask(source_dataset_path)
        self._source_approach_masks = []
        self._source_approach_error = None
        if source_dataset_path is not None:
            try:
                self._source_approach_masks = load_source_approach_masks(
                    source_dataset_path, demo_keys=kwargs.get('demo_keys'),
                )
            except (KeyError, ValueError, OSError) as exc:
                # Older source datasets need no new annotations when their
                # original entrance is reachable. Missing proof only blocks
                # the optional trim after an original strict IK failure.
                self._source_approach_error = str(exc)

    def generate_eef_subtask_trajectory(
        self, env_id, eef_name, subtask_ind, all_randomized_subtask_boundaries,
        runtime_subtask_constraints_dict, selected_src_demo_inds,
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
            env_id, eef_name, subtask_ind, all_randomized_subtask_boundaries,
            runtime_subtask_constraints_dict, selected_src_demo_inds,
        )
        if panel:
            source_index = int(selected_src_demo_inds[eef_name])
            start, end = all_randomized_subtask_boundaries[eef_name][source_index, subtask_ind]
            start, end = int(start), int(end)
            # Native SkillGen prepends the measured first EEF pose when
            # configured, duplicating its source frame and gripper action.
            duplicated_first_pose = bool(
                subtask_ind == 0 or self.env_cfg.datagen_config.generation_transform_first_robot_pose)
            approach = {'source_demo_index': source_index, 'source_start_frame': start,
                        'source_end_frame': end, 'duplicated_first_pose': duplicated_first_pose,
                        'noncontact_mask': None}
            if source_index < len(self._source_approach_masks):
                source = self._source_approach_masks[source_index]
                mask = source['noncontact_approach'][start:end].copy()
                if duplicated_first_pose:
                    mask = np.concatenate((mask[:1], mask))
                approach.update(source_demo_key=source['demo_key'], noncontact_mask=mask)
            self._panel_source_approach[(env_id, eef_name, subtask_ind)] = approach
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
        if (self.perturbation is not None and subtask_ind == 0
                and self.env_cfg.subtask_configs[eef_name][subtask_ind].object_ref == 'knob'):
            self._prepare_recovery(env_id, eef_name, subtask_ind, trajectory,
                                   all_randomized_subtask_boundaries, selected_src_demo_inds)
        return trajectory

    def _prepare_recovery(self, env_id, eef_name, subtask_ind, trajectory, boundaries, selected):
        """Send the first transition to a perturbed entrance; rejoin later in merge."""
        source_index = int(selected[eef_name])
        start, end = (int(value) for value in boundaries[eef_name][source_index, subtask_ind])
        mask = self._knob_approach_mask[start:end]
        # Native SkillGen prepends the measured first pose for the first subtask.
        mask = np.concatenate((mask[:1], mask))
        if len(mask) != len(trajectory):
            mask = mask[1:]
        if len(mask) != len(trajectory):
            raise ValueError('knob approach mask does not match the skill waypoint count')
        rejoin = rejoin_index(mask, self.perturbation['rejoin_fraction'])
        knob = self.env.get_object_poses(env_ids=[env_id])['knob'][0]
        entrance = trajectory[0].pose
        target = perturbed_entrance(entrance.detach().cpu().numpy(),
                                    knob.detach().cpu().numpy(), self.perturbation)
        self.motion_planner.transition_goal_override = target
        if self.perturbation.get('transition_gripper_m') is not None:
            self._transition_gripper = float(self.perturbation['transition_gripper_m'])
        self._pending_recovery[(env_id, eef_name, subtask_ind)] = rejoin
        self.recovery_audit.update(
            source_start_frame=start, rejoin_waypoint_index=rejoin,
            perturbed_entrance_world=target.tolist(),
            nominal_entrance_world=entrance.detach().cpu().numpy().tolist())

    def _convert_planned_trajectory_to_waypoints(self, motion_planner, gripper_action):
        """Hold a partly closed gripper during the deliberately perturbed transition only."""
        if self._transition_gripper is not None:
            gripper_action = torch.full_like(gripper_action, self._transition_gripper)
            self._transition_gripper = None
        return super()._convert_planned_trajectory_to_waypoints(motion_planner, gripper_action)

    def merge_eef_subtask_trajectory(
        self, env_id, eef_name, subtask_index, prev_executed_traj, subtask_trajectory,
    ):
        """Re-anchor once and prepend a native cuMotion correction path."""
        rejoin = self._pending_recovery.pop((env_id, eef_name, subtask_index), None)
        if rejoin is not None:
            correction = self._plan_recovery(env_id, subtask_trajectory, rejoin)
            # The native first-subtask merge interpolates from the measured pose,
            # which is still the perturbed entrance; start from the correction end.
            config = self.env_cfg.subtask_configs[eef_name][subtask_index]
            merged = WaypointTrajectory()
            merged.add_waypoint_sequence(WaypointSequence(sequence=[correction[-1]]))
            merged.merge(subtask_trajectory, num_steps_interp=config.num_interpolation_steps,
                         num_steps_fixed=config.num_fixed_steps,
                         action_noise=float(config.apply_noise_during_interpolation) * config.action_noise)
            merged.pop_first()
            return correction + merged.get_full_sequence().sequence
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
            source = self._panel_source_approach.pop(
                (env_id, eef_name, subtask_index), {},
            )
            mask = source.pop('noncontact_mask', None)
            duplicated_first_pose = source.pop('duplicated_first_pose', False)
            metadata = {'env_id': int(env_id), 'eef_name': eef_name,
                        'subtask_index': int(subtask_index), **source}
            if self._source_approach_error is not None:
                metadata['source_annotations_error'] = self._source_approach_error
            try:
                adjustment = select_reachable_entrance(
                    subtask_trajectory, kinematics=self.motion_planner.kinematics,
                    base_world=self.env.get_robot_base_pose(env_id),
                    start_joints=self.env.get_arm_joint_positions(env_id),
                    noncontact_mask=mask,
                )
            except EntranceSelectionError as exc:
                self.entrance_adjustments.append({**metadata, **exc.metrics})
                raise
            if 'source_start_frame' in metadata:
                adjustment['selected_source_frame'] = metadata['source_start_frame'] + max(
                    0, adjustment['selected_waypoint_index'] - int(duplicated_first_pose),
                )
            entrance_audit = {**metadata, **adjustment, 'applied_trimmed_waypoint_count': 0}
            self.entrance_adjustments.append(entrance_audit)
            selected_waypoint = subtask_trajectory[adjustment['selected_waypoint_index']]
            print(
                f"Selected panel entrance: {adjustment['status']}, "
                f"requested trim {adjustment['trimmed_waypoint_count']}/"
                f"{adjustment['original_waypoint_count']} waypoints, "
                f"joint margin {adjustment['selected_joint_margin_rad']:.6f} rad",
                flush=True,
            )
            try:
                connected = self.motion_planner.update_world_and_plan_motion(
                    target_pose=selected_waypoint.pose,
                    expected_attached_object=None, env_id=env_id,
                )
            except Exception as exc:
                # Do not attribute the previous plan's result to an exception
                # before this planner call produced its own native response.
                record_connection_audit(entrance_audit, connected=False, error=exc)
                raise
            record_connection_audit(
                entrance_audit, connected=connected, result=self.motion_planner.last_result,
            )
            if not connected:
                status = entrance_audit['connection_status']
                reason = entrance_audit.get('connection_reason', 'no collision-checked trajectory')
                raise RuntimeError(
                    f'cuMotion could not connect the refreshed panel entrance: {status}; {reason}',
                )
            # Retain the original trajectory when the collision planner rejects
            # the selected endpoint. A valid IK solution alone never applies a trim.
            trim_trajectory_prefix(subtask_trajectory, adjustment['trimmed_waypoint_count'])
            entrance_audit['applied_trimmed_waypoint_count'] = adjustment['trimmed_waypoint_count']
            correction = self._convert_planned_trajectory_to_waypoints(
                self.motion_planner, subtask_trajectory[0].gripper_action,
            )
            prev_executed_traj = correction
        return correction + super().merge_eef_subtask_trajectory(
            env_id, eef_name, subtask_index, prev_executed_traj, subtask_trajectory,
        )

    def _plan_recovery(self, env_id, subtask_trajectory, rejoin):
        """Plan from the perturbed entrance back onto the seed approach line.

        cuMotion's conservative spheres can reject approach waypoints near the
        knob as colliding targets; only then retry earlier, still contact-free
        waypoints, down to the first one after the original entrance.
        """
        self.motion_planner.transition_goal_override = None
        self.motion_planner.approach_standoff_m = 0.0
        self.recovery_audit['status'] = 'recovery_planning'
        attempts = self.recovery_audit.setdefault('rejoin_attempts', [])
        clearance = self.perturbation.get('direct_clearance_m')
        for candidate in range(rejoin, 0, -max(1, rejoin // 4)):
            self.motion_planner.direct_clearance_override_m = clearance
            try:
                connected = self.motion_planner.update_world_and_plan_motion(
                    target_pose=subtask_trajectory[candidate].pose,
                    expected_attached_object=None, env_id=env_id,
                )
            except Exception as exc:
                record_connection_audit(self.recovery_audit, connected=False, error=exc)
                raise
            finally:
                self.motion_planner.direct_clearance_override_m = None
            status = (self.motion_planner.last_result or {}).get('status')
            attempts.append({'waypoint_index': candidate, 'status': status})
            if connected or status != 'INVALID_TARGET_CSPACE_POSITION':
                break
        record_connection_audit(self.recovery_audit, connected=connected,
                                result=self.motion_planner.last_result)
        if not connected:
            raise RuntimeError(
                'cuMotion could not connect the perturbed entrance back to the knob approach: '
                f"{self.recovery_audit.get('connection_reason', 'no trajectory')}")
        self.recovery_audit.update(status='recovery_connected', rejoin_waypoint_index=candidate)
        trim_trajectory_prefix(subtask_trajectory, candidate)
        correction = self._convert_planned_trajectory_to_waypoints(
            self.motion_planner, subtask_trajectory[0].gripper_action,
        )
        self.recovery_audit['correction_waypoints'] = len(correction)
        self.recovery_audit['path_mode'] = (self.motion_planner.last_result or {}).get('path_mode')
        return correction
