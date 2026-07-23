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

"""Run NVIDIA's unchanged pick-and-place tree on the simulation clock."""

import argparse
import sys

from isaac_ros_manipulation_interfaces.action import MultiObjectPickAndPlace

from isaac_ros_manipulation_pick_and_place.scripts import (
    multi_object_pick_and_place,
)

from rclpy.parameter import Parameter
from rclpy.utilities import remove_ros_args


_TERMINAL_WORKFLOW_STATUSES = frozenset({
    MultiObjectPickAndPlace.Result.FAILED,
    MultiObjectPickAndPlace.Result.SUCCESS,
    MultiObjectPickAndPlace.Result.PARTIAL_SUCCESS,
    MultiObjectPickAndPlace.Result.INCOMPLETE,
})


class _TerminalWorkflowQuiescence:
    """Pause simulation tree ticks after one accepted workflow finishes."""

    def __init__(self, orchestrator) -> None:
        self._orchestrator = orchestrator
        self._goal_started = False
        self._quiesced = False

    def __call__(self, tree) -> None:
        """Latch an active goal, then pause on an official terminal status."""
        if self._quiesced:
            return

        if self._orchestrator.is_orchestrator_busy:
            self._goal_started = True
        if not self._goal_started:
            return

        blackboard = self._orchestrator.blackboard
        if not blackboard.exists('workflow_status'):
            return
        workflow_status = blackboard.workflow_status
        if workflow_status not in _TERMINAL_WORKFLOW_STATUSES:
            return

        # Cancel only the ROS timer that drives tree ticks. The official action
        # server and its executor must remain alive long enough to deliver the
        # terminal result to the goal client.
        if tree.timer is None:
            return
        tree.timer.cancel()
        self._quiesced = True
        self._orchestrator.logger.info(
            'One-shot simulation workflow reached terminal status '
            f'{workflow_status}; behavior-tree ticking is paused'
        )


def main() -> None:
    """Configure the official orchestrator without patching its tree."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--behavior-tree-config-file', required=True)
    parser.add_argument('--blackboard-config-file', required=True)
    parser.add_argument('--log-level', default='info')
    parser.add_argument('--print-ascii-tree', action='store_true')
    parser.add_argument('--use-sim-time', action='store_true')
    parser.add_argument('--quiesce-on-terminal', action='store_true')
    arguments = parser.parse_args(remove_ros_args(args=sys.argv)[1:])

    orchestrator = (
        multi_object_pick_and_place.MultiObjectPickPlaceOrchestrator(
            behavior_tree_config_file=arguments.behavior_tree_config_file,
            blackboard_config_file=arguments.blackboard_config_file,
            print_ascii_tree=arguments.print_ascii_tree,
            manual_mode=False,
            log_level=arguments.log_level,
            frame_prefix='',
        )
    )
    if arguments.use_sim_time:
        result = orchestrator.tree.node.set_parameters([
            Parameter('use_sim_time', value=True),
        ])[0]
        if not result.successful:
            orchestrator.shutdown()
            raise RuntimeError(
                f'failed to enable simulation time: {result.reason}'
            )
    if arguments.quiesce_on_terminal:
        orchestrator.tree.add_post_tick_handler(
            _TerminalWorkflowQuiescence(orchestrator)
        )
    orchestrator.run()


if __name__ == '__main__':
    main()
