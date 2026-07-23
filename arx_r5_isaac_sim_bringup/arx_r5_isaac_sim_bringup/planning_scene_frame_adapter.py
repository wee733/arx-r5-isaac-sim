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

"""Normalize NVIDIA's static MoveIt scene to the ARX planning frame."""

from copy import deepcopy

from moveit_msgs.msg import PlanningScene
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class PlanningSceneFrameAdapter(Node):
    """Rewrite only static collision-object headers from world to base_link."""

    def __init__(self) -> None:
        super().__init__('planning_scene_frame_adapter')
        self._input_topic = str(
            self.declare_parameter(
                'input_topic',
                '/cumotion/static_planning_scene_raw',
            ).value
        ).strip()
        self._output_topic = str(
            self.declare_parameter(
                'output_topic',
                '/planning_scene',
            ).value
        ).strip()
        self._source_frame = str(
            self.declare_parameter('source_frame', 'world').value
        ).strip()
        self._target_frame = str(
            self.declare_parameter('target_frame', 'base_link').value
        ).strip()
        if not self._input_topic or not self._output_topic:
            raise ValueError('planning-scene topics must be non-empty')
        if self._input_topic == self._output_topic:
            raise ValueError('planning-scene input and output topics must differ')
        if not self._source_frame or not self._target_frame:
            raise ValueError('planning-scene frames must be non-empty')
        if self._source_frame == self._target_frame:
            raise ValueError('planning-scene source and target frames must differ')

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self._publisher = self.create_publisher(
            PlanningScene,
            self._output_topic,
            qos,
        )
        self._subscription = self.create_subscription(
            PlanningScene,
            self._input_topic,
            self._on_scene,
            qos,
        )
        self._cached_scene = None
        self._last_subscription_count = 0
        self._late_joiner_timer = self.create_timer(
            0.25,
            self._republish_for_late_joiner,
        )
        self.get_logger().info(
            f'Adapting {self._input_topic} collision objects from '
            f'{self._source_frame} to {self._target_frame} on '
            f'{self._output_topic}'
        )

    def _on_scene(self, message: PlanningScene) -> None:
        scene = deepcopy(message)
        converted = 0
        for collision_object in scene.world.collision_objects:
            frame_id = collision_object.header.frame_id
            if frame_id == self._source_frame:
                collision_object.header.frame_id = self._target_frame
                converted += 1
            elif frame_id != self._target_frame:
                self.get_logger().error(
                    f'Refusing collision object {collision_object.id!r} in '
                    f'unexpected frame {frame_id!r}'
                )
                return
        if not scene.world.collision_objects:
            return
        if converted == 0:
            self.get_logger().debug('Static planning scene already normalized')
        scene.is_diff = True
        self._cached_scene = scene
        self._publisher.publish(scene)
        self.get_logger().info(
            f'Published {len(scene.world.collision_objects)} collision objects '
            f'in {self._target_frame}'
        )

    def _republish_for_late_joiner(self) -> None:
        count = self._publisher.get_subscription_count()
        if self._cached_scene is not None and count > self._last_subscription_count:
            self._publisher.publish(self._cached_scene)
        self._last_subscription_count = count


def main() -> None:
    """Run the planning-scene frame adapter."""
    rclpy.init()
    node = PlanningSceneFrameAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
