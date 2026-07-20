# Third-party notices

Most original integration code in this repository is licensed under
Apache-2.0 as stated in the top-level `LICENSE` file. The repository does not
redistribute ARX robot meshes, NVIDIA Isaac Sim assets, or NVIDIA Isaac ROS
binaries.

## ARX R5A configuration

The following files are adaptations of configuration from
[`Isaac_Ros_CuMotion_ArxR5a`](https://github.com/wee733/Isaac_Ros_CuMotion_ArxR5a)
at commit `c85c2c7c84bb630f30f87904bebebbd83dddd2db`:

- `arx_r5_isaac_sim_bringup/urdf/r5a.isaac_sim.ros2_control.xacro`
- `arx_r5_isaac_sim_bringup/urdf/r5a.isaac_sim.urdf.xacro`
- `arx_r5_isaac_sim_bringup/config/initial_positions.yaml`
- `arx_r5_isaac_sim_bringup/config/moveit_controllers.yaml`
- `arx_r5_isaac_sim_bringup/config/ros2_controllers.yaml`

Those files retain the ARXrobotics/wee733 copyright and are distributed under
BSD-3-Clause. The complete terms are in `LICENSES/BSD-3-Clause.txt`.

Robot geometry, SRDF, XRDF, and the authoritative cuMotion URDF remain external
dependencies and are consumed at runtime rather than copied into this repository.

## NVIDIA and MoveIt

`arx_r5_isaac_sim_bringup/launch/arx_r5a_isaac_sim.launch.py` is adapted from
NVIDIA's Isaac ROS cuMotion Isaac Sim launch example. Its source header retains
the NVIDIA Apache-2.0 notice and the complete Willow Garage/PickNik
BSD-3-Clause notice inherited from the MoveIt tutorial source.

The integration follows NVIDIA's versioned
[cuMotion MoveIt with Isaac Sim tutorial](https://nvidia-isaac-ros.github.io/v/release-4.5/concepts/manipulation/cumotion_moveit/tutorial_isaac_sim.html)
and the Isaac Sim 5.1 ROS 2 MoveIt sample architecture. Isaac Sim, Isaac ROS
cuMotion, ROS 2, MoveIt, and `topic_based_ros2_control` retain their respective
licenses.

`dependencies.repos` pins but does not redistribute PickNik Robotics'
`topic_based_ros2_control`; its source remains under its upstream BSD-3-Clause
license when imported into a workspace.
