# Architecture

The repository deliberately separates three sources of truth:

1. The ARX description repository owns URDF geometry, joint limits, SRDF,
   MoveIt kinematics, and XRDF collision spheres.
2. This repository owns simulation transport, Isaac Sim scene construction,
   startup ordering, and simulation-only controller configuration.
3. Isaac ROS 4.5 owns the cuMotion planner and MoveIt plugin.

The simulator imports the plain `r5a_cumotion.urdf`; MoveIt expands the local
`r5a.isaac_sim.urdf.xacro`, which includes the same plain URDF plus a
`topic_based_ros2_control/TopicBasedSystem` block. cuMotion receives the same
plain URDF and the ARX XRDF. This prevents the real hardware plugin
(`ArxR5aSystem`) from entering the simulation process.

The Isaac articulation publishes all eight DOFs. `TopicBasedSystem` owns only
the seven independent joints (`joint1..joint7`) and ignores `joint8` from the
incoming state message. Keeping the state-only mimic joint out of the
ros2_control hardware block is intentional: the upstream transport constructs
command names from every hardware joint, so including `joint8` without a
command interface would produce a malformed eight-name/seven-position
`JointState`. PhysX owns the `joint8 <- joint7` mimic relation.

## Runtime ownership

| Process | Owns | Reads | Writes |
|---|---|---|---|
| Isaac Sim 5.1 | Physics and articulation | `/isaac_joint_commands` | `/isaac_joint_states`, `/clock` |
| ros2_control | Trajectory interpolation | `/isaac_joint_states` | `/isaac_joint_commands`, controller actions |
| MoveIt | Robot state and execution | `/joint_states` | `FollowJointTrajectory` goals |
| cuMotion | GPU motion planning | URDF, XRDF, MoveIt request | Motion plan result |

The OmniGraph runs on `OnPhysicsStep`, so commands, joint states, and clock use
the fixed physics cadence rather than the display frame rate.

The PhysX ground plane is not automatically part of the cuMotion planning
world. With `read_esdf_world=False` and `add_ground_plane=False`, this phase
validates controller execution only; environment-aware collision planning
requires the later Nvblox/ESDF integration.

## Startup ordering

1. Isaac Sim imports the fixed-base articulation and starts publishing state.
2. `ros2_control_node` starts with `TopicBasedSystem`.
3. After three seconds, the joint-state broadcaster activates.
4. After five seconds, arm and gripper controllers activate.
5. After six seconds, `move_group` starts with a fresh simulated state.

The timers avoid controller activation failure before the first
`/isaac_joint_states` message, matching the NVIDIA Isaac ROS 4.5 example.
