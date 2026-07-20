# ARX R5A cuMotion Simulation for Isaac Sim

[中文](README.md)

This repository provides the simulation-only half of the ARX R5A cuMotion
integration. It implements a complete MoveIt execution loop for Isaac Sim 5.1:

```text
MoveIt FollowJointTrajectory
  -> ros2_control TopicBasedSystem
  -> /isaac_joint_commands
  -> Isaac Sim articulation
  -> /isaac_joint_states
  -> ros2_control and MoveIt
```

The baseline is Ubuntu 24.04, ROS 2 Jazzy, Isaac Sim 5.1, and Isaac ROS 4.5.
Robot meshes and planning descriptions remain in
[`Isaac_Ros_CuMotion_ArxR5a`](https://github.com/wee733/Isaac_Ros_CuMotion_ArxR5a);
they are not duplicated here.

For setup, launch, validation, USD generation, and known model limitations,
see the [Chinese guide](README.md). The main commands are:

```bash
# Clean Isaac Sim terminal (start this first; do not source system Jazzy)
export ARX_R5_DESCRIPTION_SHARE=/path/to/isaac_ros_manipulation_arx_r5a_robot_description
./scripts/run_isaac_sim.sh

# Isaac ROS terminal
source install/setup.bash
ros2 launch arx_r5_isaac_sim_bringup arx_r5a_isaac_sim.launch.py
```

Both terminals must use the same `ROS_DOMAIN_ID` and RMW implementation.
