# ARX R5A cuMotion Simulation for Isaac Sim

[中文](README.md) · [Architecture](docs/architecture.md) · [Model validation](docs/model-validation.md)

[![CI](https://github.com/wee733/arx-r5-isaac-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/wee733/arx-r5-isaac-sim/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

<p align="center">
  <img src="pictures/arx.png" alt="ARX R5A planning and execution in RViz and Isaac Sim" width="100%">
</p>

This repository is the simulation companion to
[`Isaac_Ros_CuMotion_ArxR5a`](https://github.com/wee733/Isaac_Ros_CuMotion_ArxR5a).
It connects an ARX R5A articulation in Isaac Sim 5.1 to MoveIt 2 and NVIDIA
cuMotion through `ros2_control`:

```text
cuMotion / MoveIt FollowJointTrajectory
  -> ros2_control TopicBasedSystem
  -> /isaac_joint_commands
  -> Isaac Sim articulation
  -> /isaac_joint_states
  -> ros2_control / MoveIt
```

The current release has been validated end to end in simulation: cuMotion
**Plan / Execute** succeeds in RViz, both the manipulator
`FollowJointTrajectory` and gripper `GripperCommand` actions return
`SUCCEEDED`, and joint states continuously return from Isaac Sim to MoveIt.

The tested baseline is Ubuntu 24.04, ROS 2 Jazzy, Isaac ROS 4.5, cuMotion 4.5,
and Isaac Sim 5.1.0 installed in a conda environment named `isaaclab`.

`joint1` through `joint6` form the `manipulator` planning group. `joint7` is
the commanded gripper joint; `joint8` follows it through the PhysX mimic
constraint and must not be commanded independently.

## Prerequisites and source layout

You need an NVIDIA GPU, a working Isaac ROS 4.5 environment, `conda`, and the
`isaaclab` environment containing the Isaac Sim Python package. Verify the
latter in a clean host terminal:

```bash
conda activate isaaclab
python -c 'import isaacsim; print(isaacsim.__file__)'
```

The commands below match the layout used for the verified run:

```text
~/workspace/
├── isaac_ros_source/
│   ├── arx-r5-isaac-sim/
│   ├── topic_based_ros2_control/
│   └── isaac_ros_manipulation/arx-r5-moveit/
└── arx_r5_sim_ws/
    ├── build/
    ├── install/
    └── log/
```

The ARX meshes, URDF, SRDF, XRDF, and MoveIt configuration stay in the
`arx-r5-moveit` checkout instead of being duplicated here. Existing checkouts
can be kept; the commands below create any missing repositories and pin the
model and TopicBasedSystem revisions used for the verified run:

```bash
export ISAAC_ROS_SRC="$HOME/workspace/isaac_ros_source"
export SIM_REPO="$ISAAC_ROS_SRC/arx-r5-isaac-sim"
export ARX_REPO="$ISAAC_ROS_SRC/isaac_ros_manipulation/arx-r5-moveit"
export TOPIC_SRC="$ISAAC_ROS_SRC/topic_based_ros2_control"

mkdir -p "$ISAAC_ROS_SRC/isaac_ros_manipulation"
[[ -d "$SIM_REPO/.git" ]] || \
  git clone https://github.com/wee733/arx-r5-isaac-sim.git "$SIM_REPO"
[[ -d "$ARX_REPO/.git" ]] || \
  git clone https://github.com/wee733/Isaac_Ros_CuMotion_ArxR5a.git "$ARX_REPO"
[[ -d "$TOPIC_SRC/.git" ]] || \
  git clone https://github.com/PickNikRobotics/topic_based_ros2_control.git "$TOPIC_SRC"

git -C "$ARX_REPO" checkout c85c2c7c84bb630f30f87904bebebbd83dddd2db
git -C "$TOPIC_SRC" checkout 6bd8d55e1c4ad3188770fe5c8b93b942bcede4a2
```

For another checkout layout, only the source variables below need to change.

## Build

Enter the Isaac ROS environment first:

```bash
isaac-ros activate
```

Then run the following commands inside the shell opened by `isaac-ros`:

```bash
source /opt/ros/jazzy/setup.bash

export ISAAC_ROS_SRC="$HOME/workspace/isaac_ros_source"
export ARX_SIM_WS="$HOME/workspace/arx_r5_sim_ws"
export TOPIC_SRC="$ISAAC_ROS_SRC/topic_based_ros2_control"
export ARX_MOVEIT="$ISAAC_ROS_SRC/isaac_ros_manipulation/arx-r5-moveit"
export ARX_DESC="$ARX_MOVEIT/isaac_ros_manipulation_arx_r5a_robot_description"
export SIM_REPO="$ISAAC_ROS_SRC/arx-r5-isaac-sim"
export SIM_SRC="$SIM_REPO/arx_r5_isaac_sim_bringup"

mkdir -p "$ARX_SIM_WS"
cd "$ARX_SIM_WS"
colcon build \
  --symlink-install \
  --cmake-clean-cache \
  --base-paths "$TOPIC_SRC" "$ARX_DESC" "$SIM_SRC" \
  --packages-up-to arx_r5_isaac_sim_bringup \
  --cmake-args -DBUILD_TESTING=OFF

source install/setup.bash

ros2 pkg prefix topic_based_ros2_control
ros2 pkg prefix isaac_ros_manipulation_arx_r5a_robot_description
ros2 pkg prefix arx_r5_isaac_sim_bringup
```

A successful build ends with `Summary: 3 packages finished`. The deprecation
warning from `topic_based_ros2_control` and the setuptools message about
listing Git files are non-blocking. Keep `-DBUILD_TESTING=OFF`; otherwise that
upstream package requests the optional `ros_testing` dependency.
`--cmake-clean-cache` also removes cache left by a previous failed configure.
All three package-prefix checks should resolve under `$ARX_SIM_WS/install`.

## Run

Use two terminals. Start Isaac Sim first, wait for the robot articulation and
ROS topics to be created, and then start MoveIt/cuMotion. Both terminals must
use the same `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION`.

### Terminal 1: Isaac Sim in conda `isaaclab`

Open a **new host terminal**. Do not source `/opt/ros/jazzy`, and do not start
this process inside the Isaac ROS container: Isaac Sim 5.1 uses Python 3.11,
whereas system Jazzy can inject Python 3.12 modules into `PYTHONPATH`.

```bash
conda activate isaaclab

export ISAAC_ROS_SRC="$HOME/workspace/isaac_ros_source"
export SIM_REPO="$ISAAC_ROS_SRC/arx-r5-isaac-sim"
export ARX_MOVEIT="$ISAAC_ROS_SRC/isaac_ros_manipulation/arx-r5-moveit"
export ARX_DESC="$ARX_MOVEIT/isaac_ros_manipulation_arx_r5a_robot_description"
export ISAAC_SIM_PYTHON="$CONDA_PREFIX/bin/python"
export ARX_R5_DESCRIPTION_SHARE="$ARX_DESC"
export ROS_DOMAIN_ID=23
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

"$ISAAC_SIM_PYTHON" -c 'import isaacsim; print(isaacsim.__file__)'
cd "$SIM_REPO"
./scripts/run_isaac_sim.sh
```

The script locates the Isaac Sim installation from the selected Python and
adds its bundled Jazzy ROS bridge libraries automatically. A healthy startup
reports `/R5a/root_joint`, joints `joint1` through `joint8`, and the
`/isaac_joint_states` and `/isaac_joint_commands` topics.

For a headless session, add `--headless`:

```bash
./scripts/run_isaac_sim.sh --headless
```

### Terminal 2: cuMotion, MoveIt, ros2_control, and RViz

Open another terminal and enter the Isaac ROS environment:

```bash
isaac-ros activate
```

Then run inside that shell:

```bash
source /opt/ros/jazzy/setup.bash

export ARX_SIM_WS="$HOME/workspace/arx_r5_sim_ws"
export ROS_DOMAIN_ID=23
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

source "$ARX_SIM_WS/install/setup.bash"
ros2 launch arx_r5_isaac_sim_bringup \
  arx_r5a_isaac_sim.launch.py \
  start_cumotion:=True \
  start_rviz:=True
```

Wait until the joint-state broadcaster, manipulator controller, and gripper
controller are active and MoveIt reports that planning can begin.

## Plan and execute in RViz

In the **MotionPlanning** panel, select:

- Planning Group: `manipulator`
- Planning Pipeline: `isaac_ros_cumotion`
- Planner ID: `cuMotion`

Move the interactive end-effector marker to a reachable pose, click **Plan**,
then click **Execute**. The motion should appear in Isaac Sim while joint
states continuously return to MoveIt.

## Verify the ROS bridge and controllers

Run these commands in Terminal 2 while both launch processes are active:

```bash
export ISAAC_ROS_SRC="$HOME/workspace/isaac_ros_source"
export SIM_REPO="$ISAAC_ROS_SRC/arx-r5-isaac-sim"

"$SIM_REPO/scripts/check_ros_graph.sh"
ros2 topic hz /isaac_joint_states
ros2 topic hz /isaac_joint_commands
ros2 topic echo /isaac_joint_states --once
```

The check succeeds only when the expected topics and actions exist and these
controllers are `active`:

- `joint_state_broadcaster`
- `manipulator_controller`
- `gripper_controller`

The script is a read-only health check; it does not plan or move the robot.
`/isaac_joint_states` should publish continuously, while
`/isaac_joint_commands` normally has traffic only during trajectory
execution. No rate output before clicking **Execute** is therefore expected.

## Common messages

| Message or symptom | Meaning / action |
|---|---|
| `listing git files failed - pretending there aren't any` | A non-blocking setuptools file-listing message with external `--base-paths`; the package is fine if it finishes. |
| `TopicBasedSystem::on_init ... deprecated` | An upstream Jazzy compatibility warning that does not prevent execution. |
| `ros_testing` cannot be found | The full `-DBUILD_TESTING=OFF` argument did not reach CMake; rerun the complete build command without splitting a path across lines. |
| `PYTHONPATH contains Python 3.12` | The Isaac Sim terminal was polluted by system Jazzy. Open a clean terminal and activate only `conda isaaclab`. |
| `No 3D sensor plugin(s) defined for octomap updates` | Expected for the current stage because Nvblox/ESDF is not enabled. |

## Repository structure

```text
arx-r5-isaac-sim/
├── arx_r5_isaac_sim_bringup/  # ROS package, launch, control, and simulation code
├── scripts/                    # Isaac Sim launcher and ROS graph health check
├── pictures/                   # Project images
├── docs/                       # Architecture and model-validation notes
└── dependencies.repos          # Pinned external source dependencies
```

See [architecture.md](docs/architecture.md) for the interface design and
[model-validation.md](docs/model-validation.md) for the known geometry, TCP,
collision-mesh, and XRDF coverage limitations.

## Current scope

This release provides a fixed-base joint execution loop. By default,
`read_esdf_world=False` and `add_ground_plane=False`, so visible Isaac Sim
obstacles are not yet part of the cuMotion planning world. Cameras, Nvblox
ESDF, perception, and calibrated grasping are intentionally left for the next
integration stage.

This is a community integration, not an official ARX Robotics or NVIDIA
release. Original integration code is Apache-2.0 licensed; derived files keep
their upstream notices. See [NOTICE.md](NOTICE.md) for details.
