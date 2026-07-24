# ARX R5A cuMotion Simulation for Isaac Sim

[中文](README.md) · [AprilTag pick-and-place demo](docs/apriltag-pick-place-demo.md) ·
[Architecture](docs/architecture.md) · [Model validation](docs/model-validation.md)

[![CI](https://github.com/wee733/arx-r5-isaac-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/wee733/arx-r5-isaac-sim/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

<p align="center">
  <img src="pictures/arx.png" alt="ARX R5A planning and execution in RViz and Isaac Sim" width="100%">
</p>

This repository is the simulation companion to
[`Isaac_Ros_CuMotion_ArxR5a`](https://github.com/wee733/Isaac_Ros_CuMotion_ArxR5a).
It connects an ARX R5A articulation in Isaac Sim 5.1 to MoveIt 2 and NVIDIA
cuMotion through `ros2_control`. RViz uses the MoveIt cuMotion planning
plugin; the automatic pick-and-place tree calls the cuMotion action directly
and uses MoveIt for trajectory execution:

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
The generated fixed-`Camera_1` AprilTag regression scene also completed
detection, planning, grasp, transport, and placement with
`workflow_status=1`. After placement, the estimated cube center was
`(0.29907, 0.17823, 0.02520) m`, about `2.0 mm` from the destination tag in the
table plane. That metric belongs only to the generated regression scene.
Camera publishing, TF, and AprilTag perception are verified on the original
authored USD, with the ZED X in front of ARX. However, on 2026-07-23 its source
pose was approximately `base_link (0.892, -0.180, -0.173) m`, and cuMotion
returned `INVERSE_KINEMATICS_FAILURE`. Neither authored ZED X nor D455 can
currently be claimed as an end-to-end pick-and-place pass.

The tested baseline is Ubuntu 24.04, ROS 2 Jazzy, Isaac ROS 4.5, cuMotion 4.5,
and Isaac Sim 5.1.0 installed in a conda environment named `isaaclab`.

`joint1` through `joint6` form the `manipulator` planning group. `joint7` is
the commanded gripper-controller joint. `joint8` is not an independent
controller DOF; TopicBasedSystem mirrors the `joint7` position into the eighth
transport command so both articulation fingers receive a drive target.

## Repository responsibilities

| Repository | Responsibility |
|---|---|
| NVIDIA `isaac_ros_manipulation` | Upstream messages/actions, object selection, the Multi-Object Pick-and-Place behavior-tree implementation, cuMotion/MoveIt integration, and reusable manipulation servers. |
| `isaac_ros_manipulation_arx_r5a` | Connects the upstream workflow to ARX R5A bringup and provides the AprilTag object server and robot adapters. Its model dependency comes from [`Isaac_Ros_CuMotion_ArxR5a`](https://github.com/wee733/Isaac_Ros_CuMotion_ArxR5a). |
| This `arx-r5-isaac-sim` repository | Owns only the Isaac Sim workcell, RGB-D camera, ROS 2 bridge, TopicBasedSystem loop, and simulation-specific parameters that must not alter the real-robot configuration. |

URDF, SRDF, XRDF, meshes, and MoveIt configuration remain in the ARX model
repository. The upstream behavior-tree implementation, standard action types,
and cuMotion interfaces are unchanged; this repository only overrides grasp
and approach/retract parameters for the tabletop geometry.

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
│   ├── install/                         # NVIDIA Isaac ROS manipulation overlay
│   ├── arx-r5-isaac-sim/
│   ├── isaac_ros_manipulation_arx_r5a/ # ARX manipulation overlay
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
ARX adapter, model, and TopicBasedSystem revisions used for the verified run:

```bash
export ISAAC_ROS_SRC="$HOME/workspace/isaac_ros_source"
export SIM_REPO="$ISAAC_ROS_SRC/arx-r5-isaac-sim"
export ARX_MANIP_REPO="$ISAAC_ROS_SRC/isaac_ros_manipulation_arx_r5a"
export ARX_REPO="$ISAAC_ROS_SRC/isaac_ros_manipulation/arx-r5-moveit"
export TOPIC_SRC="$ISAAC_ROS_SRC/topic_based_ros2_control"

mkdir -p "$ISAAC_ROS_SRC/isaac_ros_manipulation"
[[ -d "$SIM_REPO/.git" ]] || \
  git clone https://github.com/wee733/arx-r5-isaac-sim.git "$SIM_REPO"
[[ -d "$ARX_MANIP_REPO/.git" ]] || \
  git clone https://github.com/wee733/isaac_ros_manipulation_arx_r5a.git "$ARX_MANIP_REPO"
[[ -d "$ARX_REPO/.git" ]] || \
  git clone https://github.com/wee733/Isaac_Ros_CuMotion_ArxR5a.git "$ARX_REPO"
[[ -d "$TOPIC_SRC/.git" ]] || \
  git clone https://github.com/PickNikRobotics/topic_based_ros2_control.git "$TOPIC_SRC"

git -C "$ARX_MANIP_REPO" checkout 37ccc2dfa928a1bb66da6b83934c351990734dbe
git -C "$ARX_REPO" checkout v0.3.0
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
export ARX_MANIP_REPO="$ISAAC_ROS_SRC/isaac_ros_manipulation_arx_r5a"
export ARX_SIM_WS="$HOME/workspace/arx_r5_sim_ws"
export TOPIC_SRC="$ISAAC_ROS_SRC/topic_based_ros2_control"
export ARX_MOVEIT="$ISAAC_ROS_SRC/isaac_ros_manipulation/arx-r5-moveit"
export ARX_DESC="$ARX_MOVEIT/isaac_ros_manipulation_arx_r5a_robot_description"
export SIM_REPO="$ISAAC_ROS_SRC/arx-r5-isaac-sim"
export SIM_SRC="$SIM_REPO/arx_r5_isaac_sim_bringup"

# The upstream Isaac ROS manipulation workspace must already be built.
source "$ISAAC_ROS_SRC/install/setup.bash"

cd "$ARX_MANIP_REPO"
colcon build \
  --symlink-install \
  --packages-up-to isaac_ros_manipulation_arx_r5a_bringup
source install/setup.bash

mkdir -p "$ARX_SIM_WS"
cd "$ARX_SIM_WS"
colcon build \
  --symlink-install \
  --cmake-clean-cache \
  --allow-overriding isaac_ros_manipulation_arx_r5a_robot_description \
  --base-paths "$TOPIC_SRC" "$ARX_DESC" "$SIM_SRC" \
  --packages-up-to arx_r5_isaac_sim_bringup \
  --cmake-args -DBUILD_TESTING=OFF

source install/setup.bash

ros2 pkg prefix isaac_ros_manipulation_arx_r5a_bringup
ros2 pkg prefix topic_based_ros2_control
ros2 pkg prefix isaac_ros_manipulation_arx_r5a_robot_description
ros2 pkg prefix arx_r5_isaac_sim_bringup
```

The adapter build should finish two packages, and the simulation build should
finish three packages. The deprecation
warning from `topic_based_ros2_control` and the setuptools message about
listing Git files are non-blocking. Keep `-DBUILD_TESTING=OFF`; otherwise that
upstream package requests the optional `ros_testing` dependency.
`--cmake-clean-cache` also removes cache left by a previous failed configure.
`--allow-overriding` explicitly selects the local description pinned to
`v0.3.0` over any package with the same name in the Isaac ROS underlay.
The ARX bringup prefix should resolve under `$ARX_MANIP_REPO/install`; the
other three package-prefix checks should resolve under `$ARX_SIM_WS/install`.

### AprilTag demo runtime dependencies

Install these packages for the Jazzy/Isaac ROS shell. They do not belong in
the conda `isaaclab` environment:

```bash
sudo apt update
sudo apt install \
  ros-jazzy-isaac-ros-apriltag \
  ros-jazzy-isaac-ros-cumotion-object-attachment \
  ros-jazzy-py-trees \
  ros-jazzy-py-trees-ros
```

An unpacked ROS overlay under `/tmp` was used during dependency diagnosis.
Such an overlay has no durable setup contract and may disappear after a reboot
or cleanup; it is not a reproducible installation. Use the packages above or
an equivalent source-built workspace, then validate from a fresh shell.

## Run

Use two terminals. Start Isaac Sim first, wait for the robot articulation and
ROS topics to be created, and then start MoveIt/cuMotion. Both terminals must
use `ROS_DOMAIN_ID=25` and `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` unless both
values are deliberately overridden together.

### Terminal 1: Isaac Sim in conda `isaaclab`

Open a **new host terminal**, activate only conda `isaaclab`, and run only
Isaac Sim there. Do not source `/opt/ros/jazzy`, and do not start this process
inside the Isaac ROS shell: Isaac Sim 5.1 uses Python 3.11, whereas system
Jazzy can inject Python 3.12 modules into `PYTHONPATH`.

If `~/.bashrc` sources ROS automatically, `conda activate` does not undo those
exports. Start a shell without startup files first:

```bash
bash --noprofile --norc
source /home/lbz/miniforge3/etc/profile.d/conda.sh
conda activate isaaclab
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION LD_LIBRARY_PATH
```

```bash
conda activate isaaclab

export ISAAC_ROS_SRC="$HOME/workspace/isaac_ros_source"
export SIM_REPO="$ISAAC_ROS_SRC/arx-r5-isaac-sim"
export ARX_MOVEIT="$ISAAC_ROS_SRC/isaac_ros_manipulation/arx-r5-moveit"
export ARX_DESC="$ARX_MOVEIT/isaac_ros_manipulation_arx_r5a_robot_description"
export ISAAC_SIM_PYTHON="$CONDA_PREFIX/bin/python"
export ARX_R5_DESCRIPTION_SHARE="$ARX_DESC"
export ROS_DOMAIN_ID=25
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

Open another terminal, do not activate conda `isaaclab`, and enter the Isaac
ROS environment:

```bash
isaac-ros activate
```

Then run inside that shell:

```bash
source /opt/ros/jazzy/setup.bash

export ARX_SIM_WS="$HOME/workspace/arx_r5_sim_ws"
export ROS_DOMAIN_ID=25
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

## Authored-USD dual-camera manipulation demo

The committed `assets/scenes/arx_sim.usd` provides two mutually exclusive
perception entries. Both reuse the same Isaac ROS manipulation servers,
upstream Multi-Object Pick-and-Place behavior tree, cuMotion, MoveIt, and
ros2_control after perception.

The LFS asset is a byte-for-byte copy of the original authored USD (SHA-256
`7162ec49dedd9dabe7748a9fd1da16d8b2164cf385c297ac8c94797efd61f1c0`). It is
treated as a read-only source layer; every runtime patch is authored into an
anonymous session layer.

| Mode | Mount | Image | cuAprilTag raw | Manipulation input |
|---|---|---|---|---|
| ZED X eye-to-hand | Fixed to authored `base_link` (in front of ARX) | `/zed_x/left/image_raw` | `/zed_x/tag_detections_raw` | `/zed_x/tag_detections` |
| D455 eye-in-hand | Fixed to the moving `link6` | `/d455/color/image_raw` | `/d455/tag_detections_raw` | `/d455/tag_detections` |

The ZED X position and orientation come exactly from the USD scene you authored
in Isaac Sim. It is in front of the ARX workcell; the simulator never moves the
camera behind the arm or repositions the robot to manufacture a "reachable"
layout.

Both modes use the official
`nvidia::isaac_ros::apriltag::AprilTagNode` with the CUDA/cuAprilTag backend.
Its original IDs, corners, and poses remain available on the profile's
`*_raw` topic. Small planar tags can produce an ambiguous low-pixel PnP pose,
so a simulation-side pose refiner re-solves the official corners with the
exact-stamp rectified image, CameraInfo, `cornerSubPix`, and OpenCV
IPPE_SQUARE. It converts only a small ROI around the four corners, preserves
the ID, corners, frame, and original timestamp, and publishes the result on
the non-raw manipulation topic. A missing/invalid matching image, failed
solve, or reprojection error above `2 px` drops that detection from the
refined stream instead of interleaving a native pose. The official output
remains intact on `*_raw`. The downstream ARX object adapter then uses that
unchanged image timestamp to transform the pose into `base_link`.

Both simulator wrappers select `--authored-layout as-authored` by default. This
profile has no pose overrides, so `/R5a`, ZED, D455, objects, targets, and any
other authored environment geometry remain exactly where you placed them.
Runtime changes are confined to the anonymous USD session layer: physics,
materials, collision settings, and camera projection attributes needed by the
ROS bridge. They never modify `assets/scenes/arx_sim.usd`. Camera vertical
aperture is normalized in that session layer to match the selected output size.

The authored profiles load the tabletop and four legs as five static collision
objects through the official cuMotion Static Planning Scene Server. Isaac ROS
4.5's `.scene` parser hard-codes their headers to `world`, although this file's
numeric poses are intentionally expressed in `base_link`. The raw message is
therefore isolated on `/cumotion/static_planning_scene_raw`; the ARX frame
adapter publishes the normalized canonical scene on `/planning_scene`, which
is the topic consumed by MoveIt and the behavior tree.

The placement goal is not a hard-coded world pose. At each tag 1 detection's
own timestamp, the goal client obtains the camera TF and applies the
tag-relative `link6_offset_in_tag: [0, 0, -0.315]`. Of that distance, `205 mm`
is the top-grasp distance from `link6` to the block center, `75 mm` is the block
half-height, and the remaining `35 mm` is clearance above the detected plane.
The diagnostic sequence was `-0.280 m`/`0 mm`, then `-0.295 m`/`15 mm` which
still failed, and finally the passing `-0.315 m`/`35 mm`. The explicit
engineering budget is the Object Attachment `max_overshoot` of `10 mm`, the
XRDF attached-object buffer of `2 mm`, a `10 mm` PnP depth-error allowance,
and a `5 mm` safety margin: `27 mm` total with `8 mm` remaining. A lower goal
can therefore have a valid kinematic IK solution while cuMotion correctly
returns `INVERSE_KINEMATICS_FAILURE` for collision with the static tabletop.

Tag 1 must provide five consecutive exact-time samples transformed into
`base_link`. The client takes a component-wise median and requires the maximum
pairwise Euclidean spread across the full window to be at most `10 mm`, so a
`20 mm` depth swing is rejected. Target loss or a simulation-clock rewind
clears the window. `drop_pose_ttl_sec` defaults to `0.5 s`,
`drop_pose_max_translation_spread_m` to `0.01 m`, and the object-server cache
TTL is `2.0 s`.

Because the upstream tree continuously calls `/get_objects`, authored mode
admits discovery only inside the `base_link` source zone
`[0.20, -0.30, -0.39] .. [0.40, -0.08, -0.25] m`. This prevents a placed block
from starting a second cycle. It gates only `/get_objects`; it does not alter
the active task's `/get_object_pose` cache or freshness semantics.

### ZED X: fixed eye-to-hand

Terminal 1, in a clean conda `isaaclab` environment:

```bash
conda activate isaaclab
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "$HOME/workspace/isaac_ros_source/arx-r5-isaac-sim"
./scripts/run_zedx_sim.sh
```

Terminal 2, inside the `(isaac-ros)` shell; disable automatic motion on the
first run:

```bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "$HOME/workspace/isaac_ros_source/arx-r5-isaac-sim"
./scripts/run_zedx_demo.sh auto_start:=False
```

Read-only acceptance check:

```bash
./scripts/check_ros_graph.sh --zedx
```

The original authored USD has verified ZED X image, CameraInfo, TF, and official
CUDA cuAprilTag perception, with the camera remaining in front of ARX. Its
source is currently at approximately
`base_link (0.892, -0.180, -0.173) m`; on 2026-07-23 cuMotion returned
`INVERSE_KINEMATICS_FAILURE` before contact, attachment, transport, or
placement. The old reachable-layout `6.7 mm` and `status=1` results therefore
do not qualify the current original USD.

### D455: wrist-mounted eye-in-hand

Stop the previous Isaac Sim process first so exactly one node publishes
`/clock`. Terminal 1:

```bash
conda activate isaaclab
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "$HOME/workspace/isaac_ros_source/arx-r5-isaac-sim"
./scripts/run_d455_sim.sh
```

Terminal 2, inside the `(isaac-ros)` shell:

```bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "$HOME/workspace/isaac_ros_source/arx-r5-isaac-sim"
./scripts/run_d455_demo.sh auto_start:=False
```

`run_d455_sim.sh` initializes the arm at a retained D455 startup joint seed:
`[-1.3919817209, 1.8859872818, 0.5746622086, -0.6957563162, -0.6273930669, -1.9993159771, 0.044, 0.044]`,
ordered as `joint1..joint8`. It keeps the Isaac Sim and ros2_control startup
states consistent; the original authored USD no longer claims the old
`x=960/325`, `0.74 s`, or complete pick-and-place acceptance results. Its
source is outside the configured arm workspace and cuMotion returns
`INVERSE_KINEMATICS_FAILURE`. ros2_control uses the corresponding eight startup
values, with `joint8` equal to `joint7`.
TopicBasedSystem continues to mirror the `joint7` command into `joint8`; no
controller exposes it as a second gripper DOF.
`link6 -> d455_color_optical_frame` is a fixed mount edge,
while `base_link -> link6` changes with `/joint_states`. Both the source object
and destination tag asynchronously wait for the exact-time TF at each
detection's original timestamp; neither falls back to the latest wrist pose.

This per-frame dynamic TF lookup provides timestamp-correct perception; it is
not continuous visual servoing. The adapter filters and caches a stable pose in
`base_link`. Once the behavior tree starts, cuMotion plans and executes against
that pose rather than continuously replanning from every camera frame while
the arm moves.

Read-only acceptance check:

```bash
./scripts/check_ros_graph.sh --d455
```

After perception, TF, controllers, and planning services pass inspection,
remove `auto_start:=False` to send one automatic pick-and-place task. Do not
mix the ZED simulator command with the D455 ROS script or start both
manipulation graphs at once.

Authored demos default to `quiesce_on_terminal:=True`. Once the official action
reaches a terminal result, the wrapper pauses only the behavior-tree timer;
the action server and executor remain alive so the result is delivered, while
the tree no longer repeats discovery at 100 Hz. Set
`quiesce_on_terminal:=False` to retain upstream continuous ticking, or restart
the ROS demo for another one-shot task.

## Generated fixed-camera AprilTag tabletop demo

The complete demo uses `tag36h11:0` on a red cube as the source object and
places it on table tag `tag36h11:1`. Start the workcell in the conda
`isaaclab` terminal:

```text
Isaac Sim RGB + CameraInfo
  -> Isaac ROS Rectify + GPU AprilTag
  -> official /tag_detections output (unchanged)
  -> ARX AprilTag Object Server
  -> upstream Multi-Object Pick-and-Place action contract
  -> cuMotion action -> MoveIt ExecuteTrajectory -> ros2_control
  -> /isaac_joint_commands -> Isaac Sim
```

```bash
./scripts/run_isaac_sim.sh --scene tabletop
```

Then start perception, the behavior tree, cuMotion, MoveIt, and the controller
chain in the `(isaac-ros)` terminal:

```bash
./scripts/run_apriltag_demo.sh
```

The default launch automatically sends one pick-and-place goal. Use
`auto_start:=False` for perception-only inspection. See the
[AprilTag demo runbook](docs/apriltag-pick-place-demo.md) for dependencies,
workspace setup, verification, and current physics limitations.

The validated simulation uses a `0.14 m` top-grasp seed, `65 mm`
approach/retract distances, and a `0.165 m` drop offset for `link6`. With the
current approximately `157.6 mm` fingertip reach, this leaves about `7.6 mm`
of nominal clearance above the table without changing the real-robot grasp
configuration.

For a first run, use `auto_start:=False` and verify both tags, `/get_objects`,
`/get_object_pose`, `/arx_r5_demo/drop_pose`, and all three controllers before
sending the workflow manually. Full acceptance requires
`Successfully loaded 1 grasp poses`, the simulated cube attach/release logs,
and `workflow_status: 1`; the runbook contains the exact commands. Launch also
cross-checks the tabletop YAML against the ARX Tag map and camera calibration,
so a partial custom configuration fails explicitly instead of silently using
inconsistent TF or object dimensions.

## Verify the ROS bridge and controllers

Run these commands in Terminal 2 while both launch processes are active:

```bash
export ISAAC_ROS_SRC="$HOME/workspace/isaac_ros_source"
export SIM_REPO="$ISAAC_ROS_SRC/arx-r5-isaac-sim"
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

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
| `PYTHONPATH contains Python 3.12` | The Isaac Sim terminal was polluted by system Jazzy. Open a clean terminal and activate only `conda isaaclab`; if `~/.bashrc` sources `/opt/ros/jazzy/setup.bash`, move that command to the ROS terminal instead. |
| `LD_LIBRARY_PATH contains a ROS installation` | The Isaac Sim terminal inherited ROS shared-library paths. In a clean Isaac Sim shell, run `unset LD_LIBRARY_PATH` before starting the scene. |
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

This release provides a fixed-base execution loop, a fixed ZED X eye-to-hand
profile, a wrist-mounted D455 eye-in-hand profile, and the generated fixed
camera regression scene. The two authored-camera profiles are mutually
exclusive. Their static `.scene` adds the tabletop and four legs as five
objects on canonical `/planning_scene`. Because `read_esdf_world=False` and
`add_ground_plane=False`, there is still no dynamic Nvblox ESDF, separate
ground plane, or coverage for other visible scene obstacles.

AprilTag replaces the SAM/FoundationPose perception front end, while object
selection, behavior-tree actions, cuMotion, MoveIt, and ros2_control retain the
same interface chain. In authored profiles, the source object pose comes from
bounded IPPE refinement of the official cuAprilTag corners, exact-time TF, and
the configured tag-to-object transform; the official native result remains on
`*_raw`. The destination-tag client turns its detection into a task-specific
drop pose with TF from the same image timestamp. Its `35 mm` placement
clearance is also defined along the detected tag normal, not as a hard-coded
world destination. These are simulation pose/task adapters, not a new tag
detector. The D455 path is timestamp-correct but is not continuous visual
servoing.

In the authored USD, the red block waits as a collidable kinematic rigid body.
The `joint7` linear drive has a finite maximum force of `8 N` by default,
adjustable with `--gripper-drive-max-force`. TopicBasedSystem sends the same
position target to `joint7` and `joint8`, avoiding load-induced finger
divergence when relying only on the PhysX mimic; `joint8` remains outside the
controller DOF set. Attachment requires all three conditions:
the gripper aperture and `grasp_frame` distance are within their thresholds,
and the block maintains bilateral PhysX contact with `link7` and `link8` for
consecutive physics steps. The default is `3` steps and is adjustable with
`--grasp-contact-steps`. The block then becomes dynamic and a FixedJoint to
`link6` stabilizes transport; opening the gripper deletes the joint. This is
not a purely friction-generated grasp. cuMotion Object Attachment separately
maintains the planning-scene attachment. The generated `Camera_1` regression
scene keeps its older visual attachment. Its `2.0 mm` result and
`workflow_status=1`, plus its `7.6 mm` nominal clearance, do not qualify the
authored profiles. ZED X and D455 have
verified camera, TF, and AprilTag perception, but the original source at
`base_link (0.892, -0.180, -0.173) m` produced
`INVERSE_KINEMATICS_FAILURE` on 2026-07-23, so authored end-to-end acceptance
remains incomplete. Nvblox/ESDF, calibrated physical
grasping, real TCP/camera extrinsics, collision geometry, and emergency-stop
validation remain required before real hardware use. Simulation success does
not guarantee hardware success.

This is a community integration, not an official ARX Robotics or NVIDIA
release. Original integration code is Apache-2.0 licensed; derived files keep
their upstream notices. See [NOTICE.md](NOTICE.md) for details.
