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

The Isaac articulation publishes all eight DOFs, and the TopicBasedSystem
hardware block contains all eight with position command and position/velocity
state interfaces. `joint1..joint7` are independent controller joints;
`joint8` declares `mimic=joint7` and `multiplier=1`. On write, the plugin emits
eight names and eight positions and overwrites the final position with the
`joint7` command. On read, it likewise mirrors the controller-visible state.
This explicitly drives both fingers under load while keeping `joint8` out of
the arm and gripper controller DOF sets, and avoids a malformed
eight-name/seven-position `JointState`.

## Runtime ownership

| Process | Owns | Reads | Writes |
|---|---|---|---|
| Isaac Sim 5.1 | Physics and articulation | `/isaac_joint_commands` | `/isaac_joint_states`, `/clock` |
| ros2_control | Trajectory interpolation | `/isaac_joint_states` | `/isaac_joint_commands`, controller actions |
| MoveIt | Robot state and execution | `/joint_states` | `FollowJointTrajectory` goals |
| cuMotion | GPU motion planning | URDF, XRDF, MoveIt request | Motion plan result |

The OmniGraph runs on `OnPhysicsStep`, so commands, joint states, and clock use
the fixed physics cadence rather than the display frame rate.

## Tabletop AprilTag workflow

The optional tabletop demos replace only the perception front end. Isaac Sim
publishes RGB-D and camera calibration topics; Isaac ROS Rectify and the
official GPU `nvidia::isaac_ros::apriltag::AprilTagNode` produce raw
detections. For authored cameras, a simulation-side refiner re-solves the
official CUDA corners from an exact-stamp rectified-image ROI with
cornerSubPix, CameraInfo, and IPPE_SQUARE, preserving IDs, corners, frames,
and timestamps. Unmatched or failed samples are dropped from the refined
stream; the unchanged official stream remains on `*_raw`. The ARX AprilTag
object server exposes the refined stream through the upstream `/get_objects`
and `/get_object_pose` action contracts. The destination tag is converted into a standard
`MultiObjectPickAndPlace` target pose.

The authored USD has two mutually exclusive perception profiles:

| Profile | Camera mount | RGB input | Official raw | Refined manipulation input |
|---|---|---|---|---|
| `zedx` | fixed `base_link -> zed_x_left_camera_optical_frame` (eye-to-hand) | `/zed_x/left/image_raw` | `/zed_x/tag_detections_raw` | `/zed_x/tag_detections` |
| `d455` | fixed `link6 -> d455_color_optical_frame` (eye-in-hand) | `/d455/color/image_raw` | `/d455/tag_detections_raw` | `/d455/tag_detections` |

The authored ZED X mount alone carries a local `+10 deg` Y rotation so its
view includes the placement area. `/R5a` remains level; the wrist-mounted D455
therefore receives no unintended global camera tilt.

The `run_zedx_sim.sh` and `run_d455_sim.sh` entry points select the `reachable`
layout. It exists only in the stage's anonymous session layer: the robot is
moved into the reachable work area and the ZED mount is counter-translated so
that its authored world pose and local `+10 deg` pitch remain unchanged. The
committed `assets/scenes/arx_sim.usd` is never rewritten. The same session
layer normalizes vertical camera aperture to the selected render aspect ratio,
which keeps CameraInfo square-pixel intrinsics (`fx` approximately equal to
`fy`) without persisting a camera edit.

For D455, `robot_state_publisher` supplies the time-varying
`base_link -> link6` transform from joint states. The ARX adapter looks up that
chain at every detection's original image timestamp, composes it with the fixed
`link6 -> d455_color_optical_frame` mount, and caches the object in
`base_link`. The destination-tag client independently uses the same exact-time
rule. Both requests wait asynchronously for delayed TF and discard a frame on
timeout; neither falls back to a latest transform. This is necessary because
the upstream `GetObjectPose` result contains a bare `Pose` without a frame or
timestamp. The ZED X path uses the same adapter contract, even though its
complete `base_link -> zed_x_left_camera_optical_frame` transform is fixed.

Exact-time processing is not continuous visual servoing. The perception path
filters and caches stable poses in `base_link`; once a pick-and-place goal has
started, cuMotion does not continuously replan the active trajectory from each
new camera frame.

The authored USD runtime is the sole publisher for its non-identity
`world -> base_link` and camera-mount transforms. The ROS motion launch disables
its legacy identity transform in these profiles, preventing two TF authorities.

From that boundary onward, the demo reuses NVIDIA's pick-and-place behavior
tree and public interfaces unchanged:

```text
Isaac ROS cuAprilTag raw detections
  -> IPPE pose refinement (authored-camera profiles)
  -> ARX object actions
  -> /multi_object_pick_and_place
  -> NVIDIA Multi-Object Pick-and-Place behavior tree
  -> /cumotion/motion_plan
  -> MoveIt /execute_trajectory
  -> FollowJointTrajectory / GripperCommand
  -> /isaac_joint_commands
  -> Isaac Sim articulation
```

Simulation-only YAML selects one reachable top grasp and shorter tabletop
approach/retract offsets. It does not fork the upstream behavior-tree code or
change the real-robot grasp configuration.

For manual RViz planning, MoveIt can use the registered cuMotion planning
plugin. In the automatic pick-and-place tree, the planning nodes call the
standalone `/cumotion/motion_plan` action directly; MoveIt then receives the
returned `RobotTrajectory` through `/execute_trajectory` and forwards it to the
controllers.

## Static planning scene

The authored-camera workflow loads `authored_usd_table.scene` through NVIDIA's
Static Planning Scene Server. It contains five collision objects: the tabletop
and four legs. Isaac ROS 4.5 assigns `world` to every parsed object header, but
the numeric poses in this integration's file are intentionally expressed in
`base_link`. The topics are therefore separated explicitly:

```text
NVIDIA Static Planning Scene Server
  -> /cumotion/static_planning_scene_raw  (five objects, raw world headers)
  -> planning_scene_frame_adapter         (header normalization only)
  -> /planning_scene                      (five objects, base_link, canonical)
  -> MoveIt and the pick-and-place workflow
```

The adapter does not apply a second geometric transform; it corrects the
parser-assigned headers while preserving the `base_link` numeric poses. The
private raw topic prevents either consumer from accidentally using the
mislabelled scene.

`read_esdf_world=False` and `add_ground_plane=False` still mean there is no
dynamic Nvblox ESDF, separate ground plane, or collision representation for
other visible Isaac Sim obstacles. The five static table objects are an
environment-aware planning input, but not a complete digital twin.

## Simulated grasp attachment

For the authored USD, the red block begins as a collidable kinematic rigid
body. The `joint7` linear drive has a finite maximum force of `8 N` by default;
`--gripper-drive-max-force` changes that limit. TopicBasedSystem mirrors the
`joint7` position target into the `joint8` transport slot, so both articulation
fingers are actively driven even if their raw PhysX states diverge under load.
`joint8` is still not exposed as a second gripper-controller DOF.

Attachment requires the commanded aperture to be below the close threshold,
the block to be within the configured `grasp_frame` distance, and simultaneous
PhysX contact between the block and both finger bodies, `link7` and `link8`.
All conditions must remain true for consecutive physics steps before the
runtime switches the block to dynamic, enables CCD, and creates a FixedJoint to
`link6`. The default is `3` steps and `--grasp-contact-steps` configures it.
Opening the gripper removes the joint and clears the contact gate. The
FixedJoint exists only to stabilize transport after verified bilateral
contact; this is not a grasp formed purely by contact forces and friction.
cuMotion Object Attachment independently maintains the attached collision
object in its planning scene.

The generated fixed-`Camera_1` regression scene retains its older visual
attachment controller. Its recorded `workflow_status=1` and placement metric
must not be presented as final acceptance of either authored-camera profile.

## Startup ordering

1. Isaac Sim imports the fixed-base articulation and starts publishing state.
2. `ros2_control_node` starts with `TopicBasedSystem`.
3. After three seconds, the joint-state broadcaster activates.
4. After five seconds, arm and gripper controllers activate.
5. After six seconds, `move_group` starts with a fresh simulated state.

The timers avoid controller activation failure before the first
`/isaac_joint_states` message, matching the NVIDIA Isaac ROS 4.5 example.
