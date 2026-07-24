# ARX R5A AprilTag 桌面抓放 Demo

本仓库提供三种共用同一 manipulation 后半段的 AprilTag 入口：

- authored USD 中固定在 `base_link` 的 ZED X，属于 eye-to-hand；
- authored USD 中固定在 `link6` 的 D455，属于 eye-in-hand；
- 程序生成的固定 `Camera_1`，保留为原始回归场景。

它们都观察两个 `tag36h11` 标签：

- ID `0` 贴在红色物体上，是待抓取物体；程序生成场景使用 50 mm 方块，authored
  USD 使用 `50 × 50 × 150 mm` 红块；
- ID `1` 贴在桌面另一侧，是投放目标。

下面的 `2.0 mm` 结果特指程序生成的固定相机回归场景：完整流程已经端到端跑通并
返回 `workflow_status=1`。投放后由 tag 0 计算出的
方块中心为 `(0.29907, 0.17823, 0.02520) m`，与 tag 1 目标中心的平面距离约为
`2.0 mm`，这组指标不能用于 authored 模式。2026-07-23 使用原始 authored 布局时，
source 在 `base_link` 中约为 `(0.892, -0.180, -0.173) m`，cuMotion 返回
`INVERSE_KINEMATICS_FAILURE`。因此目前只能宣称 ZED X 位于 ARX 前方且两套相机的
感知链已验证，不能宣称 authored ZED X 或 D455 已完成端到端抓放。

本文档的已验证通信配置为：

```bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

两个终端必须使用相同的值。

## 三个仓库分别负责什么

| 仓库 | 责任边界 |
|---|---|
| NVIDIA `isaac_ros_manipulation` | 官方消息/action、对象选择、Multi-Object Pick-and-Place 行为树实现、cuMotion/MoveIt 接口和通用 manipulation server。 |
| `isaac_ros_manipulation_arx_r5a` | 把官方工作流接到 ARX R5A，提供 ARX bringup、AprilTag object server 和实机侧适配。 |
| 本仓库 `arx-r5-isaac-sim` | 只维护 Isaac Sim 场景、相机、ROS 2 Bridge、仿真控制闭环，以及不应进入实机仓库的仿真专用抓取和路径参数。 |

ARX 的 URDF、SRDF、XRDF、网格和 MoveIt 配置来自
`Isaac_Ros_CuMotion_ArxR5a` 的 `arx-r5-moveit` checkout。官方 `py_trees`
实现、标准 action 名称和行为树语义保持不变；本仓库只覆盖桌面场景所需的抓取 seed、
approach/retract 距离和投放末端位姿。

## 接口链

```text
Isaac Sim
  /clock
  /isaac_joint_states
  RGB + Depth + CameraInfo
    -> Isaac ROS Rectify + GPU AprilTag
    -> authored: *_raw 官方检测
    -> authored: IPPE pose refiner -> /zed_x/tag_detections 或 /d455/tag_detections
    -> generated: /tag_detections
    -> ARX AprilTag Object Server
         /get_objects
         /get_object_pose
    -> /multi_object_pick_and_place
    -> NVIDIA Multi-Object Pick-and-Place py_trees
    -> /cumotion/motion_plan
    -> MoveIt /execute_trajectory
    -> ros2_control FollowJointTrajectory + GripperCommand
    -> /isaac_joint_commands
    -> Isaac Sim articulation
```

tag 1 的检测由目标客户端转换成 `/arx_r5_demo/drop_pose`，随后作为标准
`MultiObjectPickAndPlace` goal 的 `target_poses`。因此 AprilTag 只替代
SAM/FoundationPose 等感知前端，后面的对象选择、行为树 action、cuMotion、MoveIt、
ros2_control 和 Isaac Sim 控制话题仍走标准接口。

## 环境必须分成两个终端

| 终端 | 环境 | 运行内容 |
|---|---|---|
| 终端 1 | 宿主机 Conda `isaaclab`，Python 3.11 | 只运行 Isaac Sim 5.1 和其内置 ROS 2 Bridge。不要 source Jazzy，也不要进入 `isaac-ros`。 |
| 终端 2 | `isaac-ros activate` 后的 Jazzy/Isaac ROS shell，Python 3.12 | 运行 AprilTag、行为树、cuMotion、MoveIt、ros2_control 和 RViz。不要激活 Conda `isaaclab`。 |

把 ROS 的 Python 3.12 模块带入 Isaac Sim Python 3.11，通常会出现
`PYTHONPATH contains Python 3.12`、`rclpy` ABI 或动态库冲突。

## Authored USD 双相机快速启动

两种 authored USD 模式一次只能运行一套。Isaac Sim 和 ROS 终端必须选择相同相机，
并保持相同的 `ROS_DOMAIN_ID` 与 `RMW_IMPLEMENTATION`。

两个仿真入口默认使用 `--authored-layout as-authored`。该 profile 不包含任何位姿覆盖，
所以 `/R5a`、ZED、D455、物体、目标和你在 Isaac Sim 中设计的环境都保持原样。运行时
只在匿名 USD session layer 中添加必要的物理、材质、碰撞和相机投影属性；退出 Isaac
Sim 后这些修改消失，源文件 `assets/scenes/arx_sim.usd` 不会被覆盖。

authored USD 的相机输出为 16:10。启动时会在同一个非持久 session layer 中按
`vertical_aperture = horizontal_aperture × height / width` 规范相机投影，避免用
16:9 aperture 发布 16:10 CameraInfo。验收时 `fx` 应约等于 `fy`；这项修正不会写回
用户设计的 USD。

### ZED X eye-to-hand

终端 1（干净的 Conda `isaaclab`，不要 source Jazzy）：

```bash
conda activate isaaclab
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "$HOME/workspace/isaac_ros_source/arx-r5-isaac-sim"
./scripts/run_zedx_sim.sh
```

终端 2（`isaac-ros activate` 后的 shell，不要激活 Conda）：

```bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "$HOME/workspace/isaac_ros_source/arx-r5-isaac-sim"
./scripts/run_zedx_demo.sh auto_start:=False
```

ZED X 左目输出 `/zed_x/left/image_raw` 和 `/zed_x/left/camera_info`；官方
Isaac ROS AprilTag 输出 `/zed_x/tag_detections_raw`，pose refiner 输出供 manipulation
消费的 `/zed_x/tag_detections`。相机安装 TF 是固定的
`base_link -> zed_x_left_camera_optical_frame`。相机位于你 authored USD 设计的 ARX
前方。运行时外参从 USD 层级变换读取并校验，
不会在 ROS launch 中再手写或叠加任何角度。

原始 authored USD 已验证 ZED X 图像、CameraInfo、TF 和官方 CUDA cuAprilTag 感知；
当前 source 位于 `base_link ≈ (0.892, -0.180, -0.173) m`，2026-07-23 的 cuMotion
规划返回 `INVERSE_KINEMATICS_FAILURE`，尚未进入接触、attach、搬运和投放阶段。旧
reachable 布局的 `6.7 mm` 和 `status=1` 结果不属于当前原始 USD。

### D455 eye-in-hand

先停止 ZED X 模式，确认系统中不存在旧 `/clock` publisher。终端 1：

```bash
conda activate isaaclab
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "$HOME/workspace/isaac_ros_source/arx-r5-isaac-sim"
./scripts/run_d455_sim.sh
```

终端 2：

```bash
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "$HOME/workspace/isaac_ros_source/arx-r5-isaac-sim"
./scripts/run_d455_demo.sh auto_start:=False
```

D455 输出 `/d455/color/image_raw` 和 `/d455/color/camera_info`；官方 Isaac ROS
AprilTag 输出 `/d455/tag_detections_raw`，refiner 输出 `/d455/tag_detections`。安装边
`link6 -> d455_color_optical_frame` 固定，`base_link -> link6` 则由 URDF、
`/joint_states` 和 `robot_state_publisher` 随关节实时更新。ARX object adapter 使用
检测消息原始时间戳转换到 `base_link`，避免把腕部相机的旧图像配上最新关节 TF。
启动脚本使用保留的 D455 启动关节位
`[-1.3919817209, 1.8859872818, 0.5746622086, -0.6957563162, -0.6273930669, -1.9993159771, 0.044, 0.044]`，
依次对应 `joint1..joint8`，用于让 Isaac Sim 与
ros2_control hardware 的全部八个关节使用一致初值；`joint8` 初值等于 `joint7`，之后由
TopicBasedSystem 的 `mimic=joint7, multiplier=1` 生成第八个位置命令，而不是独立的
controller DOF。源物体和目标 Tag 都使用异步 exact-time TF 查询，允许渲染帧先于对应动态
TF 到达，但绝不回退到 latest TF。

该启动位不再作为原始 authored USD 的感知或规划验收声明；旧文档中的
`x=960/325`、`0.74 s` 和 `workflow_status=1` 来自已废弃的 reachable 布局。当前原始
布局 source 超出配置的机械臂工作域，cuMotion 返回 `INVERSE_KINEMATICS_FAILURE`。

“逐帧动态 TF”表示每条检测都使用它自己的采集时间，不表示 continuous visual
servo。object adapter 会在 `base_link` 中过滤并缓存稳定目标；行为树开始后，cuMotion
对该目标执行一次规划，不会在机械臂运动过程中根据每个新图像帧持续更新轨迹。

tag 1 的投放目标使用 `5` 个连续、按各自检测时间戳转换到 `base_link` 的样本。目标
客户端对 x/y/z 分量分别取中值，并要求完整窗口内任意两点的最大欧氏距离不超过
`10 mm`；因此 `20 mm` 深度摆动会被拒绝。目标丢失或 `/clock` 回退会清空窗口，不能
把不连续观察拼成一次稳定判定。`drop_pose_ttl_sec` 默认 `0.5 s`，
`drop_pose_max_translation_spread_m` 默认 `0.01 m`。

官方行为树会持续 discovery。authored 配置因此把 `/get_objects` 限制在
`base_link` source-zone `[0.20, -0.30, -0.39] .. [0.40, -0.08, -0.25] m`；红块放到
tag 1 后位于区外，不会触发第二轮。该门控只过滤新的 `/get_objects` 发现结果，不会
改变当前任务使用的 `/get_object_pose` 缓存或 freshness；object server 的 pose TTL
仍为 `2.0 s`。

### 无运动验收与自动运行

保持前两个终端运行，在另一个 `(isaac-ros)` shell 中执行对应检查：

```bash
./scripts/check_ros_graph.sh --zedx
# 或
./scripts/check_ros_graph.sh --d455
```

检查脚本会确认选定相机的图像、CameraInfo、depth、refined 检测话题、控制 action、controller，
以及 `/camera_1/apriltag` 的 `backends=CUDA`。这一步确认实际检测组件是官方
`nvidia::isaac_ros::apriltag::AprilTagNode`/cuAprilTag，而不是本仓库中的替代检测器。
authored 模式还可以分别 echo `*_raw` 与不带 `_raw` 的话题：前者完整保留官方输出，
后者保持相同 ID、corner、frame 和时间戳，只在 IPPE 重投影误差不超过 `2 px` 时更新
平面标签 pose。refined 流只发布成功完成同时间戳图像 ROI `cornerSubPix` 与 IPPE 的
detection；缺图、坏图或求解失败时丢帧，绝不把 native pose 混入稳定滤波。官方 pose
仍完整保留在 `*_raw`。

authored 两种模式的对象类型都是 `tagged_block`，不是下面 generated 回归场景使用的
`tagged_cube`。下面的 action 只读取缓存感知结果，不会移动机械臂：

```bash
ros2 action send_goal /get_object_pose \
  isaac_ros_manipulation_interfaces/action/GetObjectPose \
  '{object_id: 0, class_id: tagged_block}'
```

tag 0 的物体中心真值由当前 authored USD 的 `/World/Workspace/TaggedCube` 和
`/R5a/base_link` 层级变换计算得到，不再使用旧 reachable 布局的硬编码坐标。比较
AprilTag/PnP 输出时应先把它转换到 `base_link`；这不是宣称当前相机模式已经达到某个
误差指标。还应读取所选相机的 CameraInfo，确认 16:10
投影修正后 `k[0]`（`fx`）约等于 `k[4]`（`fy`）：

```bash
ros2 topic echo /zed_x/left/camera_info --once
# D455 模式改为：
ros2 topic echo /d455/color/camera_info --once
```

桌面静态规划场景采用一条 raw/canonical 边界：

```text
/cumotion/static_planning_scene_raw  # NVIDIA parser 输出，5 个 world header
  -> planning_scene_frame_adapter
/planning_scene                     # 5 个 base_link 对象，MoveIt/行为树使用
```

`.scene` 中的数值本来就是 `base_link` 坐标，adapter 只修正 Isaac ROS 4.5 parser
固定写入的 header，不再次变换数值。确认感知、TF、五个碰撞对象、controller 和规划
服务均正常后，重新运行对应脚本并去掉 `auto_start:=False`，即可自动发送一次任务。

## 前置条件与正式依赖

按照主 [README](../README.md) 完成基础仿真构建，并准备以下布局：

```text
~/workspace/
├── isaac_ros_source/
│   ├── install/                         # NVIDIA Isaac ROS manipulation overlay
│   ├── isaac_ros_manipulation/          # 官方源码 checkout
│   ├── isaac_ros_manipulation_arx_r5a/
│   │   └── install/                     # ARX manipulation overlay
│   ├── isaac_ros_manipulation/arx-r5-moveit/
│   └── arx-r5-isaac-sim/
└── arx_r5_sim_ws/
    └── install/                         # description、TopicBasedSystem、sim bringup
```

在 Jazzy/Isaac ROS 环境中正式安装运行依赖：

```bash
sudo apt update
sudo apt install \
  ros-jazzy-isaac-ros-apriltag \
  ros-jazzy-isaac-ros-cumotion-object-attachment \
  ros-jazzy-py-trees \
  ros-jazzy-py-trees-ros
```

这些包不安装到 Conda `isaaclab`。开发诊断期间曾把 deb 临时解压到
`/tmp/arx_demo_ros_overlay_0722/opt/ros/jazzy`；该目录可能在重启或清理后消失，
也没有稳定的 workspace source 顺序，不能作为可复现安装。正式验收应在安装 apt 包
之后，用不包含该 `/tmp` 路径的全新 shell 重跑。

Isaac ROS Python 仍需要能找到包含 `torch` 的 site-packages。启动脚本默认使用：

```text
/var/lib/isaac-ros-cli/isaac-ros/lib/python3.12/site-packages
```

其他布局可通过 `ISAAC_ROS_PYTHON_SITE` 覆盖。

用于仿真的 ARX 描述必须为 `v0.3.0` 或更新版本，因为行为树 attachment 需要
`grasp_frame`，cuMotion XRDF 需要 `attached_object`：

```bash
git -C "$HOME/workspace/isaac_ros_source/isaac_ros_manipulation/arx-r5-moveit" \
  checkout v0.3.0
```

切换模型版本或修改本仓库配置后，在 `(isaac-ros)` shell 中重建：

```bash
export ISAAC_ROS_SRC="$HOME/workspace/isaac_ros_source"
export ARX_SIM_WS="$HOME/workspace/arx_r5_sim_ws"

source /opt/ros/jazzy/setup.bash
source "$ISAAC_ROS_SRC/install/setup.bash"
source "$ISAAC_ROS_SRC/isaac_ros_manipulation_arx_r5a/install/setup.bash"

cd "$ARX_SIM_WS"
colcon build \
  --symlink-install \
  --base-paths "$ISAAC_ROS_SRC/arx-r5-isaac-sim/arx_r5_isaac_sim_bringup" \
  --packages-select arx_r5_isaac_sim_bringup \
  --cmake-args -DBUILD_TESTING=OFF

source "$ARX_SIM_WS/install/setup.bash"
```

## 程序生成固定相机模式：终端 1

打开全新宿主机终端，只激活 Conda `isaaclab`：

如果 `~/.bashrc` 自动 source ROS，先使用 `bash --noprofile --norc`，然后执行：

```bash
source /home/lbz/miniforge3/etc/profile.d/conda.sh
conda activate isaaclab
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION LD_LIBRARY_PATH
```

```bash
conda activate isaaclab

export ISAAC_ROS_SRC="$HOME/workspace/isaac_ros_source"
export SIM_REPO="$ISAAC_ROS_SRC/arx-r5-isaac-sim"
export ISAAC_SIM_PYTHON="$CONDA_PREFIX/bin/python"
export ARX_R5_DESCRIPTION_SHARE="$ISAAC_ROS_SRC/isaac_ros_manipulation/arx-r5-moveit/isaac_ros_manipulation_arx_r5a_robot_description"
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

"$ISAAC_SIM_PYTHON" -c 'import isaacsim; print(isaacsim.__file__)'
cd "$SIM_REPO"
./scripts/run_isaac_sim.sh --scene tabletop
```

无界面运行时：

```bash
./scripts/run_isaac_sim.sh --scene tabletop --headless
```

等待以下话题持续出现，再启动 ROS 半边：

```text
/clock
/isaac_joint_states
/camera_1/color/image_raw
/camera_1/color/camera_info
/camera_1/aligned_depth_to_color/image_raw
/camera_1/aligned_depth_to_color/camera_info
```

只允许一个 Isaac Sim 场景发布 `/clock`。重复启动会导致仿真时间回跳，从而破坏 TF、
控制器和行为树超时判断。

## 程序生成固定相机模式：终端 2

打开另一个终端：

```bash
isaac-ros activate
```

进入 `(isaac-ros)` shell 后执行：

```bash
export ISAAC_ROS_WS="$HOME/workspace/isaac_ros_source"
export ARX_R5A_MANIPULATION_WS="$ISAAC_ROS_WS/isaac_ros_manipulation_arx_r5a"
export ARX_SIM_WS="$HOME/workspace/arx_r5_sim_ws"
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd "$ISAAC_ROS_WS/arx-r5-isaac-sim"
./scripts/run_apriltag_demo.sh
```

默认会启动 RViz，并在两个标签连续稳定后自动发送一次抓放任务。服务器常用：

```bash
./scripts/run_apriltag_demo.sh headless:=True start_rviz:=False
```

第一次运行建议先禁止自动运动：

```bash
./scripts/run_apriltag_demo.sh auto_start:=False
```

脚本会依次 source 官方 Isaac ROS、ARX manipulation 和仿真 overlay，补入 Isaac ROS
Python site-packages，检查必要 ROS 包，等待相机话题，并拒绝存在多个 `/clock`
publisher 的环境。不同目录布局可查看 `./scripts/run_apriltag_demo.sh --help`。

## 程序生成固定相机模式：第三个终端无运动验收

打开第三个 `(isaac-ros)` shell，并 source 与终端 2 相同的 overlay：

```bash
export ISAAC_ROS_WS="$HOME/workspace/isaac_ros_source"
export ARX_R5A_MANIPULATION_WS="$ISAAC_ROS_WS/isaac_ros_manipulation_arx_r5a"
export ARX_SIM_WS="$HOME/workspace/arx_r5_sim_ws"
export ROS_DOMAIN_ID=25
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

source /opt/ros/jazzy/setup.bash
source "$ISAAC_ROS_WS/install/setup.bash"
source "$ARX_R5A_MANIPULATION_WS/install/setup.bash"
source "$ARX_SIM_WS/install/setup.bash"
```

先检查 ROS 图和控制器：

```bash
ros2 topic info /clock
ros2 topic hz /camera_1/color/image_raw
ros2 topic echo /tag_detections --once
ros2 control list_controllers

ros2 action list -t | grep -E \
  'get_objects|get_object_pose|multi_object_pick_and_place|motion_plan|execute_trajectory|follow_joint_trajectory|gripper_cmd|attach_object'
```

预期 `/clock` 的 `Publisher count` 为 `1`，三个 controller 均为 `active`，并且
`/tag_detections` 同时包含 ID `0` 和 `1`。

下面两个 action 只读取感知结果，不会移动机械臂：

```bash
ros2 action send_goal /get_objects \
  isaac_ros_manipulation_interfaces/action/GetObjects \
  '{}'

ros2 action send_goal /get_object_pose \
  isaac_ros_manipulation_interfaces/action/GetObjectPose \
  '{object_id: 0, class_id: tagged_cube}'
```

`/get_objects` 应只返回 `object_id: 0`；tag 1 是目标标记，不是待抓物体。初始方块中心
应接近 `(0.30, -0.18, 0.025) m`。

检查由 tag 1 生成的 `link6` 投放目标：

```bash
ros2 topic echo /arx_r5_demo/drop_pose --once
```

预期约为：

```text
frame_id: base_link
position:    [0.30, 0.18, 0.1655]
orientation: [0.0, 0.7071068, 0.0, 0.7071068]  # x, y, z, w
```

这里的 pose 是 `link6` 目标，不是标签平面或方块中心。

## 程序生成固定相机模式：手动触发完整抓放

下面的命令会移动机械臂。只在前面的无运动检查全部通过，且当前运行使用
`auto_start:=False` 时执行：

```bash
ros2 action send_goal \
  /multi_object_pick_and_place \
  isaac_ros_manipulation_interfaces/action/MultiObjectPickAndPlace \
  "{mode: 0, target_poses: {header: {frame_id: base_link}, poses: [{position: {x: 0.30, y: 0.18, z: 0.1655}, orientation: {x: 0.0, y: 0.7071068, z: 0.0, w: 0.7071068}}]}, class_ids: []}" \
  --feedback
```

正常结果包括：

```text
Successfully loaded 1 grasp poses
Configured and activated joint_state_broadcaster
Configured and activated manipulator_controller
Configured and activated gripper_controller
You can start planning now!
[arx-r5-sim] tagged cube attached at grasp_frame
[arx-r5-sim] tagged cube released from grasp_frame
workflow_status: 1
```

已验证运行中，cuMotion 的 approach、grasp、retract、投放规划以及 MoveIt
`/execute_trajectory`、机械臂 JTC 和夹爪 action 均成功完成。投放后的 tag 0 视觉中心为：

```text
x: 0.29907
y: 0.17823
z: 0.02520
```

相对 tag 1 目标的平面误差约为 `2.0 mm`。

## 程序生成固定相机模式：为什么使用顶抓与 65 mm 偏移

实机 ARX 配置原有的四个水平抓取会在这个桌面场景中要求 `link6` 过低，当前 IK
无法到达，并会出现 `INVERSE_KINEMATICS_FAILURE`。本仓库没有修改实机 grasp
文件，而是在仿真 blackboard 中仅为 `tagged_cube` 指向本地顶抓：

```yaml
object_frame: /tagged_cube
gripper_frame: /link6

position: [0.0, 0.0, 0.14]
orientation: {w: 0.7071068, xyz: [0.0, 0.7071068, 0.0]}
```

当前指尖碰撞几何从 `link6` 延伸约 `157.6 mm`。`0.14 m` 顶抓 seed 使夹爪包围
50 mm 方块，同时让最终 `link6` 高度约为 `0.165 m`，指尖距离桌面的名义净空约为
`7.6 mm`。

仿真行为树参数还把 grasp approach 和 retract 从实机配置的 `100 mm` 缩短为
`65 mm`：

```yaml
grasp_approach_offset_distance: [-0.065, 0.0, 0.0]
retract_offset_distance: [0.0, 0.0, 0.065]
```

这只是适配当前 ARX 工作空间和桌面几何。行为树节点实现、action 名称、重试逻辑、
cuMotion 规划请求和 MoveIt 执行接口保持不变。

投放目标使用：

```yaml
plane_z: 0.0005
link6_offset: [0.0, 0.0, 0.165]
link6_rotation: [0.0, 0.7071068, 0.0, 0.7071068]
```

因此 `/arx_r5_demo/drop_pose` 的 z 约为 `0.1655 m`。视觉 attachment 在抓取瞬间
保存方块相对 `grasp_frame` 的完整位姿，并在搬运中保持该偏移，避免方块或标签突然
翻转。

相关文件：

- `config/tabletop_tagged_cube_grasps.yaml`
- `config/tabletop_apriltag_behavior_tree.yaml`
- `config/tabletop_apriltag_blackboard.yaml`
- `config/tabletop_apriltag_demo.yaml`

如果启动仍显示加载 4 个 grasp pose，说明运行的是旧 install space；重新构建
`arx_r5_isaac_sim_bringup` 并最后 source `$ARX_SIM_WS/install/setup.bash`。

## AprilTag 检测来源与适配边界

检测由 NVIDIA Isaac ROS 4.5 的
`nvidia::isaac_ros::apriltag::AprilTagNode` 完成，`backends=CUDA`。程序生成场景直接
发布 canonical `/tag_detections`；authored 双相机将官方原始消息与 manipulation 输入
明确分开：

```text
ZED X: /zed_x/tag_detections_raw -> pose refiner -> /zed_x/tag_detections
D455:  /d455/tag_detections_raw  -> pose refiner -> /d455/tag_detections
```

refiner 不重新检测标签。它用原始时间戳精确配对 rectified 图像，只解码官方四个
corner 包围的小 ROI，先执行 `cornerSubPix`，再结合 rectified CameraInfo 通过 OpenCV
`SOLVEPNP_IPPE_SQUARE` 处理小像素平面标签的 PnP 二义性；ID、corner、frame 和
detection header 时间戳保持不变。缺少配对图像、图像无效、重投影误差大于 `2 px`
或求解失败时从 refined 流丢帧，native pose 仅保留在 raw 话题。因而 raw 话题始终
可用于比较，canonical 话题才由 object server 与目标客户端消费。可用以下日志确认
底层检测实际加载的是 cuAprilTag：

```text
Using cuAprilTag implementation.
```

Isaac Sim 的 `/clock`、相机消息和关节 TF 会经不同 DDS 端点到达。authored 双相机的
object adapter 对每条 detection 使用 `0.5 s` wall-time deadline 异步等待同一时间戳
的 TF，缓存 pose 的 TTL 为 `2.0 s`；目标 Tag 客户端的 exact-time TF deadline 为
`2.0 s`。`future_tolerance_sec=0.5` 只允许消息时间戳相对当前 `/clock` 的有界到达
次序偏差，不是 TF 插值容差，也不会回退到 latest TF。

程序生成的固定相机回归场景仍使用 `future_tolerance_sec=0.2`；它与 authored 动态 TF
超时不能混为一谈。所有链路都保留 detection header，实机默认 future tolerance 仍为
`0.0 s`。

ARX AprilTag Object Server 不是检测器；它只把官方
`AprilTagDetectionArray` 适配成 NVIDIA manipulation 行为树使用的
`GetObjects`/`GetObjectPose` action，并应用已配置的 tag-to-object 固定变换。NVIDIA
没有提供 AprilTag 到这两个 manipulation action 的通用转换节点，因此该接口适配必须
由 ARX 集成提供。

目标标签 ID 1 仍由仿真目标客户端转换成业务定义的投放末端位姿；这是 demo 的任务
语义适配，不是 AprilTag 检测或姿态估计算法。迁移到实机时必须使用真实相机标定，并
重新评估 AprilTag 或 FoundationPose 的姿态误差。

authored 红块的投放偏移为：

```yaml
link6_offset_in_tag: [0.0, 0.0, -0.315]
```

它完全相对于实时检测到的 tag 1 定义，不使用世界坐标真值。标签 `+Z` 指向桌内，因而
负 Z 把 `link6` 抬离桌面；`0.315 m = 0.205 m` 顶抓中心偏移 `+ 0.075 m` 红块半高
`+ 0.035 m` 投放净空。诊断顺序为 `-0.280 m`/`0 mm`、
`-0.295 m`/`15 mm`（仍失败）、`-0.315 m`/`35 mm`（通过）。当前碰撞预算明确包含
Object Attachment `max_overshoot=10 mm`、XRDF attached-object buffer `2 mm`、PnP
深度误差预算 `10 mm` 和额外安全余量 `5 mm`，合计 `27 mm`，还剩 `8 mm`。纯运动学
可以求解较低目标，但加入官方 Object Attachment 的 29 个碰撞球与静态桌面后会表现为
`INVERSE_KINEMATICS_FAILURE`。新增净空只改变 tag-relative 任务几何，不绕过 AprilTag
感知或碰撞检查。

## 关键 launch 参数

| 参数 | 默认值 | 用途 |
|---|---:|---|
| `auto_start` | `True` | 两个标签稳定后自动发送一次完整抓放 action。 |
| `start_rviz` | `True` | 启动 RViz。 |
| `headless` | `False` | 关闭 ROS 感知工作流的可视化窗口；Isaac Sim headless 由终端 1 的 `--headless` 控制。 |
| `start_orchestrator` | `True` | 启动官方行为树接口的仿真 orchestrator。 |
| `quiesce_on_terminal` | `True` | action 返回终态后只暂停行为树 tick，保留 executor/action server 以可靠送达 result，并避免 one-shot 成功后重复 discovery；设为 `False` 可恢复官方连续 tick。 |
| `start_goal_client` | `True` | 发布 tag 1 对应的 drop pose，并可自动发送 workflow goal。 |
| `start_object_attachment` | `True` | 加载 cuMotion Object Attachment，维护规划场景中的 attached object。 |
| `drop_pose_ttl_sec` | `0.5` | tag 1 中值目标在发送 workflow 前允许的最大年龄，单位秒。 |
| `drop_pose_max_translation_spread_m` | `0.01` | 5 帧窗口内最大两点平移距离，超过时拒绝该窗口。 |
| `cumotion_time_dilation_factor` | `1.0` | 设置 cuMotion server/MoveIt 插件的默认时间尺度；自动行为树使用行为树 YAML 中的 `0.10`。 |

## 当前物理边界

- authored USD 红块先作为带 collider 的 kinematic 刚体等待。`joint7` linear drive 的
  最大力有限，默认 `8 N`，可用 `--gripper-drive-max-force` 调整。TopicBasedSystem 发布
  八个 joint name 和八个 position，并把 `joint7` 目标复制到 `joint8`，使两个 articulation
  关节同时受驱动；夹爪 controller 仍只拥有 `joint7`。
- 物理附着必须同时满足夹爪开度阈值、`grasp_frame` 距离阈值，以及红块与 `link7`、
  `link8` 的双侧 PhysX 接触。三项条件连续保持若干物理步后，运行时才将红块切为
  dynamic、启用 CCD，并创建连接 `link6` 的 FixedJoint。默认要求连续 `3` 步，可用
  `--grasp-contact-steps` 调整；张开夹爪会删除 joint 并清空接触门控。FixedJoint 只用于
  搬运稳定，这不是纯靠接触力和摩擦自然形成的夹持。cuMotion Object Attachment 是
  另一层机制，只维护规划场景中的 attached object。
- 程序生成的固定 `Camera_1` 回归场景仍使用 VisualAttachmentController，不能用它的
  `2.0 mm` 结果证明 authored FixedJoint 抓放精度。
- authored 模式的 `authored_usd_table.scene` 已将桌面与四条腿作为五个静态碰撞对象
  加入 canonical `/planning_scene`。`enable_nvblox=False`、`read_esdf_world=False`、
  `add_ground_plane=False` 表示仍无动态 Nvblox ESDF、独立 ground plane 和其他可见
  障碍物，而不是“桌子完全没有进入规划世界”。
- RGB-D 相机会发布彩色、深度和两份 CameraInfo，但这个版本的 AprilTag 前端只消费
  彩色图和对应 CameraInfo。
- 固定相机外参、tag 1 的目标投放平面和物体尺寸来自仿真真值。authored canonical
  pose 由官方 corners 经有界 IPPE refinement 后得到，raw 官方 pose 始终保留用于
  对照。`2.0 mm`、`workflow_status=1` 和 `7.6 mm` 名义净空只属于 generated 场景，
  不是对真实相机噪声或 authored 模式的指标。
- D455 使用 detection 原始时间戳的动态 TF，但当前任务是稳定 pose 后的一次规划执行，
  不是 continuous visual servo。
- ZED X 和 D455 authored 模式已验证图像、TF 和官方 cuAprilTag 感知；ZED 位于 ARX
  前方。但原始布局 source 为 `base_link ≈ (0.892, -0.180, -0.173) m`，cuMotion
  于 2026-07-23 返回 `INVERSE_KINEMATICS_FAILURE`，尚未完成端到端抓放。
- `7.6 mm` 是基于当前网格和目标姿态计算的名义净空，不等于完成了 PhysX 接触碰撞、
  自碰撞或实机安全验收。
- 这个 demo 验证了话题、action、行为树、规划和控制接口闭环，但不能替代真实 TCP、
  相机外参、抓取力、碰撞几何、控制周期、限位和急停验收。仿真成功不能保证实机必然
  成功。

## Authored 模式专项排查

通用环境、workspace source 和 controller 问题统一见主 README 的“常见提示”，这里仅保留
authored 抓放特有的检查项：

| 现象 | 处理 |
|---|---|
| `Successfully loaded 4 grasp poses` | 旧 blackboard/config 仍在 overlay 中；重建仿真包并最后 source `$ARX_SIM_WS/install/setup.bash`。 |
| `INVERSE_KINEMATICS_FAILURE` 且 `link6` 目标接近桌面 | 仍在使用旧水平抓取或旧 approach 参数；确认加载 1 个 `0.14 m` 顶抓和 `65 mm` 偏移。 |
| authored 抬升成功，但 `Plan To Drop Pose` 连续 `INVERSE_KINEMATICS_FAILURE` | 检查安装后的 `authored_usd_apriltag_demo.yaml` 是否仍为旧值；当前值应为 tag-relative `-0.315 m`，它给附着的 150 mm 红块保留 `35 mm` 名义净空，可越过 cuMotion 的桌面碰撞裕量。 |
| 指尖触桌或投放高度仍约 `0.1355 m` | 仍在使用旧 drop 配置；新 `link6_offset.z` 为 `0.165`，drop pose z 应约 `0.1655 m`。 |
| 看见标签但没有自动运动 | 确认没有设置 `auto_start:=False`，并检查 `/multi_object_pick_and_place`、`/cumotion/motion_plan`、`/execute_trajectory` 和 controller actions。 |

<!-- Copyright 2026 wee733; SPDX-License-Identifier: Apache-2.0 -->
