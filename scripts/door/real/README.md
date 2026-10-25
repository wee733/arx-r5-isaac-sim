# GR00T → ARX R5 实机控制

本目录只负责实机控制接口。摄像头、GR00T GPU 服务和模型闭环效果不属于本轮实现。
使用最新 B2ARX 模型的 `eef / gripper / joints` 状态协议；不修改旧版仿真策略服务。

2026-10-25 的控制标定与测试见 [当前验收记录](../../../configs/real/commissioning_20261025.json)。
已拉取训练原资产，完成软件坐标约定核对、夹爪双向编码器标定和有界跟踪补偿。
模型坐标接口的 5 mm 实机测试：稳定段位移 4.87 mm，返回位置误差 1.19 mm；
单指 20 mm 目标的标称尺寸反馈约 19.94 mm。位置由编码器 FK 计算，未做外部量具测量。
**稳定误差与运动误差分开验收**：2 秒单程的测试换向瞬态最大约 5.84 mm，
6 秒单程测试的最大误差降为 4.10 mm，稳定段误差 0.32 mm、返回误差 0.50 mm。
不能用稳定段结果宣称整段轨迹均在 3 mm 内。旧版失败数据仍保留在
[初始记录](../../../configs/real/validation_20261025.json)。

## 当前结构

```text
drivers/arx_r5/                  唯一 ARX 驱动、SDK、官方手册、原有 Pico/主从控制
ros2/arx_vla_control/            C++ 执行器、独立看门狗、原生运动学测试
arx_door/real/contract.py        GR00T 观测/动作协议；无 ROS/CUDA/Isaac 依赖
configs/real/arx_r5.json         坐标变换、夹爪映射、时序与执行范围
scripts/door/real/              构建、启动、实机小幅测试、坐标核对
build/real/ install/real/        统一 ROS 构建产物（不进 Git）
outputs/real-diagnosis/         当前跟踪、被动 CAN 与夹爪标定原始记录（不进 Git）
outputs/real-validation/        初始测试原始记录（不进 Git）
```

`R5` 与 `r5remote` 已合并。以 `r5remote` 的当前工作文件为基础，补入旧版独有的
`remote_realsense.rviz`。旧仓库、Git 历史和旧 ROS 构建产物保存在项目旁的备份目录，
详见 `drivers/CONSOLIDATION.json`。主项目中的驱动是普通目录，避免提交时误变成只含
HEAD、不含本地 Pico 实现的嵌套 gitlink。SDK 原有 LICENSE 随目录保留。

Pico 与主从脚本已切换到统一 install；`build-pico` 构建原 Pico 包，需原 XRoboToolkit SDK。
两条控制链仍须独占同一机械臂，驱动启动入口会拒绝重复的 `R5Controller`。

## 数据路径与执行语义

```text
实际 RobotStatus → 原生 FK → 坐标/夹爪状态转换 → /vla/observation
                                                   ↓（未来相机与模型服务）
官方 GR00T 后处理输出 → Python contract → /vla/chunk
                                             ↓
executor：时间采样、SE(3) 插值、原生 IK、解分支/残差检查、限速
                                             ↓ /vla/candidate
watchdog：独立进程、反馈/命令超时、控制权/模式/速度检查
                                             ↓ /vla_r5/arm_cmd
原有 R5Controller(normal, mode=5) → 原有 SDK → slcan1 → 机械臂
```

- 观测和执行为 50 Hz。GR00T 动作保留 30 Hz 时间轴；第 i 个目标到达时间为
  观测采样时间 + i/30 秒（训练数据是 pre-action 观测）。执行器按真实时间插值，
  最后一帧保持至 N/30 秒；新 chunk 不再从滞后的实测位置重新插值，避免每次重规划回退。
- 新 chunk 替换旧 chunk，使用执行器记录的观测时间；推理耗时占用轨迹时间预算。
  不允许把过期结果重新从第零步播放。默认 Python 接口取 8/16 步，可显式选择 1..16。
- 启用会产生新 session。跨 session、重复序号、倒序观测、未知观测、超过 0.3 秒的观测、
  已到期动作、非有限数、错误单位或超过范围的目标均拒绝。
- IK 复用 Pico 使用的 `arx::KinematicSolver`。求解前后检查分支变化、实测关节跟踪差、
  FK 残差和配置关节限位。上层不依赖 Pico SDK、Isaac Sim 或 MoveIt。
- 原厂 SDK 增益不变；增加低速积分位置补偿，增益 0.6/s，上限 ±0.02 rad，带抗饱和。
  原始 IK 目标和补偿后的命令分别限速，默认 0.15 rad/s；停止/重新启用时清空积分。
  状态公开 `q_nominal / q_correction / q_command`，可检查补偿是否达到上限。
- 默认 **不启用执行**。`execute=false` 的预览节点完全不发布 `/vla/candidate`。
- chunk 耗尽后保持实测位置，等待新 chunk；错误则锁定为 disabled，需要明确重新启用。
- 看门狗在执行器停止发命令约 150 ms 后发出保持并锁定。反馈过期时冻结最后一个有界目标。
  它能覆盖推理/执行进程退出，不能覆盖看门狗与驱动同时退出、主机断电或 SDK 内部故障。
- `RobotStatus` 时间戳是驱动发布时间；当前 SDK 不提供逐电机 CAN 采样时间。
  ROS 反馈新鲜度不等价于已经验证所有电机的底层总线反馈新鲜度。

## 构建与启动

在项目根目录执行，使用系统 ROS 2 Jazzy（不需要模型 Python 环境）：

```bash
bash scripts/door/real/control.sh build
# 可选：同时构建原 Pico 控制
bash scripts/door/real/control.sh build-pico

# CAN 未启动时执行；已启动则只检查，不抢占现有进程
bash scripts/door/real/can_up.sh

# 三个独立终端；启动后 executor/watchdog 均为 disabled
bash scripts/door/real/control.sh driver
bash scripts/door/real/control.sh watchdog
bash scripts/door/real/control.sh executor

# 另一个终端，只读取反馈
bash scripts/door/real/control.sh inspect
```

`VLA_CAN`、`VLA_CAN_DEVICE` 可以覆盖默认 `slcan1`、`/dev/arxcan1`；`VLA_CONFIG`
可选择另外的配置。DDS 固定为 localhost、Domain 18，与原 Pico 本地控制一致。
如这些进程已经运行，直接复用，不能重复启动。

本机当前内核 `6.8.12-1021-tegra` 原先没有 slcan。此次使用本机匹配的内核 headers，
从 [Linux v6.8 官方镜像源码](https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/+/v6.8/drivers/net/can/slcan/)
编译 `slcan-core.c / slcan-ethtool.c / slcan.h`，安装至
`/lib/modules/6.8.12-1021-tegra/extra/slcan.ko` 并加载；未修改内核或重启机器。
构建源码和产物保存在 `outputs/slcan-build/`。以后换内核需使用相应 headers 重编，不能复用旧 vermagic 模块。

## 微调模型接入入口

本机发布的 `models/groot-n17-b2arx-door` 是 **B2ARX EEF 模型**。
旧 `scripts/door/policy/server.py` 属于 `single_arm` 关节动作模型，不能直接复用其协议。
实机入口统一从 `control.sh groot` 进入，配置集中在 `configs/real/groot.json`：

```bash
# 一次性准备：复用 Thor 已安装的 CUDA PyTorch，其余依赖装进项目独立 venv
bash scripts/door/real/control.sh groot setup
# 首次需要已获批 Cosmos-Reason2-2B 的 Hugging Face 登录；不会改 Git credential helper。
# 仅检查真实权重加载、合成输入推理和输出形状，不连接机器人
bash scripts/door/real/control.sh groot verify

# 模型服务独立终端，默认 127.0.0.1:5568，使用 checkpoint 自带 processor/statistics
bash scripts/door/real/control.sh groot serve
# driver/watchdog/executor 按前文启动后，再启动策略循环
bash scripts/door/real/control.sh groot run --language 'Open the door.'
# 随时停止策略与控制，保留原驱动的位置保持
bash scripts/door/real/control.sh groot stop
```

`groot preview` 只把模型输出发布到 `/vla/policy_preview`，不会启用控制或发布 `/vla/chunk`。
`run` / `preview` 可加 `--duration 10` 限定运行时间；默认持续运行，支持 Ctrl+C / SIGTERM。
配置覆盖用 `VLA_GROOT_CONFIG=/absolute/path/config.json`。默认源代码采用本模型训练版本；
启动不计算权重 SHA，也不扫描训练数据。旧仿真服务和旧训练配置不受影响。

相机本轮不接设备。策略入口只预留两个标准 `sensor_msgs/Image` 输入：
`/vla/camera/wrist/image_raw` 与 `/vla/camera/ego/image_raw`（可在配置中改名）。
未来提供 640×480 `rgb8` 或 `bgr8`，带本机 ROS 时钟的采集时间戳即可。
不提供假图像补位，不自动启动或配置任何相机。缺图时报告 `waiting_for_wrist_and_ego_RGB`，
保持 disabled；执行中图像/机械臂反馈过期则停止。图像过期阈值为 0.2 秒，双路时间差最多 0.05 秒。

推理工作线程只负责官方 PolicyClient；ROS 主循环持续处理状态和停止请求。
每次请求固定一份观测，返回后使用原 session、观测编号和原动作时间轴。
默认每次保留 16 步、按 8 步间隔重新观测，至多一个请求在途，不排队旧动作。
观测至结果发布预留 30 ms 传输余量；超过执行器的 0.3 秒新鲜度预算就停止，
不会改时间戳来伪装实时性。模型服务应该先完成 warm-up，再启动策略循环。
`/vla/policy/status` 报告等待原因、推理耗时、观测年龄和序号。
异常、正常退出和显式 stop 均禁用 executor/watchdog；进程被强杀时仍由原执行器的 chunk 到期保持兜底，
再次启动前执行一次 `groot stop` 清理仍然启用的会话。

本机已加载发布的真实权重，合成输入 warm-up 后两次 GPU 推理为约 122 / 142 ms，
输出形状为 `(1,16,6)` 与 `(1,16,1)`；实际官方 RPC 往返两次约 148 / 124 ms。
这只验证运行环境和协议，不代表视觉任务效果。
控制接入做了一项隔离 Domain 72 的联调：官方 RPC → runner → 原生 executor/watchdog → 假驱动，
覆盖缺图不启用、正常动作执行、延迟结果停止；不打开 CAN。
复跑入口（使用已准备的模型运行环境）是：

```bash
source scripts/door/real/env.sh
PYTHONPATH="$PWD/generated/runtime/groot_b2arx/source:$PYTHONPATH" \
  GROOT_PATCH_MISTRAL=1 NO_ALBUMENTATIONS_UPDATE=1 \
  generated/runtime/groot_b2arx/.venv/bin/python scripts/door/real/test_groot_runner.py
```

对应文件只有：`serve_groot.py`（加载/服务）、`run_groot.py`（ROS 推理循环）、
`groot.sh`（环境入口）、`setup_groot.sh`（本机依赖）。原生 IK、限速、关节补偿和看门狗继续复用原实现。

## 不接相机的验证

```bash
bash scripts/door/real/control.sh test
source scripts/door/real/env.sh
python3 scripts/door/real/test_contract.py
python3 scripts/door/real/integration_test.py
```

集成测试启动真实 executor/watchdog 二进制与一个假驱动，固定在独立 Domain 71，**不打开 CAN**。
覆盖保持、chunk 到期、重复序号、过期推理、反馈丢失、竞争发布者、只读记录器共存，
以及执行器 SIGKILL 后独立看门狗保持。看门狗按 `driver_node_name` 识别唯一驱动订阅者，
不会把只读诊断订阅者误判成第二台驱动；真实命令发布者仍必须独占。

已连接机械臂时，以下脚本会启用、测试、保持并禁用执行；驱动继续保持最终位置。
默认只保持，运动参数均有硬上限。记录包含实测状态、命令、看门狗状态和结果：

```bash
bash scripts/door/real/control.sh smoke --output outputs/real-validation/hold.json
bash scripts/door/real/control.sh smoke --motion-m 0.005 --axis 2 --duration 8 \
  --output outputs/real-validation/motion_z_5mm.json
bash scripts/door/real/control.sh smoke --gripper-delta 0.2 --duration 5 \
  --output outputs/real-validation/gripper_02.json
```

运动测试为当前末端位置附近的平滑往返，不归零、不走预设姿态；幅度限制为 0–10 mm。
`smoke` 默认峰值/返回容差为 3 mm，可用 `--position-tolerance-m` 指定；这是任务级门槛，
不是 GR00T 模型要求。峰值达标不等于整个运动过程中都达标。

详细的上升—保持—返回—保持测试同时记录 ROS 状态、目标和只读 CAN 报文：

```bash
source scripts/door/real/env.sh
python3 scripts/door/real/diagnose_tracking.py --frame model --gripper-metres 0.02 \
  --ramp-s 6 --output outputs/real-diagnosis/model_frame_slow_20mm.json
python3 scripts/door/real/summarize_tracking.py outputs/real-diagnosis/model_frame_slow_20mm.json
```

此脚本构造与 GR00T 解码输出相同形状的绝对目标，并通过 `from_policy_reply` 送入真机。
它没有运行模型推理。默认单程 2 秒、各保持 4 秒，`--ramp-s` / `--hold-s` 可调整；
夹爪受 0.5 SDK 单位/秒限制，从闭合到全开至少需要约 10 秒，不能要求它跟随更快的开口目标。
摘要分别报告 `settled_tracking_passed` 与 `continuous_tracking_passed`，原始脚本的
`complete` 只表示测试完整执行。SDK 模式的 `smoke --gripper-delta` 只是编码器响应检查。

停止时先调用 `/vla/enable=false`，再调用 `/vla/watchdog/enable=false`，留下保持目标；
之后可结束这两个适配进程。直接退出驱动会进入原厂 SDK 的清理流程，不能把它与保持等同。

## 接入 GR00T 的调用约定

```python
from arx_door.real.contract import observation, from_policy_reply

# state_msg 来自 /vla/observation；RGB 以后由相机模块提供。
obs = observation(state_msg, {"wrist": wrist_rgb, "ego": ego_rgb}, "Open the door")
reply, info = policy.get_action(obs)  # 使用训练固定版本与保存的 processor
chunk = from_policy_reply(reply, state_msg, sequence=next_sequence, execute_steps=8)
# JSON 序列化后发布 std_msgs/String 到 /vla/chunk；不得修改 session/观测编号。
```

输入 state 是 `eef(6)`、`gripper(1)`、`joints(6)`，均来自实测反馈。
动作必须是官方 Policy API 已经反归一化、恢复相对动作后的绝对物理目标：
`eef (1,16,6)` 为 XYZ + rotvec，`gripper (1,16,1)` 为单侧手指米数。
不得把原始网络 tensor 直接传入，不重复叠加状态。

模型帧 chunk 使用 `frame=model_mount_link6`；实机诊断使用 `frame=sdk_base_tool`。
夹爪默认 `gripper_units=single_finger_m`。允许模型训练范围及少量回归误差
（−0.001 至 0.045 m），执行端将物理开口限在单指 0–0.040 m，超界样本累计到
`gripper_clipped_samples`。0.044 m 请求会饱和为 0.040 m，不按比例缩小所有动作。
更大越界、非有限数会拒绝。只有 SDK 帧诊断允许 `gripper_units=sdk_command`。

## 坐标与夹爪标定依据

`configs/real/arx_r5.json` 的 `frames_verified / gripper_verified` 已设为 true。
它们表示下列软件约定与编码器映射已核对，不表示完成外部计量：

- 训练原资产为 `assets/b2arx_robot/my_robot.urdf`，提交 `a3af4d4`；对应 USD 和依赖已拉取。
  原生 SDK FK 的 XYZ 会减去零关节时的位置，旋转不减零位；已用 100 组关节独立 XML FK
  核对。模型安装基座 → SDK 原点平移为 `(-0.0977, -0.00004956, -0.1635) m`。
  两端末端都是 link6，不是指尖。模型输入的 eef 由实机 FK 映射到这个公共安装基座系。
- SDK 与训练 URDF 的腕部尺寸不同，采样位置差最大约 4.08 mm。这是关节相关的几何差异，
  不用虚构的固定 base/tool 偏移掩盖。外部定位精度和实际接触任务未验证。
- 关节限位取 SDK 实际夹限和训练限位的交集，q2/q3 下界保留 −0.03 rad 以容纳当前编码器零偏。
  SDK 的 q5/q6 限位比先前配置窄，现已在上层提前约束，避免底层静默夹限。
- 夹爪命令 0/1/2.5/4/5 做了双向往返，闭合反馈约 −0.01850，全开约 4.89738，
  最大双向滞回约 0.01144 编码器单位；命令采用实测曲线的分段线性逆映射。
  保持位置也使用同一映射，避免把编码器读数直接当命令。
- 单指米数按**官方标称总开口 80 mm**线性换算，未使用卡尺实测；状态公开
  `gripper_opening_reference`。训练上限总开口 88 mm 与真机标称 80 mm 的差异显式饱和处理。

[坐标记录](../../../configs/real/frame_calibration.json) 与
[夹爪记录](../../../configs/real/gripper_calibration.json) 保存来源和实测点。
重新核对坐标（无机械臂运动）或重新扫描夹爪（会完整开合）分别使用：

```bash
source scripts/door/real/env.sh
python3 scripts/door/real/calibrate_frames.py
python3 scripts/door/real/calibrate_gripper.py --help
```

## 跟踪调试结论与使用范围

被动 CAN 抓包确认关节目标已经到达电机命令，主要问题是小幅关节运动存在滞后/死区，
并叠加旧版每个 chunk 从实测位置重新插值的问题。现已修复动作时间轴，并加入有界积分补偿。
5 mm 目标的两次 SDK 坐标测试稳定位移为 5.32 / 4.13 mm，最后两秒平均返回 Z 偏差为 −0.34 / +0.69 mm；
模型坐标 + 20 mm 单指动作也已通过稳定误差检查。原厂增益和 CAN 驱动保持原样。

另试过 1.5/s 积分增益，出现约 9.59 mm 瞬时偏差，**未采用**；最终配置保持 0.6/s。
这些记录只覆盖当前姿态附近的小幅运动，不能推成任意姿态、速度、接触负载下的跟踪保证。
目前模型服务与控制推理循环已接入；双相机设备未接，真实视觉闭环与接触任务尚未验收。
完整初始梳理见根目录 `CONTROL_VLA_INTEGRATION.md`；当前实现与边界以本文件和当前验收记录为准。

## 无相机的真实模型输出实机演示

`model_smoke.py` 提供独立的 `capture / execute --output <记录.json>` 诊断：
用当前实测状态和空白双图请求真实模型，保留原始预测，再按最多 10 mm 平移、
0.02 rad 旋转、3 mm 单指开度变化缩放，慢速回放并返回。回放使用新的执行时间轴，
不能当作未经修改的在线策略闭环；正常 `groot run` 仍禁止缺图运行。

2026-10-25 实测：推理 178 ms，原始预测最大位移约 50.25 mm，平移缩放至约 19.9%。
16 秒执行期间，编码器 FK 实测最大位移 9.39 mm，标称总开口约 5.15 mm，
返回误差约 1.88 mm；随后 executor/watchdog 均禁用，原驱动保持位置。
原始预测和执行记录位于 `outputs/real-validation/model_hardware.json` 与
`model_hardware.execution.json`。这证明模型输出能经过控制链驱动真机，不证明无视觉开门能力。
