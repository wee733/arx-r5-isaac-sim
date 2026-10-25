# GR00T → ARX R5 实机控制接入梳理

> 本文保留实施前的检查快照。2026-10-25 已实现并测试控制适配层，两个驱动目录已合并为
> `drivers/arx_r5/`；当前结构、启动命令、实机测试与仍待标定项见
> [实机控制手册](scripts/door/real/README.md)。以下“尚未实现”描述仅代表初始检查时状态。

检查日期：2026-10-25。范围：项目结构、实机状态反馈、动作转换、调度和执行；双摄像头接入另做。

## 结论与证据范围

用户已确认 Thor 上的机械臂控制多次成功，本报告把这条链路作为已有基础。
本次只检查源码、配置、随仓库 PDF 和模型元数据，没有启动驱动、发送 CAN/ROS 控制命令或运行模型。

最新 B2ARX checkpoint 已交付；实机驱动、状态反馈和原生 FK/IK 也已有实现。
主要缺口是 **B2ARX 模型协议与现有实机控制之间的适配器，以及独立于模型推理的动作调度**。
无需为这项接入重新实现 CAN 驱动，也无需把完整 Isaac Sim / AutoData 搬到实机执行进程。

## 1. 当前目录的职责

| 目录 | 当前职责 | 实机推理是否需要 |
|---|---|---|
| `arx_door/envs/`、`scene/`、`autodata/` | 仿真、本体与门、专家数据增广和验收 | 运行时不需要；需要取出训练坐标和运动学约定 |
| `scripts/door/dataset/`、`configs/dataset/` | LeRobot 导出、13D 状态与 EEF 动作定义 | 复用接口定义，不需要重新导出数据 |
| `scripts/door/training/` | 固定版本 GR00T 全量微调与权重导出 | 推理时不需要训练器 |
| `models/groot-n17-b2arx-door/` | 最新推理权重、processor、统计和 embodiment 映射 | 需要，配置必须成套使用 |
| `arx_door/policy/`、`scripts/door/policy/` | 旧版关节空间模型服务与仿真闭环 | 可复用进程隔离/RPC 思路，不能原样用于最新模型 |
| `arx_r5_isaac_sim_bringup/` | ROS/cuMotion 与仿真控制链 | 不是实机 VLA 的必经路径 |
| `R5/` | 较早的 SDK/遥操作副本，含官方 PDF | 参考与已有控制基线 |
| `r5remote/` | 更新的双 Jazzy 遥操作及本地 Pico 控制实现 | 建议复用此处驱动、消息和原生 FK/IK |

`R5/`、`r5remote/` 都带独立 `.git`；它们不是主仓库已登记的子模块。
本次检查时主仓库将两目录列为未跟踪；`r5remote` 的 Pico 包、脚本和配置也仍有未跟踪内容。
后续固定实机部署版本时，要同时保存这些已用过的工作区文件，不能仅记录 `r5remote` 的 HEAD。
本次未移动目录、修改原有文件、提交或改变嵌套仓库。

## 2. 已有的两条实机链路

### 2.1 主从遥操作

`RemoteControl /arm_master_l_status → R5Controller(remote_slave) → InterfacesThread → slcan1 → 电机`

- 消息含 7 个位置值及可选夹爪力矩；前 6 个用于关节控制。
- `FollowCmdCallback` 使用 `POSITION_CONTROL`；位置夹爪调用 `setCatch(joint_pos[6] * 5)`。
- 默认 200Hz 指主臂消息发布配置，不是模型运行频率，也不是已测得的电机闭环频率。
- 从臂完整状态发布默认关闭，可通过 `remote_publish_slave_status` 启用。
- 网络遥操作用 Domain 17；不能把主臂目标值当作实机从臂的测量状态。

### 2.2 本地 Pico 控制：更适合复用的 VLA 接口

`Pico preview → pico_ik_bridge → RobotCmd → R5Controller(normal) → InterfacesThread → slcan1`

回传：`R5Controller → RobotStatus → pico_ik_bridge`。

源码中已具备：

- 实测 6 轴关节角、夹爪反馈、末端 XYZ/RPY、速度和电流字段。
- ARX 原生 FK/IK，Thor 链接 ARM64 的 `libarx_r5_src-arm64.so`。
- 50Hz 的桥接定时器；IK 残差、关节跳变与跟踪差检查、关节限速。
- `RobotCmd.mode = POSITION_CONTROL (5)` 的命令封装。
- 输入/反馈超时、松开时以测量关节位置覆盖旧目标、仅预览不发真实命令的模式。

`pico_control.sh driver` 配置 normal 模式，当前话题是 `/pico_r5/arm_cmd` 与
`/pico_r5/arm_status`，Domain 18、loopback。话题可重命名为专用 VLA 名称；重命名不是底层驱动重写。
VLA 可以绕过 Pico 输入，复用其中的状态读取、运动学和关节输出逻辑。

官方 PDF 和 `CmdCallback` 也支持 `RobotCmd.mode=4` 的末端控制，输入为 XYZ/RPY。
这是可选路径。初期优先采用已有 Pico 的显式 IK + mode 5 路径，便于在发送前检查解与记录转换结果。

## 3. 最新模型的实际协议

以发布包的 `processor_config.json` 和 `configs/dataset/b2arx_eef_n17.py` 为准：

| 字段 | 约定 |
|---|---|
| 状态 `eef` | 6 维，当前 R5a 安装座到 `R5a_link6` 的 XYZ（m）和旋转向量（rad） |
| 状态 `gripper` | 1 维，单侧手指位移，单位 m |
| 状态 `joints` | 6 维，R5a_joint1…6，单位 rad |
| 动作 `eef` | XYZ_ROTVEC；processor 中声明 RELATIVE / EEF / state_key=eef |
| 动作 `gripper` | ABSOLUTE；单侧手指位移，训练约定 0–0.044m |
| 时序 | 模态选取 16 个动作时间索引，训练控制时间间隔 1/30s |
| 语言 | `annotation.human.task_description`，不是旧版阶段提示字段 |
| 视觉 | wrist、ego 双 RGB；本报告不实现相机接入 |

推理服务必须使用训练固定版本 `dac54b313160c201c450f97df0fd74ebd379c892` 和保存的 processor/统计。
原始网络输出与官方 Policy API 后处理后的输出要区分：由对应 processor 负责反归一化和相对 SE(3)
动作恢复；桥接只接收经过验证的物理坐标目标，不重复加当前状态或再次做相对动作恢复。
接入时用记录样本确认实际输出语义和形状，不自行把模型配置 `action_horizon=40` 改成 16。

## 4. 还缺的控制适配

### A. 新模型服务协议

已有 `server.py` 强制检查 `single_arm/gripper`，`contract.py` 要求 7D 关节状态，
默认配置仍是旧 checkpoint 和旧官方代码版本。它还读取
`experiment_cfg/dataset_statistics.json`，而最新发布包采用根目录 `statistics.json` 等文件。

需要独立 B2ARX 配置和接口：正确加载发布包，读取其保存的统计，接收 `eef/gripper/joints`，
返回带观测编号、时刻及动作时间间隔的 `eef/gripper` chunk。仅替换 checkpoint 路径不足以运行。
GPU 推理保留在独立 Python 环境，ROS 执行器复用现有 Jazzy/C++ 控制环境；不以最新上游要求
替代此 checkpoint 的固定版本要求。

### B. 实机状态转换

从实际驱动的 `RobotStatus` 构造模型状态，而非使用上次发出的目标：

1. 对齐六轴名称、排列、正方向及零点。
2. 由实测关节做训练一致的 FK，或对驱动 `end_pos` 做经过核对的坐标变换。
3. RPY 转旋转矩阵，再转 rotation vector；RPY 与旋转向量不能逐项互抄。
4. 夹爪反馈换算成训练中的单侧手指位移（m）。

保留反馈接收时间、驱动消息时间与样本编号，后续供双相机同步使用。当前 `RobotStatus` 时间戳
由发布函数填写，不等于已证明七个电机在同一时刻采样。

### C. 坐标系和末端定义

仿真明确使用 `R5a → R5a_link6`。实机 SDK 的 base/TCP 与其是否重合，目前不能仅凭
“都叫末端位姿”判定。需要训练 URDF/资产中的相关变换，以及实机 FK 对照记录。

若 D 为 SDK 基座、M 为训练安装座、L 为训练 link6、E 为 SDK 末端：

`T_D_E = T_D_M × T_M_L × T_L_E`

模型目标转换到 SDK 后做 IK；实机反馈反向转换后进入模型。
先对若干组相同关节角比较训练 FK 与 SDK FK，辨别固定基座/TCP 偏移、零位或几何差异。
所需是准确的运动学与坐标约定，不是恢复所有历史仿真视频。

### D. 夹爪双向映射

模型使用单侧手指位移 0–0.044m；官方《03-ROS2-单臂R5-SDK》说明命令 0–5 对应夹持开口
0–80mm；现有遥操作代码又使用反馈第七项乘 5 作为命令。

这三种量不能直接互传。需要单独定义并校对：

- 模型单指位移 → 实际双指开口 → SDK 0–5 命令。
- SDK/编码器反馈 → 实际开口 → 模型单指位移。
- 开闭方向、零点、最大有效开口、接触后的反馈语义。

若两侧完全对称，双指开口才可近似为单指位移的两倍；训练上限对应约 88mm，超过手册
标称 80mm，不能静默线性拉伸整个范围。应明确超范围处理并在记录中标记。

### E. 动作 chunk 调度

模型输出多个未来目标，不能一次性全部发送，也不能把 30Hz 轨迹直接按 50/200Hz 加速执行。
需独立执行循环按轨迹时间调度，必要时对位置插值、对旋转做正确的旋转插值，然后求 IK。

- 16 步按 30Hz 对应约 0.533s 的动作时间预算。
- 旧配置 execution_horizon=8 对应约 0.267s；这不是已验证的实机最佳值。
- 推理与执行分开；按模型延迟选择执行跨度，记录目标落后和跟踪误差。
- 新 chunk 到来时明确切换点与过期动作丢弃规则，避免执行排队的旧动作。
- 旧 `rollout.py` 在推理时暂停物理仿真，这种方式不能等价迁到真实机械臂。

### F. 自主执行的状态管理

复用 Pico 已有的 IK/跟踪差/限速机制，但不能依赖握持键维持整个 VLA 状态机。
需要明确启用、运行、保持、故障、恢复及控制权切换；Pico、主从遥操作与 VLA 应由单一执行者输出命令。

可见的 R5Controller 回调本身未实现独立命令超时；原生库内部是否有相应机制未由本次静态检查确认。
模型或桥接进程停顿/退出时的策略，要由独立于推理的执行层负责。
不把 mode 0 当作通用停止，因为手册定义它为力矩清零；具体保持/阻尼选择需沿用本体验证结果。
实机限位也应使用经过确认的范围：SDK URDF 的 ±10rad 等占位值不适合作为部署限位。

## 5. 建议接入结构（尚未实现）

```text
已有双 RGB 接口（待接）＋语言
                  ↓
实机状态转换 → GR00T 独立模型服务
     ↑            ↓ 物理单位 EEF/gripper 动作 chunk
RobotStatus    时间调度 → 坐标/夹爪转换 → 原生 IK → 关节约束
     ↑                                                   ↓
     └──────── R5Controller(normal) ← RobotCmd(mode=5) ───┘
                       ↓
                  已有 SDK / CAN / 机械臂
```

建议在主仓库增加 `arx_door/real/`（纯协议/转换）、`configs/real/`（校准与时序）、
`scripts/door/real/`（离线校验入口）；增加一个小 ROS C++ 执行包复用原生 FK/IK 与消息。
驱动保留在 `r5remote`。复用 Pico 的相关控制代码时去掉 XRoboToolkit/Pico 的输入依赖，
不要把 VLA 服务绑定到手柄 SDK。这次只提出目录方案，没有创建这些实现目录。

首轮不要求引入 MoveIt/cuMotion；已有原生 IK 可完成模型 EEF → 关节目标转换。
未来若需规划自由空间接近路径，可另外接入，但规划器不应无记录地改写接触操作段的策略轨迹。

## 6. 不接摄像头时，可以先完成什么

1. 固定上述模型/状态/动作协议与物理单位，完成样本打包和解包。
2. 补齐训练坐标与夹爪映射；离线验证 FK、IK 和转换的往返误差。
3. 用合成或已记录 EEF chunk 驱动“只发布预览”的调度器，确认顺序、时间和过期处理。
4. 在后续明确开展实机验证时，复用既有驱动逐项验证状态与命令，再接完整 VLA。

没有双相机时仍可完成以上控制接口；用空白/占位图或录制图做服务测试，只能证明接口运行，
不能证明模型能根据当前实景完成开门。

## 7. 本次核对的主要文件

- `models/groot-n17-b2arx-door/README.md`、`processor_config.json`
- `configs/dataset/b2arx_eef_n17.py`
- `arx_door/envs/b2arx.py`、`arx_door/autodata/b2arx_pose_codec.py`
- `arx_door/policy/contract.py`、`scripts/door/policy/server.py`、`rollout.py`
- `r5remote/README.md`、`00-sh/ROS2/pico_control.sh`
- `r5remote/ROS2/R5_ws/src/ARX_R5_ros2_V7/arx_r5_controller/src/R5Controller.cpp`
- `r5remote/ROS2/R5_ws/src/pico_r5_bridge/src/pico_ik_bridge.cpp`、`CMakeLists.txt`
- `R5/00-readme/03-ROS2-单臂R5-SDK.pdf`

注意：Pico 的 README 保留早期 preview 阶段描述，OPERATIONS 的参数表也与当前 YAML 不完全一致。
本报告以当前源码/配置判断已有能力，不将历史文档中的默认参数作为部署参数。
