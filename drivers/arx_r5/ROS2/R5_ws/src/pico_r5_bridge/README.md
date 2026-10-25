# Pico → R5

当前已增加原生 ARX 逆解与实机执行。完整运行和调参请看 [OPERATIONS.md](OPERATIONS.md)。
以下是早期阶段一记录，仅描述只读 preview，不适用于 execute-full；其中默认值以当前 YAML 为准。

此包复用已安装的 XRoboToolkit C++ SDK，将手柄数据转换为 ROS2 调试位姿。
**不链接 ARX 驱动、不访问 CAN、不发布 PosCmd/RemoteControl，不具备实机执行开关。**
目标以零位置、单位四元数为虚拟参考，不代表机械臂当前 TCP，也尚未经过逆解。

## Thor 上使用

前提：PC Service 已运行，Pico 在同一热点连接服务，并开启 Controller + Send。

```bash
cd /home/thor/workspaces/r5remote/ROS2/R5_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select pico_r5_bridge --cmake-args -DBUILD_TESTING=ON
ctest --test-dir build/pico_r5_bridge --output-on-failure
cd /home/thor/workspaces/r5remote
bash 00-sh/ROS2/pico_preview.sh
```

运行中的状态可在另一终端查看：

```bash
bash /home/thor/workspaces/r5remote/00-sh/ROS2/pico_preview.sh --status
```

脚本使用 domain 18、loopback 和独立锁文件，不依赖 `remote_env.sh` 的固定网卡，
也不会启动机械臂。只发布以下三个话题：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/pico_preview/controller_pose` | `geometry_msgs/PoseStamped` | 新帧的原始手柄位姿 |
| `/pico_preview/target_pose` | `geometry_msgs/PoseStamped` | 50 Hz 虚拟相对目标 |
| `/pico_preview/status` | `std_msgs/String`（JSON） | 1 Hz 或状态变化时发布 |

PoseStamped 的时间是 Thor 本地发布时间；原始 Pico 时间在 status 的
`source_timestamp_ns`。`receive_age_ms` 是距最后一帧有效数据到达的时间，
**不是端到端网络延迟**。本机降频不会减少 Pico 发送的无线流量。

## 手柄验收

1. 先松开右手握持键：应显示 `released`。
2. 按住握持键（grip ≥ 0.6），应显示 `following_preview`；开始时目标不跳变。
3. 按住并分别向前、左、上移动约 5 cm。临时坐标假设为 `[-z, -x, y]`，
   应分别看到 target_xyz 的 x、y、z 增加约 0.015 m。实际方向必须人工验证。
4. 松开（grip ≤ 0.3）后目标保持，再移动手柄并重新握持，目标不应瞬间跳变。
5. 扳机应使 `trigger` 从 0 变到接近 1；本阶段不生成夹爪命令。
6. 关闭 Send，约 0.2 s 后应显示 `input_timeout`，目标保持。
7. 恢复 Send，必须先松开握持键再按住才允许跟随。

默认平移倍率 0.3，线速度上限 0.05 m/s，角速度上限 0.3 rad/s。
每次握持的位移范围 0.1 m、转角范围 0.5 rad；超过范围需松开重试。
这些均为**预览参数**，不是经过验证的机械臂安全参数。

状态 `pico_not_focused` 表示应用失焦，`pose_jump` 表示原始位姿突跳，
`invalid_tracking_packet` 表示缺字段或数据不合法，均需恢复有效输入并松开重新使能。
输入无效时不会继续朝旧目标插值。默认固定第一个有效设备 ID，防止另一设备接管。

## 已知边界与下一阶段

- SDK 采样中没有看到独立的控制器 tracking-valid 标志。`focus`、时间戳和非零
  四元数不能充分证明手柄追踪有效；静止与持续返回冻结位姿也无法可靠区分。
- 当前超时基于本机接收时间，不能检出所有 TCP 队列中的旧帧。实机前需加入经过
  验证的源时间新鲜度判断/时间同步策略。
- 未进行真实 TCP 标定、正逆解、关节限位、碰撞检查或动力学控制。
- 下一阶段需获取机械臂实测状态作为锚点，验证内置逆解或外部求解器，并在
  R5Controller 一侧独立实现超时和停止策略后，才允许接入执行话题。
- 不要将本包 target_pose 直接转换后转发到 `/ARX_VR_L`。

## PC Service 的 ICU 兼容依赖

Thor Ubuntu 24.04 的系统 ICU 是 74，安装的 PC Service 要求 ICU 70。
已将官方 Ubuntu ARM64 `libicu70_70.1-2_arm64.deb` 解包到
`/home/thor/.local/share/xrobotoolkit-compat/icu70`，未替换系统库。
服务停止后可用下面命令启动；已运行时不要重复启动：

```bash
nohup env LD_LIBRARY_PATH=/home/thor/.local/share/xrobotoolkit-compat/icu70/usr/lib/aarch64-linux-gnu \
  bash /opt/apps/roboticsservice/runService.sh \
  >> /home/thor/.local/share/xrobotoolkit-compat/service.log 2>&1 < /dev/null &
```

桥接本身链接已安装 SDK 的绝对路径/RPATH，不需要加载 Qt 或 ICU 兼容库。
