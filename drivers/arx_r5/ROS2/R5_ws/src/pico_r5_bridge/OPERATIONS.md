# 配置与完整运行

唯一配置：`ROS2/R5_ws/src/pico_r5_bridge/config/preview.yaml`。
preview 和 execute-full 启动时读取源目录中的同一文件。改 YAML 后重启这两个节点，
不需要重新编译；不是热更新，勿用 ros2 param set 修改构造时缓存的运动参数。

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| position_scale | 1.0 | 末端目标位移 / 手柄位移 |
| rotation_scale | 1.0 | 相对旋转角倍率，使用轴角而非逐项缩放 RPY |
| linear_speed_m_s | 0.05 | 末端目标平移限速，m/s |
| angular_speed_rad_s | 0.3 | 末端目标转动限速，rad/s |
| max_joint_speed_rad_s | 0.2 | 各关节命令限速，rad/s |
| max_clutch_translation_m | 0.1 | 单次握持映射后的三维位移模长上限，m |
| max_clutch_rotation_rad | 0.5 | 单次握持映射后的总相对转角上限，rad |

两节点共用距离和转角上限。旋转倍率 0.5、角度上限 0.785398 表示手柄 90°
对应末端目标 45°；当前未默认采用此示例，仍是 1:1、约 28.6°。
范围不是绝对工作空间或各关节角限制。超范围停止本次跟随，需松开重握，并非钳位。
实际速度还受关节限速、逆解/跟踪差检查影响。原始输入突跳与超时检查保留。

## 启动（分别在三个终端前台运行）

前提：PC Service、Pico Controller + Send、slcan1 就绪。已有同类进程时不要重复启动。

```bash
# 终端1，已有驱动则跳过；调参不需重启驱动
bash /home/thor/workspaces/r5remote/00-sh/ROS2/pico_control.sh driver
# 终端2
bash /home/thor/workspaces/r5remote/00-sh/ROS2/pico_preview.sh
# 终端3，会控制真实机械臂
bash /home/thor/workspaces/r5remote/00-sh/ROS2/pico_control.sh execute-full
```

改配置：松开握持并确认停止，Ctrl+C 停终端3，再停终端2；编辑 YAML 后依次重启
终端2、3，松开再握持。后台 nohup 实例需先核对进程 PID 再停止，不能用 Ctrl+C 停止。
`execute` 保留早期小范围平移默认值，不读取完整配置；请用 `execute-full`。
`ik-preview` 不发布真实命令。不要并行启动多个驱动或 SDK 位姿节点。

```bash
bash /home/thor/workspaces/r5remote/00-sh/ROS2/pico_control.sh status
bash /home/thor/workspaces/r5remote/00-sh/ROS2/pico_preview.sh --status
```

握持跟随，松开停止。夹爪采用相对扳机映射：闭合起步时先扣住扳机再握持，
缓慢松扳机张开，再扣下闭合。手指远离夹爪。

链路复用 XRoboToolkit SDK、ARX 原生 FK/IK、R5Controller POSITION_CONTROL。
preview 状态发布50Hz、日志约1Hz；receive_age_ms 不是端到端网络延迟。
未完善绝对关节限位、碰撞检查、独立驱动看门狗、源时间同步和追踪冻结检测。
