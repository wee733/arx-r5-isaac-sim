# 摇杆平移 + 相对姿态（需逐轴实机验收）

独立配置 `config/joystick.yaml`，不覆盖原有 `preview.yaml`。
左 axisY → 基座 X 前后；左 axisX → 基座 -Y 左右；右 axisY → 基座 Z 升降。
右 axisX 不控制运动。平移无需握持，是摇杆偏转量 × 速度 × 时间的增量积分；
回中立即停止目标积分。position_scale 不参与摇杆模式，调速度用 linear_speed_m_s。
三个方向合成速度受总速度上限约束。

右握持仅启用姿态与原有相对扳机夹爪控制。松开握持保持姿态目标，摇杆仍可平移。
旋转使用每次握持时的相对四元数，先变换坐标轴，再轴角缩放；不是 RPY 逐项相加。
rotation_axis_map 是坐标基变换的三行，-3 表示输入 -Z；默认 [-3,-1,2] 保留旧映射。
数学测试不代表实际手柄坐标系已经标定，必须逐轴检查实际方向。

启动、断线恢复、范围/逆解异常后：先双摇杆回中且松开握持，再继续。
平移上限相对每段连续推杆开始时的起点；转角上限相对每次握持起点。
松开握持不再是全运动停止按钮：要双摇杆回中并松开握持，才能停止所有跟随。

## 先只读测试

PC Service 和 Pico Controller/Send 已运行、两个控制器唤醒。确保旧 preview 已停止。

```bash
PICO_CONFIG=/home/thor/workspaces/r5remote/ROS2/R5_ws/src/pico_r5_bridge/config/joystick.yaml \
  bash /home/thor/workspaces/r5remote/00-sh/ROS2/pico_preview.sh
```

此命令不启动机械臂驱动/执行节点。依次测试左摇杆前、右，右摇杆上，回中；
保持摇杆回中、按住右握持，分别只做抬头(Pitch)、左右转(Yaw)、拧腕(Roll)约5°。
观察 left_stick/right_stick、target_xyz、target_xyzw，核对各轴方向和混合量。
首次先回中、松开一次；不要同时进行三轴旋转。

## 验证后完整启动

```bash
bash /home/thor/workspaces/r5remote/00-sh/ROS2/pico_joystick_tmux.sh
```

使用 r5-pico 会话。旧会话不会自动被替换，也不会自动切换配置。
需先妥善停止旧控制链路。完整启动需要 START 确认，会使能机械臂。
本模式默认小范围低速，用户旧的2倍比例/较高速度配置保持在原 preview.yaml 中。
绝对关节限位、碰撞检查和驱动独立看门狗等原有未完成项仍未实现。
