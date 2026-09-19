> **已停用：碰撞复核发现相机穿门与局部 IK 检查缺失。以下命令和配方仅保留作历史记录，采集入口不会执行。见 [碰撞复核](door_collision_review.md)。

# 圆把手开门试采与复现

这是纯物理仿真的候选数据试采，不是实机标定结果。示教会话只控制机器人，
门、把手和锁舌由接触与约束运动。助手首先分步查看图像/接触调试，
`collect_door_pilot.py` 再复现已验证的动作，并在各阶段检查物理状态。
本轮采用圆弧中点，尚未证明 ±30° 全范围成功。

## 两个进程

在仓库根目录运行，`ISAAC_SIM_PYTHON` 指向安装 Isaac Sim 5.1 的 Python 3.11。
不要混入 ROS Jazzy 的 Python 3.12 包。每条候选使用一个新的输出目录和新会话：

```bash
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/arx_r5_isaac_sim_bringup" \
  "$ISAAC_SIM_PYTHON" scripts/door_teach_session.py \
  --scene generated/door/scene.usd --output generated/door_teaching/my_pilot \
  --gripper-force 60 --gripper-stiffness 3000
```

另一个终端等待 `TEACHING_READY` 后：

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH "$ISAAC_SIM_PYTHON" \
  scripts/collect_door_pilot.py \
  --session-output generated/door_teaching/my_pilot --name candidate_001
env -u LD_LIBRARY_PATH -u PYTHONPATH "$ISAAC_SIM_PYTHON" \
  scripts/encode_door_attempt.py generated/door_teaching/my_pilot/candidate_001
python3 scripts/door_teach_client.py '{"op":"shutdown"}'
```

`collect_door_pilot.py` 使用本地 HTTP 端口 8877。会话在请求之间暂停物理；
工具调用等待不进入数据时间。它会拒绝非闭门初始状态，并在碰撞规划失败、
解锁行程不足、门未跟随或双指失去目标接触时结束为失败。

## cuMotion 的实际用途

- 从初始位到把手预抓位。
- 松开把手并退出后，转移到门侧沿预抓位。
- 每次规划先导出当前场景，包括已转动的门、门框和锁舌，共 64 个碰撞形状。
- 使用相邻 `arx-r5-moveit` 仓库的 URDF/XRDF；本轮调用 NVIDIA cuMotion
  1.1 原生求解器，**没有运行 ROS action 接口**。
- 接触阶段用连续 IK、小角度铰链圆弧及接触检查，不能把需碰撞接触的抓取
  当作纯自由空间避障规划。

`run_door_cumotion.sh` 需要官方 cp312 cuMotion wheel 的运行时；通过
`CUMOTION_PYTHON` 和 `CUMOTION_RUNTIME_DIR` 设置 Python 3.12 与运行时目录。
本机试验副本在忽略目录 `generated/runtime/cumotion`，不提交厂商二进制。
机器人安装处的台座顶部 6 厘米只从规划几何中让开固定基座，物理台座保留。

## 本轮资产参数和已修复问题

- 指爪原始单凸包封住了实际夹持开口；会话对原网格做凸分解，保留碰撞。
- joint7 采用有力上限的位置驱动，joint8 使用刚性 mimic 对应机械齿轮联动。
  60 N 上限和 3000 N/m 刚度是仿真实验值，不是实测夹持力。
- 段间延续上一条夹爪指令；不能拿受阻后的实际开度重新开始闭合插值，
  否则会在转动起点短暂卸载。
- 圆把手抓取用 link6 到抓取点约 0.145 m 的浅夹位置，防止指尖顶到门板。
- 门板边缘并非均匀厚度：板体局部 Y 为 0.0038–0.0418 m，后侧边框为
  0.0457–0.0657 m；自由侧边 X 约 0.743 m。换抓以整个边缘厚度计算中心。
- 同高侧抓受当前基座位置和关节限位限制；已验证的恢复动作采用高度
  0.56 m、夹爪向下倾斜 60° 的低位侧抓。具体是否适合实机，需用户看录像判断。
- 正面逆时针约 45° 解锁；先握把手拉到约 15°，松开，再夹侧沿拉到 30°。
  30° 是试采终止角度，不代表用户已指定最终实机开门角度。

以上几何和姿态仅适用于当前门资产。更换门或机器人时应重新配置并验证，
不能把此试采配方当作任意机器人通用开门策略。
PhysX 刚性 mimic 的依据见 [NVIDIA PhysX 文档](https://nvidia-omniverse.github.io/PhysX/physx/latest/_api_build/classPxArticulationReducedCoordinate.html)。

## 产物和审核

每条 `candidate_*` 下有原始 PNG、`wrist.mp4`、`overview.mp4`、
`trajectory.npz` 和 `episode.json`。采样 30 Hz，物理步长 1/120 s；
图像和实际状态在动作执行之前记录。状态和动作均为 6 关节弧度 + joint7 米制位置。
`contacts`、`door`、`phase` 仅用于诊断；外部总览不作为策略输入。

`success: true` 只代表本轮物理阶段检查通过，所有候选仍为
`review_required: true`。调试恢复轨迹和失败轨迹与完整候选分开存放。
腕部相机固定于 link6，光轴相对工具前方向下 30°；内外参尚未实机标定，
侧抓时门板会遮挡一部分手指，不能保证任务全程双指都完整可见。

审核后再按 [LeRobot 导出说明](door_lerobot_export.md) 生成正式训练候选；
官方 GR00T N1.7 loader 和 sim-to-real 效果尚未验证。不要用 `TEST_ONLY`
格式测试文件或失败片段充当成功专家示范。
