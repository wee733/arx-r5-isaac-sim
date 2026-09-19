# 侧抓碰撞复核与旧配方停用

用户在回放中指出侧抓碰撞后，原“完整成功候选”的判断撤销。
`pilot02/candidate_002`、`pilot03/candidate_003` 的 `success` 已改为 false，
`quality_status` 为 `rejected_collision_review`。原始视频、轨迹和修改前
`episode.pre_collision_review.json` 均保留。门确实完成了开启动作，但这不等于合格示范。

## 已确认的原因

1. **门板没有从 cuMotion 中遗漏。** 两条路径的原始规划请求均包含
   `/World/Door/door_panel/Colliders/panel`，尺寸约 0.725×0.038×0.790 m，
   并包含门框、边框等共 64 个环境碰撞体。
2. **腕部相机缺少机器人碰撞几何。** 相机只是 link6 下的虚拟 Camera prim，
   URDF/XRDF 没有对应的相机机身碰撞包络。根据实际关节记录、基座变换和
   每帧门角重建相机坐标后，样本 1 有 57 帧相机原点处于门板实体内部，
   首次出现在 36.567 s；样本 2 有 56 帧，首次在 35.967 s。
   这些帧分布在多个区段，累计分别约 1.90 s、1.87 s，不能误称连续持续时间。
   穿入最深约 9.8 mm、11.0 mm。cuMotion 转移段已经发生穿入，因此不能
   只归因于后续 IK。
3. **后续局部 IK 没有碰撞检查。** 下降、侧沿接近和拉门只验证 IK、限位和速度。
   记录器只监测两指与把手/门板接触，成功条件没有覆盖非预期的机身、连杆、
   门框接触。这也是必须修正的独立缺陷。

额外的离线 FCL 检查使用 URDF link1–link6 三角网格和门板长方体，没有找到
这些连杆与主门板相交；这不是全场景无碰撞证明，也不覆盖实际相机机身。
不能据此否认用户看到的问题，更不能把相机穿门当成普通视觉遮挡。
证据位于 `generated/door_collision_audit/`，包括 `camera_crossing.png`、
`camera_rejection.json`、`mesh_panel_audit.json` 和原始规划请求。

## 正常高度侧抓的新证据

之前“同高不可达”只是在有限的固定角度下得到的结果，不能推广到所有侧抓姿态。
重新改变侧向接近角后，在当前把手高度约 0.928 m、工具俯仰 0°、门板局部
偏航 150°（相对正侧向斜约 30°）找到了可达的预抓姿态。
保持门板等全部碰撞体，cuMotion 返回 SUCCESS，路径约 10.41 s。
该路径的虚拟相机原点到主门板最小距离约 89.8 mm。

这只证明了一个**同高预抓路径候选**：尚未执行新的物理抓边与拉门，
相机原点距离也不能替代完整机身碰撞检查。低位侧抓并非必要条件。
另外两个角度的候选被 cuMotion 判为无效目标，失败结果原样保留。

## 当前处理及恢复条件

- 旧低位 `collect_door_pilot.py` 入口已停用，防止继续产出错误的成功标签。
  保留代码用于定位历史问题；回归检查确认它在访问仿真器之前就退出。
- 原始诊断数据保留，两条示范资格撤销。LeRobot 导出器已有的失败/审核门控会拒绝它们。
- 后续应为实际安装的相机、夹爪及支架建立匹配的仿真和规划碰撞几何。
  自由空间转移与预接近全程验证碰撞；接触阶段只允许指定指垫与指定物体表面
  接触，其他连杆、相机、门框碰撞必须导致失败。
- 优先复核正常高度的斜向侧抓候选，完成物理回放与碰撞审核后再恢复采集。
  本次没有把新候选冒充已修好的专家数据。

## 抬高且水平侧抓的试采入口

`scripts/collect_door_upright.py` 使用把手上方的侧抓高度（默认 1.01 m）和
水平工具姿态，门板局部偏航默认 165°，即从正侧向斜着接近约 15°。
`--edge-height`、`--edge-yaw` 和 `--edge-pregrasp-distance` 可以调整候选；
预抓距离默认 0.18 m，门沿坐标仍针对当前门资产。
先由 cuMotion 规划自由空间接近和换抓转移，再用局部 IK 执行接触动作。
当前尚未把接触段变成完整的碰撞约束规划器。

配套 `door_teach_session.py` 在每个物理步前后检查虚拟相机原点到随门旋转的
主门板距离，默认至少 10 mm；也记录 link1–link6 的净接触力，默认超过
2 N 中止。cuMotion 轨迹执行前还会逐点检查虚拟相机距离。
这不包含相机机身，也不能通过净力抵消时的接触检测来证明全场景无碰撞。
相机安装位置与向下 30° 的光轴不变。

新轨迹额外保存 `camera_clearance` 和 `arm_contacts`，触发保护的试采不能标记
成功。侧沿抓取除要求双指有力，还要求两指沿门板法向的接触力方向相反，
避免把单纯顶住门边误认为夹持。所有新试采仍需回放审核才能导出训练数据。

## 本轮实际执行结果

- `upright01/candidate_005`：高度 0.928 m、偏航 150°。无视点穿入主门板，
  但靠近时触及旋钮附近，门从 15° 被推回约 11.6°，只有一指形成目标接触。
  自动结束为失败，1488 帧诊断数据保留。
- `upright02/candidate_006`：高度 1.01 m、偏航 165°，俯仰 0°。
  两段 cuMotion 规划成功，完成完整解锁、先拉门缝、松手换抓和侧沿拉门，
  最终门角约 30.59°。1955 帧、30 Hz，视频约 65.17 s。
  记录中的虚拟相机原点到主门板最小距离约 18.86 mm，非指爪连杆净接触力
  最大记录值为 0 N，未触发保护。两指在夹持检查点沿门法向的力方向相反。
  这些检查有前述覆盖限制，不能称为完整机身的无碰撞证明。

第二条 `success: true` 仅表示现有阶段检查通过，`review_required: true` 保留。
腕部图像仍有较重的门沿遮挡，尚未认定为训练数据；相机并没有人为移动或转向。
完整双视角视频、原始两路视频、轨迹和 `review_metrics.json` 均在候选目录。

后续用户明确审核通过第二条：其 `review_required` 已更新为 false，
`quality_status` 为 `user_approved_seed`，审核前元数据已备份。
该条进入 [增广路线](door_augmentation_route.md) 的种子集合；上述失败条目仍拒收。

复现时在新的会话和输出目录运行，避免复用已经打开的门：

```bash
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/arx_r5_isaac_sim_bringup" \
  "$ISAAC_SIM_PYTHON" scripts/door_teach_session.py \
  --scene generated/door/scene.usd --output generated/door_teaching/my_upright \
  --gripper-force 60 --gripper-stiffness 3000
```

另一个终端等待 `TEACHING_READY`，然后运行：

```bash
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/arx_r5_isaac_sim_bringup" \
  "$ISAAC_SIM_PYTHON" scripts/collect_door_upright.py \
  --session-output generated/door_teaching/my_upright --name candidate_000
```
