---
tags: [ARX-R5A, GR00T-N1.7, VLA, SkillGen, Data-v2, Gemini, Self-Improvement, B2-ARX]
date: 2026-10-17
supersedes: Gemini 初稿（2026-10-16）及其核验版
---

# ARX 开门 VLA：数据 v2 方案、已交付代码与后续路线

第一轮微调（Job 4075）**作废**，原因见第一节；它只保留用于复现 `evaluation_20261015_03`
（`configs/policy/arx_knob_n17.json` 已标为 deprecated）。本文给出数据 v2 的设计、
已实现并验证的代码、生产与训练命令，以及 Gemini / DoorMan / AutoData / B2 的历史路线讨论。
当前执行顺序以以下路线更新为准。

**当前目标只有一个：VLA 在仿真里闭环成功开一次门。** 论文和消融暂不考虑；微调固定为全量微调。

> **2026-10-17 路线更新：后续统一以 AutoData 为数据生成主线。**
> 官方源码已作为固定子模块接入 `third_party/AutoData`；目录职责、恢复和升级规范见
> `dependencies/README.md`，具体入口见 `scripts/door/autodata/README.md`。
> 执行顺序调整为：官方 Franka 示例 → ARX 单条生成与物理验收 → 迁移并验证数据 v2 的
> 直线路径、恢复样本、机身视角和阶段标签 → 批量数据 → VLA 全量微调与闭环评估。
> 下文旧 SkillGen 生产命令和“AutoData 放到之后”的安排保留作历史对照，不再作为新工作的优先顺序。
> 当前只完成依赖和入口结构接入，不代表 AutoData 物理验收或数据 v2 迁移已完成。

---

## 一、为什么第一轮数据不能用（全部 120 条复核结果）

| 问题 | 证据 | v2 的处理 |
| :--- | :--- | :--- |
| **自由空间段绕行且多模态** | 120 条中 120 条的第一段 cuMotion 轨迹有单关节越界超调，训练集中位 0.64 rad（37°），路径长度是直线的 1.8–3.3 倍；同一角度下 joint1 先向左或先向右并存（91:9） | 规划器新增 `direct_if_collision_free` 模式：碰撞检查通过时走最小加加速度直线关节路径，否则才退回轨迹优化器；导出时 `--max-transition-overshoot` 拦截残余绕行 |
| **接近段零方差** | 进入抓取前 20 帧，100 条之间关节标准差仅 0.7–2.8°；接触段是同一条刚体变换轨迹 | 新增接近段**恢复示范**：先到一个偏移的入口（±3 cm、±8°，夹爪半闭 14–24 mm），再规划直线修正回到种子接近线 |
| **夹爪状态归一化坍缩** | `state.gripper` 的 q01/q99 = 25.9/28.0 mm，半空 20.7 mm 被截成 −1，等同“已夹住” | 校验时 `--gripper-state-range 0 0.028` 按物理行程归一化；推理侧 `GripperGate` 二值化并去抖 |
| **阶段信息缺失** | `transition_index` 全程为 1（它只计数规划调用） | 将生成轨迹与种子逐帧对齐，继承种子已审核的细分阶段，得到 4 个阶段标签 |
| **语言不一致** | 推理 prompt 与训练文本不同 | v2 训练与推理共用 `arx_door/dataset/stages.py` 中同一份阶段指令 |
| **只有腕部视角** | 唯一的第二视角是世界固定的 overview，上狗后不存在 | 新增**机身固定 ego 相机**，挂在 ARX 基座上，模拟 B2 机身相机 |
| **单种子、窄扇区** | 训练集只有 K001 一个种子，角度 −22.5°~−13° | 第一次成功前先用 K001 把评估分布覆盖密；多种子、宽扇区放到成功之后（第五节） |

---

## 二、数据 v2 设计

### 2.1 每条轨迹的结构

```text
[ 直线自由空间段 ]──►[ (可选) 偏移入口 ]──►[ 直线修正段 ]──►[ 种子技能回放：接近→抓取→解锁→开门 ]
   planned_transition    perturbation(导出裁掉)   recovery            seed_replay
```

- `frame_source` 和 `seed_frame` 逐帧记录在导出数据的 `diagnostics/*.npz` 中。
- 恢复样本在导出时用 `--crop-before-recovery` 裁掉故意制造的偏移段，episode 从偏移位姿开始，
  只保留“从偏处纠回”的部分。
- 旧增广配置（`configs/skillgen/data_v2.json`，已删除）中 `probability: 0.5`，即约一半为名义轨迹、一半为恢复轨迹。

### 2.2 阶段语言（训练与推理同源）

| 阶段 | 覆盖的种子细分阶段 | 指令 |
| :--- | :--- | :--- |
| approach | cumotion_approach_knob, approach_knob | Move the open gripper to the front of the round doorknob. |
| grasp | grasp_knob | Close the gripper around the round doorknob. |
| unlock | unlock_round_knob | Turn the round doorknob counterclockwise until the latch retracts. |
| open | pull_knob_for_gap, pull_door_by_knob, hold_door_open | Keep holding the doorknob and pull the door open. |

推理时由 `RuleStagePrompter` 切换阶段，只用部署时可得的信号：
- 策略的夹爪命令
- 实测夹爪开度
- joint6 转角

完全不读门关节真值。它在导出数据上回放，与对齐标签的逐帧一致率为 97–98%：
- 进入 grasp 晚 3 帧
- 进入 unlock 早约 19 帧（夹爪已稳定、扭转刚开始）
- 进入 open 早约 8 帧

漏掉抓取（夹在空气上）时回到 approach，并要求先重新张开夹爪。

### 2.3 观测与动作

| 项 | v2 |
| :--- | :--- |
| 视频 | `observation.images.wrist` + `observation.images.ego`（均为 640×480，30 Hz） |
| 语言 | `annotation.human.stage_description`（逐帧）；整句任务仍保留在 `task_description` 列，供消融使用 |
| 状态 / 动作 | 不变：7D 绝对值（6 关节 rad + joint7 m）；手臂相对动作、夹爪绝对动作，horizon 16 |
| 夹爪归一化 | state[6] 按 [0, 0.028] m 物理行程归一化 |

---

## 三、已交付代码（全部有测试）

| 功能 | 文件 |
| :--- | :--- |
| 直线关节路径与碰撞检查（cuMotion 原生检查器） | `scripts/door/planner/door_cumotion_worker.py`（`path_mode`） |
| 规划器配置与逐次净空覆盖、恢复扰动与汇入点选择 | 原 `arx_door/skillgen/`，随旧增广从 main 删除（见 git 历史）；增广改由 AutoData 官方 SkillGen 承担 |
| 机身 ego 相机 | `arx_door/scene/ego_camera.py`、`arx_door/envs/door.py`（`ego_camera`） |
| 种子对齐与阶段标签、路径质量指标 | `arx_door/dataset/stages.py` |
| 导出：多视角、阶段语言、恢复裁剪、绕行门控 | `scripts/door/dataset/export_door_lerobot.py` |
| 官方 N1.7 校验：多视角、阶段语言、夹爪归一化 | `scripts/door/dataset/validate_door_n17.py` |
| N1.7 模态配置 | `configs/dataset/door_v2_n17.py`（另有两份消融配置，暂不使用） |
| 阶段提示器离线审计 | `scripts/door/dataset/audit_stage_prompter.py` |
| 推理：夹爪门控、阶段提示、多视角 | `arx_door/policy/guards.py`、`scripts/door/policy/rollout.py`、`server.py`、`client_bridge.py` |
| v2 推理配置 | `configs/policy/arx_knob_v2_n17.json`（checkpoint 路径待训练后填写） |
| Gemini / VLM 审核（可从批次报告按间隔抽样） | `arx_door/vlm/review.py`、`scripts/door/vlm/review_door_episodes.py`、`run_review.sh` |
| 专家动作回放检查（评估环境与生成环境一致性） | `scripts/door/policy/rollout.py --replay-hdf5`、`scripts/door/policy/replay_check.py` |
| 生产计划（导出与验收门槛） | `configs/dataset/v2_first_success_plan.json`；旧续跑执行器 `run_v2_plan.py` 已随旧增广删除 |
| 导出、校验、审计、验收、打包一步完成 | `scripts/door/dataset/build_v2_bundle.py` |
| 全量微调脚本（训练包内使用） | `scripts/door/training/train_v2_full_n17.sh`、`.sbatch` |
| 推理服务指定 checkpoint 与 holdout | `run_policy.sh serve --checkpoint … --dataset …` |

### 仿真端到端验证（2026-10-17，本机，强制恢复）

| 角度 | 物理验收 | 自由空间段 | 恢复段 |
| :--- | :--- | :--- | :--- |
| −17.5° | 门 33.9°、旋钮 44.6°、锁舌 15.9 mm、非手指接触 0 N、指力 25/45 N | 路径比 1.002，超调 0.002 rad（n17 中位 0.64） | 26 帧直线，比 1.003 |
| −21.0° | 门 34.0°、旋钮 44.7°、锁舌 15.9 mm、0 N、26/44 N | 1.002 / 0.002 rad | 27 帧直线，比 1.003 |

- 两条都导出为双视角、阶段语言数据，并通过官方 N1.7 loader 与统计校验。
- 第一次运行暴露了两个问题，均已修复：
  - 折叠起始位姿离底座支撑仅 14 mm，原先的 15 mm 净空门槛使直线模式从不生效，已改为 5 mm。
  - cuMotion 的保守球体会把靠近旋钮的接近路点判为碰撞，汇入点现会自动回退到更早的路点。
- 修复前的恢复修正由轨迹优化器生成，路径比 5.7，绕行 0.31 rad；修复后为直线。
- 这正说明导出端的绕行门控是必要的。

---

## 四、到第一次成功开门的路线

```text
① 生产数据（Gemini 按手册执行，本机约 9 h）
   └ docs/vla/data_v2_runbook_for_gemini.md
② 训练包 → 服务器全量微调（LLM + 视觉 + projector + 动作头）
③ checkpoint 拿回本机 → 闭环评估 4 个分布内场景
④ 第一次成功之后：闭环自演进（Filtered BC）→ 随机化 / 泛化
```

### 4.1 为什么现在不迁 AutoData、不加 DoorMan 随机化

- **AutoData**：底层就是 SkillGen 的同一套方法，它增加的是吞吐量，不是数据质量。它目前还跑不起来（缺镜像、缺固定版本），迁过去还要重写 v2 的全部修复。现有流程单条 80 秒，数据量够用。等到需要大规模多环境生成或换到 B2 时再迁。
- **DoorMan 随机化**：评估场景本来就在训练分布内，第一轮恰恰是在分布内失败的。现在加随机化只会增加学习难度。DoorMan 对当前阶段最有用的部分是纠偏数据，v2 的恢复示范就是它的离线版本。

### 4.2 本次生产计划（`configs/dataset/v2_first_success_plan.json`）

| 批次 | 种子 | 角度 | 半径 | 外观 | 条数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| train_r0590 / r0597 / r0603 / r0610 | K001 | −22.5°~−13°，步长 0.5° | 0.590 / 0.597 / 0.603 / 0.610 | 1–4 | 4 × 80 |
| holdout_r0598 | K003 | −27°~−23° | 0.5975 | 21–24 | 40 |

- 评估场景（−19°、−17.5°、−16°，外观 1–4，半径 0.595–0.605）全部落在训练覆盖范围内。
- 约一半试验带恢复扰动。扰动量按（试验种子、角度、外观）逐条独立抽取。
  - 早先版本存在一个 bug：同一批次的所有试验共用同一个扰动，现已修复并加了测试。
- 验收门槛（打包时强制检查）：
  - 训练 ≥ 240 条，holdout ≥ 24 条
  - 恢复样本占 30–70%
  - 阶段提示器一致率 ≥ 0.9

### 4.3 服务器全量微调（训练包内 `README.md`）

| 参数 | 默认 | 说明 |
| :--- | :--- | :--- |
| 训练模块 | LLM、视觉、projector、扩散动作头全部训练 | 按你的要求 |
| GPU | 4 张，每张 ≥ 80 GB | NVIDIA 官方说明：开启 tune-llm/visual 时每卡需要 80 GB 以上；脚本检测到显存不足会拒绝运行 |
| 全局 batch / 步数 / 保存间隔 | 64 / 15000 / 2500 | 显卡较小时减小 batch、增大梯度累积，保持乘积为 64 |
| 学习率 | 5e-5 | **不是官方数值**（官方没有给出全量微调的学习率），取官方默认 1e-4 的一半 |
| 代码版本 | Isaac-GR00T `d2b7e75` | 与本机推理服务一致 |

脚本会先核对：训练包哈希、官方校验结果、夹爪按 0–28 mm 归一化（已确认服务器重新生成统计时不会覆盖）、GPU 数量与显存。任一不符即拒绝启动。

### 4.4 checkpoint 回来之后（本机）

所有 `checkpoint-*` 都要拿回来：离线 loss 不能预测闭环效果，所以 5000、10000、15000 步都要评估。

```bash
# 终端一：推理服务（每个 checkpoint 用新的 output 目录；warm-up 用的 holdout 默认取训练包内的路径）
bash scripts/door/policy/run_policy.sh serve --config configs/policy/arx_knob_v2_n17.json \
  --checkpoint <checkpoint-10000> \
  --output generated/door_policy/runs/v2_server_ck10000
# 终端二：4 个分布内场景
python3 scripts/door/policy/evaluate.py --config configs/policy/arx_knob_v2_n17.json \
  --scenarios baseline position_left position_right appearance --headless
```

- 失败的 rollout 用 `bash scripts/door/vlm/run_review.sh <rollout 目录>` 生成接触表，交给 Gemini 判断失败类型。
- 阶段切换记录在 `result.json` 的 `stage_transitions` 里；原始夹爪命令记录在 `trajectory.npz` 的 `raw_gripper_command` 里。
- 两者结合可以直接看出策略是卡在接近、抓取还是开门阶段。

---

## 五、第一次成功之后

1. **闭环自演进（Filtered BC）**：用新策略跑几百次随机摆位，只保留通过物理验收的 rollout，与专家数据混合后再训练。rollout 已记录所需的全部字段，转 LeRobot 的转换器届时再写。
2. **DoorMan 式扩展**：门动力学随机化（合页阻尼、锁舌弹簧）、更宽的角度和半径、更多种子，以及专家从失败状态接管（DAgger）。
3. **AutoData 迁移**：吞吐量成为瓶颈时再做，必须带上直线规划、恢复扰动、阶段对齐和机身相机。
4. **B2+ARX**：整机模型在 `assets/b2arx_robot`（B2 搭载 ARX R5A），控制按 12 自由度腿策略加 6 自由度臂命令分离。VLA 一次输出的 16 步动作块（0.53 s）可以直接作为腿策略的臂命令预览。机身相机要按 B2 实机的位姿重新标定，再重新渲染数据。

---

## 六、执行清单

- [x] 数据 v1 全量复核，作废 Job 4075
- [x] 数据 v2 代码、测试，以及仿真端到端验证（生成 → 导出 → 官方校验 → 评估环境回放）
- [x] 生产计划、续跑执行器、打包器、全量微调脚本、Gemini 执行手册
- [ ] Gemini 按 `data_v2_runbook_for_gemini.md` 生产并打包
- [ ] 服务器全量微调
- [ ] 本机闭环评估各个 checkpoint
- [ ] 第一次成功 → 第五节
