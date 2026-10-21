# ARX 开门的 AutoData 增广

生成循环、SkillGen 算法、源数据池、cuRobo 规划器和 HDF5 记录全部是 AutoData 官方代码
（`third_party/AutoData`，版本见 `dependencies/autodata.lock.json`）。本项目只按官方扩展点提供：

| 官方扩展点 | 本项目提供 |
|---|---|
| `--external_callback`（注册环境） | `arx_door.autodata.register:register`，构建 `arx_door/envs/door.py`（Arena，`ik_abs` 动作） |
| `--task_descriptor` | `configs/autodata/tasks/arx_door_knob_hold_open.yaml` |
| `--embodiment`（官方 `absolute_pose` 类型） | `configs/autodata/embodiments/arx_r5a.yaml`：link6 绝对位姿（机器人根坐标系）+ 夹爪 |
| `--env_profile` → `PLANNER_PROFILES` | `configs/autodata/env_profiles/arx_door.yaml` → `arx_door/autodata/planner.py`（由 `assets/arx_r5a` 的 URDF+XRDF 生成 cuRobo 模型） |
| `--recorder_cfg` | `arx_door.autodata.register:recorder_cfg`：腕部相机与物理验收记录 |
| `--input_file`（标注源数据） | `convert_seed.py`：专家种子 → 机器人根坐标系的 AutoData 源 |

## 流程

```text
示范种子 (scripts/door/demos)  ─┐
                                ├─ convert_seed.py ─→ source.hdf5 ─┐
种子场景 ── prepare.py ─→ 资产包 manifest.json ─────────────────────┼─ generate.sh（官方 generate_dataset.py）
                                                                     └─→ generated.hdf5 ── accept.py ─→ accepted.hdf5
```

1. 资产包：从种子所在场景导出机器人、门、工作台和地面（不改源 USD）。

   ```bash
   generated/runtime/isaac61/.venv/bin/python scripts/door/autodata/prepare.py \
     --scene <种子场景>.usd --output generated/door_autodata/assets_<名称>
   ```

2. 源数据：把专家种子（关节动作、世界坐标）转成官方 `absolute_pose` 的动作和位姿。
   转换会校验每帧反推的基座位姿恒定、且与资产包中的机器人位置一致，不一致就拒绝。

   ```bash
   generated/runtime/isaac61/.venv/bin/python scripts/door/autodata/convert_seed.py \
     --seed generated/door_teaching/<批次>/seeds/<种子>.hdf5 \
     --assets generated/door_autodata/assets_<名称>/manifest.json \
     --output generated/door_autodata/sources/<种子>.hdf5
   generated/runtime/isaac61/.venv/bin/python scripts/door/autodata/preflight.py --source-only \
     --input generated/door_autodata/sources/<种子>.hdf5
   ```

3. 容器：按 AutoData 官方 Dockerfile 加项目构建补丁安装，包含其锁定的 Sim、Lab、Arena 和 cuRobo。
   `dependencies/patches/autodata-early-runtime-constraints.patch` 将版本约束提前到 Lab 安装前，
   避免安装过程中被新 TensorDict 升级到另一套 PyTorch/CUDA，并补齐 GR00T 轻量 RPC 客户端的
   `msgpack` / `msgpack-numpy` 依赖；不改官方源码。
   项目以只读方式挂载在 `/workspaces/arx`，`generated/door_autodata` 挂载为 `/outputs`；
   官方脚本和示例数据单独只读挂载，Kit 缓存保存在 `generated/runtime/autodata_cache`。

   ```bash
   bash scripts/door/autodata/run_container.sh --build-only          # 只安装，不打开终端
   bash scripts/door/autodata/run_container.sh --official --start   # 先跑官方 Franka 例子
   bash scripts/door/autodata/run_container.sh --start              # ARX 项目
   ```

   第一次验证官方 SkillGen 示例（宿主机执行，单环境、单条，不用 ARX 适配）：

   ```bash
   bash scripts/door/autodata/run_container.sh --exec /isaac-sim/python.sh \
     /workspaces/autodata/scripts/generate_dataset.py \
     --env_name IsaacContrib-Stack-Cube-Franka-IK-Rel --alg skillgen \
     --task_descriptor /workspaces/autodata/autodata_examples/tasks/franka_cube_stack_skillgen.yaml \
     --embodiment /workspaces/autodata/autodata_examples/embodiments/franka_ik_rel_skillgen.yaml \
     --input_file /workspaces/autodata/autodata_tests/test_data/annotated_dataset_franka_stack_skillgen.hdf5 \
     --output_file /outputs/franka_skillgen_first.hdf5 \
     --result_file /outputs/franka_skillgen_first.json \
     --generation_num_trials 1 --num_envs 1 --device cuda --viz none
   ```

4. 生成（容器内）：

   ```bash
   bash /workspaces/arx/scripts/door/autodata/generate.sh \
     --assets /outputs/assets_<名称>/manifest.json \
     --input /outputs/sources/<种子>.hdf5 \
     --output /outputs/runs/<运行名> --trials 1 --num-envs 1
   ```

   这只是在官方 `generate_dataset.py` 外面填好 ARX 的参数，`--` 之后的参数原样传给注册回调。
   `--seed` 写入本次运行的 `task.yaml`；`run.json` 同时保存种子和资产清单的
   `trial_parameters`。改变种子本身不证明数据发生了增广。

5. 物理验收：只保留整段满足严格门槛的片段。

   ```bash
   generated/runtime/isaac61/.venv/bin/python scripts/door/autodata/accept.py generated/door_autodata/runs/<运行名>
   ```

   回放复核也使用同一个容器，自动继承 `run.json` 的资产和物理配置，按记录的动作维度选择控制模式：

   ```bash
   # 以下命令在宿主机执行；--exec 使用已构建的固定版本镜像。
   bash scripts/door/autodata/run_container.sh --exec /isaac-sim/python.sh \
     /workspaces/arx/scripts/door/policy/replay_check.py /outputs/runs/<运行名>
   # GR00T 模型服务仍单独启动；容器通过 host 网络连接，容器内只运行官方 RPC 客户端。
   bash scripts/door/autodata/run_container.sh --exec /isaac-sim/python.sh \
     /workspaces/arx/scripts/door/policy/evaluate.py --headless --scenarios baseline
   ```

   默认评估产物位于 `/outputs/policy`。如指定 `--output`，须使用可写的 `/outputs` 路径。
   不能再通过旧的 `generated/runtime/local.json` 宿主机 Python 启动 Arena 回放。
   容器的源版本证据记录为启动器核对过的镜像构建标签，区别于旧运行时的 Git 检查。

安装依据：[AutoData 官方安装](https://isaac-sim.github.io/AutoData/main/pages/quickstart/installation.html)、
[Arena 官方安装](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/quickstart/installation.html)。
Arena 网页的独立安装版本与 AutoData 固定组合有差异，本项目使用 AutoData 锁文件和 Dockerfile 的组合。

## 物理验收门槛

最终门角 `>29°`、手臂接触峰值 `<=2N`、相机与门板间隙 `>=0.01m`、双指接触各 `>=3N` 且方向相反，
抓握→解锁→开缝的顺序正确，并且从开缝到结束持续握住把手。在 30Hz 的动作前后采样检查，
不宣称覆盖 120Hz 的每个物理子步。AutoData 锁存环境的 `success`（即当前验收结果）；
违规不可逆，所以 `accept.py` 以每段最后一帧的 `obs/diagnostics/runtime_accepted` 为最终结论。

## 现状

已离线验证：任务、机器人、环境 YAML 能被官方加载器读取并互相匹配；K001 种子转换后能被官方
`DataGenInfoPool` 读取（子任务区间 406–1360），官方动作编解码与 `target_eef_pose` 误差 ~1e-7；
基座位姿与资产包一致（~4e-8 m）。

2026-10-18 已完成本机镜像构建并启动容器：Arena 0.3.0、Lab 3.0.0、PyTorch 2.11.0+cu128、
NumPy 2.3.1，包含编译完成的 cuRobo。官方 `6.1.0` 镜像的实际 Sim 构建号为
`6.1.0-rc.26+release.49347.2d230af4.gl`，按原样记录，不推断为其他发行构建。
实际 XRDF 转换和容器运行时/回放入口测试 9 项通过；官方 GR00T RPC 的双视角观测与动作往返通过
（合成动作，未加载模型）；K001 资产包在容器内成功导出。宿主机相关回归 29 项通过。
详见 `generated/runtime/autodata_setup/SETUP_REPORT.md`。

2026-10-18 后续物理验证位于 `generated/door_autodata/pilot_20261018`：K001 全长
1360 帧专家回放通过；`generated_baseline_10` 是官方 SkillGen 生成的 987 帧轨迹，
物理验收、末端动作独立回放、7 维关节目标独立回放均通过。随后窄范围试产共 5 条、
4,935 帧全部通过这三项检查及官方 N1.7 / LingBot 数据加载；训练 3 条、留出 2 条。
结果见同目录的 `pilot_summary.json`、`PILOT_REPORT.md`，交付位于 `delivery`。
没有完全重复轨迹，但 96.66% 的末端交互动作复用同一 K001；这不代表具备微调所需数据量。
官方 Franka、多环境、服务器全量微调和 VLA 闭环仍须单独验证。

### 当前固定场景的规划适配

- cuRobo 网格导入使用其已有的 CPU 三角化回退，避开当前 Warp 三角化崩溃；CUDA 规划保持启用。
- URDF 的 `joint8` 是 mimic，仅独立锁定 `joint7`；仿真仍驱动双指。
- 固定 `base_link` 的近似球体穿入安装支柱，导致起始姿态被拒绝。当前闭门、固定基座
  的自由空间规划豁免固定底座的世界碰撞；移动连杆的世界碰撞和底座的自碰撞有效半径保持原值。
  逐障碍物证据与半径比较见 `generated_baseline_09/planning_failure.json`、
  `fixed_base_planning_exception.json`。不适用于移动基座或开门后的再次规划。
- 项目层在规划首次失败时退出并保存诊断，避免官方循环在第一帧动作前失败后无限重置。
  不修改官方子模块文件，也不改变物理验收条件。

## AutoData → 训练数据

AutoData 动作为根坐标系下 `xyz + quaternion_xyzw + joint7`，在 30 Hz 输入；
差分 IK 在每个 120 Hz 物理子步重新计算关节目标。记录器保存动作前的双视角 RGB、
关节状态和动作后的物理诊断，以及四个实际子步目标 `joint_target_substeps`。
关节目标不是下一帧实测关节位置。

先对通过物理验收的末端动作做 `replay_check.py`，再准备独立的关节动作候选：

```bash
/isaac-sim/python.sh scripts/door/autodata/prepare_joint_replay.py /outputs/runs/<运行名>
/isaac-sim/python.sh scripts/door/policy/replay_check.py \
  /outputs/runs/<运行名>/joint_replay_candidate --output /outputs/<关节回放目录>
```

候选使用每个控制周期的首个实际关节目标并保持四个物理子步，两种控制器并不恒等。
`joint_replay_candidate/accepted.hdf5` 只是回放输入文件名，不能据此宣称候选已获准训练。
必须完整通过关节回放，导出器还会记录两种控制器的逐关节状态误差。

导出清单是 JSON 列表，每项包含 `run`、`split`（train/holdout）、
`pose_replay_summary` 和 `joint_replay_summary`。仅导出清单对应的当前 AutoData 数据：

```bash
bash scripts/door/dataset/run_dataset_tool.sh export_autodata_lerobot.py \
  --manifest <清单.json> --split train --output <新的输出目录>
bash scripts/door/dataset/run_dataset_tool.sh validate_door_n17.py \
  --dataset <输出目录> --groot-root generated/runtime/groot_policy/source \
  --config configs/dataset/door_v2_n17.py \
  --official-commit d2b7e75b937e3ec9aa5dbc798f08b89692c49734 --gripper-state-range 0 0.044
```

磁盘动作是六个绝对关节角（rad）和单侧夹爪位移（m），不预归一化。
N1.7 使用 16 步相对手臂动作、绝对夹爪；阶段语言通过完整源交互段逐帧匹配恢复。
当前匹配仅支持固定机器人/门根位姿的窄范围试产，改变根位姿需要扩展并验证坐标变换。
训练和留出数据按种子及初始条件分开；留出统计不能用于训练归一化。

LingBot 使用相同数据的独立 LeRobot v3 副本，保留 v2.1 原件。使用 LeRobot 官方
`lerobot.datasets.v30.convert_dataset_v21_to_v30`，本地转换传入 `--push-to-hub false`；
特征映射在 `configs/dataset/arx_autodata_lingbot.yaml`，读取检查在
`scripts/door/dataset/validate_lingbot_dataset.py`。这不等于模型训练或推理验证。

```bash
bash scripts/door/dataset/run_lingbot_tool.sh convert_lingbot_dataset.py \
  --source <v2.1目录> --output <新的v3目录>
bash scripts/door/dataset/run_lingbot_tool.sh validate_lingbot_dataset.py \
  --dataset <v3目录> --source-v21 <v2.1目录> \
  --lingbot-root generated/runtime/lingbot_data/source \
  --robot-config configs/dataset/arx_autodata_lingbot.yaml --output <验证结果.json> \
  --norm-stats <训练集统计.json> --fit-norm-stats
```

该隔离数据运行时的版本清单为 `generated/runtime/lingbot_data/runtime_versions.json`。
拟合统计时 v3 目录名须为 `train`；验证留出集时去掉 `--fit-norm-stats`，复用同一统计文件。
手臂相对动作保留 16 步维度，分位数使用训练集精确计算；单侧夹爪状态和动作均显式使用
物理行程 `[0,0.044]` m，而不是该批样本的窄观测范围。官方 `Normalizer` 检查全部数值行的
归一化与反归一化。验证不下载模型；不执行图像/语言模型预处理、模型前向或训练。

## 补采新的程序化专家种子

`collect_expert.py` 在当前固定 Arena 环境中执行 LLM 辅助编写的几何/反馈专家程序，
不读取既有示范动作。它是项目的专家采集入口；官方 AutoData 负责读取标注后的专家源并
执行 SkillGen 增广。当前固定版本未提供专门的 LLM 从零开门专家生成接口。

两种参数配置为 `measured_arc` 和 `staged_arc`：根据资产的 `door_handle/grasp_target`
计算抓取中心，以末端目标、差分 IK 和真实接触执行解锁、绕铰链拉门。初始自由空间
接近使用 IK 求解后的平滑关节路径；它本身不提供几何碰撞规划证明，必须通过实际
接触/相机间隙监测及后续回放。官方 SkillGen 的自由空间过渡仍由现有 cuRobo 适配器规划。

```bash
bash scripts/door/autodata/run_container.sh --exec bash -lc '
  export PYTHONPATH=/workspaces/arx:/workspaces/autodata
  /isaac-sim/python.sh scripts/door/autodata/profile_command.py \
    --output /outputs/my_new_expert_profile -- \
    /isaac-sim/python.sh scripts/door/autodata/collect_expert.py \
      --assets /outputs/setup_20261018/assets/manifest.json \
      --output /outputs/my_new_expert --strategy measured_arc --seed 101 --headless
'
```

每次使用新目录。输出包含完整 HDF5、三视角视频、逐阶段 `controller_events.jsonl`、
实际使用的控制器源码，以及 1 秒采样的 `resources.jsonl` 和进程计时汇总。失败也保存。
内存口径是进程树 RSS（共享页可能重复），显存口径是整卡使用量；cgroup current 另含页缓存，
原始 memory.peak 属于容器生命周期。采样峰值不能当成连续精确峰值，命令退出检测分辨率为 1 秒。

成功种子仍须依次运行 `accept.py`、`replay_check.py`，并通过
`prepare_joint_replay.py` 后的独立关节回放。`pack_expert_source.py RUN` 将通过物理验收的
完整记录提取为轻量 `source.hdf5`，供官方 DataGenInfoPool 使用；原始双视角图像、关节目标
和诊断仍保留在 `accepted.hdf5`。轻量源不是训练数据替代品。继续用官方
`scripts/validate_dataset.py` 检查结构，再以 `source.hdf5` 运行本目录 `generate.sh`。
两种配置是同一控制方法下不同接近/接触/拉门参数的独立物理采集，不能宣称两种算法或场景随机化。
