# GR00T N1.7：模型返回后的推理与闭环评估

这是开门流水线的**策略评估**入口。场景、物理和动作映射复用原 ARX 环境，
模型推理使用 NVIDIA 官方 `Gr00tPolicy`、`PolicyServer` 和 `PolicyClient`。
不会调用 SkillGen、cuMotion 或示范回放来产生 rollout 动作。

当前官方源码固定为训练使用的 `d2b7e75b937e3ec9aa5dbc798f08b89692c49734`。
本入口在固定 AutoData 容器内构建 `arx_door.envs.door` Arena 环境，与生成端共用定义；
它不是 Arena Experiment Runner 的迁移验收。默认使用 v2 配置，需先准备相应 checkpoint。
复现旧 Job 4075 必须显式传入 `--config configs/policy/arx_knob_n17.json`。
Arena 现有 N1.6 DROID 示例不能仅替换模型路径后直接用于 N1.7 ARX。

## 按功能找文件

| 要做的事 | 入口 / 位置 |
|---|---|
| 安装独立推理环境 | `setup_runtime.py` |
| 本机 GPU 加载与真实观测推理 | `run_policy.sh verify --output <新目录>` |
| 启动模型服务 | `run_policy.sh serve --output <新目录>` |
| 固定场景闭环 / 变体 / 随机场景 | `evaluate.py` |
| 模型、模态、语言、执行节奏 | `configs/policy/arx_knob_v2_n17.json`（数据 v2：wrist+ego、阶段语言、夹爪门控）；`arx_knob_n17.json` 为已作废的 Job 4075，仅用于复现旧评估 |
| 夹爪二值门控、阶段提示器 | `arx_door/policy/guards.py` |
| 位置/外观变化与随机范围 | `configs/policy/evaluation_scenarios.json` |
| 可复用协议与运行辅助 | `arx_door/policy/` |
| 模型、数据和原训练报告统一入口 | `generated/door_policy/{models,datasets,training_results}` |
| 每次评估结果、录像和失败记录 | `generated/door_autodata/policy/`（容器内 `/outputs/policy/`） |

从**项目根目录**运行下列命令。每次使用新输出目录，避免覆盖已有证据。

## 运行模型与基准评估

终端一：

```bash
bash scripts/door/policy/run_policy.sh serve \
  --output generated/door_policy/runs/my_policy_server
```

服务先用 3 帧真实 holdout 观测验证 GPU 推理与 16×7 输出、保存 `receipt.json`，
再在 `127.0.0.1:5567` 服务。`POLICY_SERVER_READY` 表示可以启动仿真。
服务与仿真在不同 Python 环境中；容器里的轻量 IPC bridge 仅加载官方 PolicyClient，
通过 host 网络连接模型服务。协议不重复对 relative 动作加 state：
官方 `decode_action` 已返回绝对目标。

终端二：

```bash
bash scripts/door/autodata/run_container.sh --exec /isaac-sim/python.sh \
  /workspaces/arx/scripts/door/policy/evaluate.py --headless --scenarios baseline
```

以上命令无窗口运行一次最多 1800 步 / 60 秒仿真时间的闭环尝试。
物理 120Hz、控制 30Hz；模型每次预测 16 步，仅执行前 8 步后重新观察。
推理等待期间物理暂停，这是同步仿真评估，不证明实机实时控制能力。
当前容器启动器用于 headless 验证，录像保存在每次运行目录中。

模型服务加载后，评估器要求至少 6GiB 剩余显存才启动渲染。旧运行时一模型一场景
曾观测到合计约 11.2GiB 显存占用，该数值尚未在新 Arena 容器中复核。这个前置余量不保证其他任务不会随后占用显存。

## 场景变化与自动随机化

先跑固定场景，确认观测/动作接口正常，再分别改变位置和外观：

```bash
bash scripts/door/autodata/run_container.sh --exec /isaac-sim/python.sh \
  /workspaces/arx/scripts/door/policy/evaluate.py --headless \
  --scenarios position_left position_right appearance
```

自动采样 5 个场景，并保留基准作为对照：

```bash
bash scripts/door/autodata/run_container.sh --exec /isaac-sim/python.sh \
  /workspaces/arx/scripts/door/policy/evaluate.py --headless --scenarios baseline --random-episodes 5
```

采样范围、seed 与外观候选位于场景配置中。每次完整保存采样清单 `scenarios.json`。
位置表示机器人相对门旋钮的角度/半径；门的世界坐标、铰链和相机安装不随之改变。
每个变体在新 USD 中构建，以保留资产、物理和可复现性；原场景不修改。
不同于训练数据生成，此处模型必须自己完成全部动作。采样范围仅是配置，
不能据此声称所有采样都物理可行或已评估。

## 如何看结果

- `summary.json`：请求/尝试/实际完成数，每个场景结果与成功数。资源/接口错误
  不计为完成的任务评估；`evaluation_complete=false` 明确表示未完成。
- `<场景>/simulation.log`：模拟器日志；`rollout/client.log` 为官方策略客户端日志。
- `<场景>/rollout/result.json`：终止原因、门角、双指接触、非手指接触峰值、
  相机间隙、解锁/门缝顺序、持续握持等证据。
- `trajectory.npz`：pre-state、原始命令、post-state、门状态、接触、相机间隙、
  完整预测 chunks 和每次推理耗时。所有数组无需 pickle。
- `wrist.mp4` / `overview.mp4`：完整执行录像，30fps，每步一帧。

生成数组/视频检查与图文报告：

```bash
generated/runtime/groot_policy/.venv/bin/python scripts/door/policy/report.py \
  generated/door_autodata/policy/<本次评估目录>
```

使用既有严格物理门槛：最终门角 >29°、双指力各 ≥3N 且方向相反、非手指
接触峰值 ≤2N、相机间隙 ≥1cm、抓握→解锁→门缝的时序成立，首次门缝后
持续抓住旋钮；全部满足后再持续 30 步。检查覆盖 30Hz pre/post-action 样本，
不宣称每个 120Hz 子步都已审计。任务失败也保留轨迹和录像。

## 已有训练与离线评估

job 4075 已完成 10000 步。job 4127 已完成 20 条 / 26903 帧开环评估，
arm MAE 0.00548812 rad、夹爪 MAE 0.000138544m。原始报告通过
`generated/door_policy/training_results/训练结果分析.md` 可达。
这些是命令预测误差，不能解释成闭环成功率。

## 新机器准备

已有官方源码归档、最终 checkpoint 和原数据时：

```bash
python3 scripts/door/policy/setup_runtime.py \
  --source-archive <Isaac-GR00T-d2b7e75-latest-with-git.tar.gz> \
  --download-base-model
```

使用官方锁文件安装 PyTorch 推理依赖；跳过 TensorRT 三个加速包，避免为尚未进行
的引擎导出下载数 GB 的 TensorRT 库。首次官方全量安装中该下载多次超时；PyTorch
推理不依赖它们。不会改 `groot_data`、Isaac Sim 或系统驱动。

基础模型下载使用用户已有 HF 授权，不复制 token。也可把同一 revision 的完整
基础 snapshot 放入配置指定的 HF 缓存。模型/数据软链接在本机已建立；迁移时需
按实际存放位置修改配置并恢复引用，不能只复制软链接。

官方参考：[Policy API](https://github.com/NVIDIA/Isaac-GR00T/blob/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/getting_started/policy.md)、
[部署闭环](https://github.com/NVIDIA/Isaac-GR00T/blob/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/getting_started/real_world_deployment.md)、
[Arena Variations](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/concepts/variations/variations.html)。
