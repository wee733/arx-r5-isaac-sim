# SkillGen 单腕部开门数据 → GR00T N1.7

本任务使用官方 Isaac Lab SkillGen 将种子中的接触技能迁移至不同基座位置，
通过项目 cuMotion 适配器重新规划自由空间路径，每条重新执行物理并渲染。
不是复制视频或直接向关节标签添加噪声。最终用途是 GR00T VLA 微调，无 RL。

## 数据与动作契约

- 训练视频仅 `observation.images.wrist`，640×480 RGB，30 Hz，H.264/yuv420p。
  overview 视频只用于审查，不作为第二路策略输入。
- `observation.state`：动作执行前的实测 joint1..joint6（rad）及主动 joint7（m）。
- `action`：本步最终绝对控制目标，同一顺序和单位，来自 HDF5 `processed_actions`。
  joint8 是 mimic joint，不进入 7D 数据。不能用下一帧反馈位置替代命令。
- 时间戳从零开始，按每帧 1/30 秒递增；不做图像与动作的独立抽帧/重采样。
- 文件保存绝对值，官方 processor 计算手臂未来 16 步相对当前 state 的目标；
  夹爪仍为绝对位置。这不等于相邻 command 的差分。
- 任务文本经 `annotation.human.task_description` 指向 `meta/tasks.jsonl`。

用户已确认单腕部训练。旧通用服务器链路的 `front+wrist` 配置与此数据集不同，
不能直接使用旧 `prepare-arx` 双相机校验或靠复制 wrist 伪造 front。
本项目提供独立 `configs/door_wrist_n17.py`，不改旧服务端配置。

## 1. 物理增广

从仓库根目录使用 Isaac Lab Python，输出目录必须未被使用：

```bash
PYTHONPATH=arx_r5_isaac_sim_bringup \
  /home/lbz/miniforge3/envs/isaaclab/bin/python scripts/run_door_skillgen_batch.py \
  --output-root generated/door_skillgen/my_batch \
  --angles 0 0.5 1 1.5 2 2.5 3 3.5 4.5 5
```

半径保持 0.59 m、高度 0.63 m，相机保持 link6 向下 30°。
本次 pilot 在 0°～+5°附近插值，不代表解决了负偏移或覆盖了 ±30°。
每个位置会生成 `result.json`、HDF5、wrist/overview MP4、场景和规划诊断。
以 result 和数据验收为准，不能只检查 Isaac Sim 退出码。
单卡先顺序执行。本次曾尝试双仿真并行，因 SkillGen 保留整条 RGB observation
导致显存峰值超限；生成入口现将 RGB 直接写 MP4，不再在生成器 observations
保留重复图像，HDF5 数值观测不变。失败/中断尝试保留，不计入训练集。

### 腕部视觉验收

当前固定版本中，相机写入桥接使用 USD 父节点变换，而 GPU articulation 的
USD 位姿可能过期，导致动作成功但腕部画面朝向地面。GPU 场景仍保留 Fabric
以实时渲染门和机器人；另将原相机复制到静态 `/World/SkillGenWristCamera`。
每个动作执行前使用实时 link6 位姿乘以原始安装外参同步独立相机，显式渲染，
并刷新相机缓存。此渲染不推进物理时间。原相机仅作安装外参来源，不改其挂载。
同步还调用 Fabric 的 `set_world_xform`，更新层级 localMatrix；仅写 worldMatrix
会被下一次层级更新覆盖。实现参考 [NVIDIA Fabric 层级接口](https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.6.1/fabric_hierarchy.html)。
场景预处理将相机四元数规范为 double，以兼容 Isaac Lab 位姿写入。

每个视频帧保存实际/期望相机 SE(3) 至 HDF5 diagnostics，检查位置误差 ≤0.1 mm、
角度误差 ≤0.001 rad，并在 `result.json/camera_verification` 保存逐帧覆盖和最大误差。
导出拒绝缺失这些证据的旧生成数据；位姿检查之外还需抽查接触和换抓阶段 RGB。
旧 `n17_wrist_pilot3_20260916` 虽通过格式加载，但已因视角错误隔离，不可训练。
`camera_verified_20260916` 也是不可训练诊断：关闭 Fabric 虽修正了相机，却使
GPU 场景动态物体渲染冻结。导出要求 v2 证据，拒绝 v1 / Fabric=false 数据。

## 2. 导出 LeRobot v2.1

导出环境需 numpy、h5py、pyarrow，以及 ffprobe/ffmpeg：

```bash
generated/runtime/data_export/bin/python scripts/export_door_lerobot.py \
  --batch-report generated/door_skillgen/my_batch/batch_results.json \
  --output generated/door_datasets/my_dataset
```

`--batch-report` 可以重复；也可重复使用 `--generated-attempt <trial目录>`。
批次中明确失败的尝试会被排除并记入报告，显式指定失败 trial 则拒绝导出。
不能重复传入同一源或重打包相同数值轨迹来凑数。旧 `--attempt` 仅接收已审核
NPZ 示范，保留其原来的审核门槛。

导出目录：

```text
dataset/
├── data/chunk-000/episode_XXXXXX.parquet
├── videos/chunk-000/observation.images.wrist/episode_XXXXXX.mp4
├── diagnostics/episode_XXXXXX.npz
├── meta/
│   ├── info.json / modality.json / episodes.jsonl / tasks.jsonl
│   ├── episodes_stats.jsonl / stats.json
│   └── door_export_report.json
└── README.md
```

此时仅完成格式转换；basic stats 不作为官方训练统计，加载验收标记仍为 false。
自动物理验收通过不是新用户审核，原始种子的 approval 不会被导出器改写。

## 3. 官方统计与加载验收

固定 NVIDIA Isaac-GR00T 源码 commit：
`9c7e746b2cd37a810070a98ef41d290a07e806c2`。
验证器拒绝不同 commit 或被修改的 `gr00t/` 源码。

数据验收不需要模型权重，可在独立 Python 3.12 CPU 环境运行：

```bash
python3.12 -m venv generated/runtime/groot_loader312
generated/runtime/groot_loader312/bin/python -m pip install \
  torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
generated/runtime/groot_loader312/bin/python -m pip install -r requirements-groot-data.txt

generated/runtime/groot_loader312/bin/python scripts/validate_door_n17.py \
  --dataset generated/door_datasets/my_dataset \
  --groot-root /path/to/pinned/Isaac-GR00T \
  --config configs/door_wrist_n17.py
```

需要系统提供 FFmpeg 4–7 动态库以供 torchcodec0.8 解码。本机采用 CPU
torch2.9/torchcodec0.8 数据子集环境；正式训练仍使用官方完整锁定 GPU 环境。

验证器会保存旧统计至 `meta/stats_history/`，重新调用官方统计实现生成
`stats.json` 和 `relative_stats.json`，不复用可能过期的内容缓存；然后：

1. 用官方 `LeRobotEpisodeLoader` 逐条读取，检查所有行的状态、动作、索引和文本。
2. 每条用官方 torchcodec 解码首帧、中间帧、末帧；完整视频帧数由导出器 ffprobe 检查。
3. 用官方 `RelativeActionLoader` 验证全部 `(16,6)` 窗口，逐元素与
   `q_cmd[t:t+16,:6] - q_state[t,:6]` 比较；检查绝对夹爪不被变换。
4. 保存输入内容哈希、代码版本、依赖版本和检查覆盖到 `meta/n17_validation.json`；
   通过后才将 export report 的 `official_n1_7_loader_verified` 改为 true。

通过后会把单腕部配置复制到 `training/door_wrist_n17.py`，数据集可整体传输。
重新验收开始即清除旧通过标记，失败不会继续沿用先前通过结论。

## 4. 微调交接（本轮不运行）

将整个 dataset 拷至训练服务器，在固定版本官方 GPU 环境内使用：

```bash
python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /data/arx_door_dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /data/arx_door_dataset/training/door_wrist_n17.py \
  --num-gpus 1 \
  --output-dir /data/arx_door_smoke \
  --max-steps 2 --save-steps 2 --global-batch-size 1 \
  --dataloader-num-workers 0
```

先由服务器分配合适 GPU 后运行两步模型 smoke；本轮只验证数据链路，不承诺
模型 loss、微调质量或闭环开门成功。十条来自同一种子的窄位置 pilot 用于打通
训练，不作为正式泛化能力结论；同源近重复样本不能随机拆分后声称独立验证。

官方依据：[数据格式](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/getting_started/data_preparation.md)、
[模态与动作配置](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/getting_started/data_config.md)、
[自定义机器人微调](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/getting_started/finetune_new_embodiment.md)。
