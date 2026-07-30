# ARX-R5 → GR00T N1.7 数据采集与交付契约

本文把 NVIDIA 的 N1.7 要求固定成一条可执行链路：5090 端采集后，把数据传到服务器；服务器负责转换、深度校验、重新计算统计、官方 loader 验证和微调。

## 官方依据

- [NVIDIA：GR00T N1.7 Data Preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/getting_started/data_preparation.md)
- [NVIDIA：Modality / Action Configuration](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/getting_started/data_config.md)
- [NVIDIA：Custom Embodiment Fine-tuning](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/getting_started/finetune_new_embodiment.md)
- [NVIDIA：Real-world Deployment and Data Collection](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/getting_started/real_world_deployment.md)
- [NVIDIA：Hardware Recommendations](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/getting_started/hardware_recommendation.md)
- [NVIDIA Physical AI：Simulation Data Export](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-data-export.html)
- [Isaac Lab 2.3.2：Teleoperation and Mimic](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/imitation-learning/teleop_imitation.html)

固定版本是 NVIDIA Isaac-GR00T commit `9c7e746b2cd37a810070a98ef41d290a07e806c2`，不要混用 N1.7 EA 的 Python 3.10/Torch 2.7 环境。

## 一、5090 端必须记录什么

统一使用 30 Hz 数据时钟。每一行数据表示：

```text
当前观测 O_t：front RGB、wrist RGB、实测 q_state[t]
                      │
示教输入 → IK / action manager / controller target 处理
                      │
最终绝对命令目标 q_cmd[t]
                      │
同时记录 (O_t, q_cmd[t])，然后才推进 physics 到 t+1
```

绝不能把 post-step 的 `q_state[t+1]` 和 pre-step 的命令 `q_cmd[t]` 放在同一行。若遥操作输入是 EEF pose、VR delta 或归一化 action，必须在 IK/控制器之后截取最终的 6+1 绝对关节目标。

### 每帧字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `front` | `uint8[H,W,3]` RGB | 固定手外视角 |
| `wrist` | `uint8[H,W,3]` RGB | 刚性眼在手上视角 |
| `state` | `float32[7]` | 当前实测 `[joint1..joint6 rad, gripper m]` |
| `action` | `float32[7]` | 即将执行的绝对命令目标，顺序和单位同 state |
| `timestamp` | `float64` | episode 内主时钟；recorder 自动将全局 `sim_time` 减去首帧时刻 |
| `front_timestamp` | `float64` | 必填，与主时钟同一时间基准的 front 时间戳 |
| `wrist_timestamp` | `float64` | 必填，与主时钟同一时间基准的 wrist 时间戳 |
| `task` | UTF-8 文本 | 每个 episode 的自然语言任务描述 |

关节顺序固定为：

```text
joint1, joint2, joint3, joint4, joint5, joint6, gripper
```

`joint8` 是 mimic joint，不进入模型。夹爪统一记录主动 `joint7` 的位置，单位米，当前契约为 `0=闭合、0.044=打开`；采集前必须确认仿真极性。

服务器当前可见的 ARX R5 URDF 中 `joint2` 下限是精确 `0.0 rad`、上限是 `3.0 rad`；转换器只额外容忍 `0.001 rad` 的浮点误差，不能把负角度工作区当成有效数据。

手臂 `RELATIVE` 和夹爪 `ABSOLUTE` 只是训练 processor 的表示方式。NVIDIA 明确要求数据里的 state/action 均保存绝对值。16 步 action chunk 对应：

```text
q_cmd[t], q_cmd[t+1], ... q_cmd[t+15]
```

手臂相对量由 processor 统一计算为每个 `q_cmd[t+k] - q_state[t]`，不是相邻 command 之差。

## 二、相机、同步和任务质量

- front 和 wrist 固定 30 FPS、相同 episode 长度、每路恰好 `L` 帧。
- joint state 原始采样率应高于相机频率，再重采样到统一 30 Hz。
- 建议 640×480 RGB；交付编码固定为 MP4/H.264/yuv420p。分辨率不是 N1.7 硬编码要求，但整个数据集必须固定。
- 两路 camera timestamp 相对主时钟、以及 camera pair 之间的最大偏差都必须不超过半帧（30 Hz 时约 16.67 ms）。
- LeRobot loader 按帧号读取视频，不会依据 parquet timestamp 自动重新同步。
- 固定手外相机外参和腕部相机安装；第一阶段先保持光照、桌面和物体类型稳定。
- 每条 episode 必须完整成功，删除失败、模糊、停顿过长、动作跳变、回退或冗余轨迹。
- 去掉开头/结尾的长时间 idle，避免长期停在关节限位附近。

除了机械臂资产和两台相机，Isaac Sim 场景还需要：任务物体、reset、success condition、teleoperation、IK/控制器后的 action tap，以及只保存成功 episode 的 recorder。

## 三、建议数据量

NVIDIA 当前通用建议：

- 5–10 条：只验证录制、回放、转换和 loader。
- 约 30 条：极窄任务的 pilot 微调可能够用。
- 至少 100 条有效成功 episode：正式单任务微调下限。
- 200+ 条：通常更稳定。
- 需要位置、光照和初始姿态泛化时，可继续扩展到约 400 条。

先用 5–10 条完成端到端闭环，再批量采集，避免在 action 定义或同步错误时浪费数百条示教。

## 四、推荐交付格式：ARX raw-v1

本阶段 5090 端统一生成便于同步审计的 portable raw-v1，再由服务器转换：

```text
arx_raw/
├── dataset.json
└── episodes/
    ├── episode_000000/
    │   ├── episode.json
    │   ├── trajectory.npz
    │   ├── front.mp4
    │   └── wrist.mp4
    └── episode_000001/
        └── ...
```

模板见 `configs/arx_r5/raw_dataset.example.json`。`trajectory.npz` 必须包含：

```python
timestamp:       float64[T]
front_timestamp: float64[T]  # 必填
wrist_timestamp: float64[T]  # 必填
state:           float32[T, 7]
action:          float32[T, 7]
```

`episode.json` 至少包含：

```json
{
  "episode_index": 0,
  "task": "pick up the red cube and place it in the bowl",
  "success": true,
  "length": 240
}
```

### 可直接拷到 5090 的 recorder

`scripts/arx_raw_recorder.py` 不依赖 Isaac Sim，可复制进你的 Isaac 工程。在 action 已完成 IK/控制器处理、但 physics 尚未推进的位置调用：

```python
from arx_raw_recorder import ArxRawRecorder

recorder = ArxRawRecorder(
    "/data/arx_raw",
    fps=30,
    action_source="isaac_joint_commands",
    encoder="auto",  # 优先 libx264，否则 libopenh264；启动时立即验证
)

recorder.start_episode("pick up the red cube and place it in the bowl")

# 每个统一 30 Hz step：
recorder.append(
    front_rgb=front_rgb_uint8,        # HWC RGB，不是 BGR/RGBA
    wrist_rgb=wrist_rgb_uint8,
    state=q_measured_7d,              # apply action 前的当前反馈
    action=q_absolute_target_7d,       # IK/action manager 后的最终绝对目标
    timestamp=sim_time,                # 可传全局时钟，recorder 自动按 episode 归零
    front_timestamp=front_capture_time,
    wrist_timestamp=wrist_capture_time,
)

recorder.end_episode(success=task_succeeded)
```

Isaac Lab 2.3.2 的默认顺序是 `process_action → record_pre_step → apply_action → write_data_to_sim`。PinkIK 一类 action term 到 `apply_actions()` 才计算最终 joint target，因此默认 pre-step recorder 读不到当前目标，而 post-step recorder 又会把 `O_{t+1}` 与 `q_cmd[t]` 错配。

ARX 集成必须让 action term 在 `process_actions()` 阶段完成计算并缓存最终绝对 6+1 target；自定义 pre-step hook 同时读取 `O_t` 与这份缓存，随后 `apply_actions()` 只发送同一缓存。具体转换为：

- `JointPositionAction`：缓存 processed absolute joint target。
- `RelativeJointPositionAction`：缓存 `current_joint_pos + processed_action`。
- PinkIK / EEF delta：在 process 阶段完成 IK，缓存求得的绝对 6D arm target。
- binary gripper：先映射为主动 `joint7` 的绝对米制位置，再拼成第 7 维。

默认 recorder 中的归一化 action、EEF delta、relative processed action 或 post-step state 都不能直接作为当前数据的 `action`。

## 五、传到服务器

推荐可断点续传的 `rsync`：

```bash
rsync -aH --info=progress2 --partial --append-verify \
  /path/to/arx_raw/ \
  buaalibozhao@SERVER:/data2/buaalibozhao/gr00t-n1.7/incoming/arx_raw/
```

也可以先生成清单：

```bash
cd /path/to/arx_raw
find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
```

服务器收到 raw-v1 后：

```bash
cd /home/buaalibozhao/Arx_GROOT

make convert-arx \
  RAW=/data2/buaalibozhao/gr00t-n1.7/incoming/arx_raw \
  DATASET=/data2/buaalibozhao/gr00t-n1.7/datasets/arx_task_v1

make prepare-arx \
  DATASET=/data2/buaalibozhao/gr00t-n1.7/datasets/arx_task_v1
```

`prepare-arx` 会依次执行：

1. metadata、全部 parquet 数值、timestamp、索引和 URDF 限位检查。
2. 每个 front/wrist MP4 的 FPS、分辨率、帧数检查，并通过 GR00T/torchcodec 抽帧解码。
3. 检查 action 是否几乎等于 feedback state，拦截错误的数据录制方式。
4. 根据 parquet 内容哈希强制刷新 `stats.json` 和 `relative_stats.json`，避免官方仅按 schema fingerprint 误复用旧统计。
5. 使用固定 N1.7 的官方 `LeRobotEpisodeLoader` 和 relative-action loader 做真实加载。

当前固定验收链路只接受上述 ARX raw-v1，并由服务器统一转换为 LeRobot v2.1；不要在 5090 端自行生成缺少 `arx_capture.json` 同步审计字段的 LeRobot 数据。

## 六、微调与验收

先找一张明确分配、至少 40 GB 空闲显存的 GPU，执行两步真实模型 smoke：

```bash
CUDA_VISIBLE_DEVICES=4 NUM_GPUS=1 \
make finetune-arx-smoke \
  DATASET=/data2/buaalibozhao/gr00t-n1.7/datasets/arx_task_v1 \
  OUTPUT_DIR=/data2/buaalibozhao/gr00t-n1.7/checkpoints/arx_task_v1-smoke
```

确认 loss 有限且保存 `checkpoint-2` 后，再正式训练：

```bash
CUDA_VISIBLE_DEVICES=4 NUM_GPUS=1 \
MAX_STEPS=10000 GLOBAL_BATCH_SIZE=32 SAVE_STEPS=1000 \
make finetune-arx \
  DATASET=/data2/buaalibozhao/gr00t-n1.7/datasets/arx_task_v1 \
  OUTPUT_DIR=/data2/buaalibozhao/gr00t-n1.7/checkpoints/arx_task_v1
```

训练后先做开环：

```bash
make arx-open-loop \
  DATASET=/data2/buaalibozhao/gr00t-n1.7/datasets/arx_task_v1 \
  CHECKPOINT=/data2/buaalibozhao/gr00t-n1.7/checkpoints/arx_task_v1/checkpoint-10000
```

训练轨迹上的 prediction/GT 曲线应随 checkpoint 推进明显靠拢，MSE/MAE 应下降。开环通过后才进入 Isaac Sim 小范围闭环。
