---
license: other
license_name: nvidia-non-commercial
license_link: LICENSE
base_model: nvidia/GR00T-N1.7-3B
tags:
  - robotics
  - groot
  - b2arx
  - full-finetuning
---

# GR00T N1.7 · B2ARX 开门全量微调

基于 `nvidia/GR00T-N1.7-3B` 的研究用衍生模型，使用无人院共享 3DGS + mesh
场景中的 B2ARX 开门演示进行全量微调。2026-10-21 04:10（Asia/Shanghai）完成
10,000 步，最终 checkpoint 为 `checkpoint-10000`。这不是 NVIDIA 官方发布的
任务模型；尚未完成留出集模型评估、仿真闭环或实机验证。

## 下载与加载

权重通过 Git LFS 存储。只下载本模型：

```bash
git lfs install
git lfs pull --include='models/groot-n17-b2arx-door/*.safetensors'
```

四个标准 safetensors 分片共约 6.91 GB，每片小于 1.9 GB。它们由完整 checkpoint
重新分片得到，没有量化或改变张量精度；全部 1,031 个张量经读回逐一核对，字节一致。
模型目录可作为官方 GR00T policy 的 `model_path`，embodiment 为 `NEW_EMBODIMENT`。
先安装与训练一致的 [Isaac-GR00T 固定版本](https://github.com/NVIDIA/Isaac-GR00T/tree/dac54b313160c201c450f97df0fd74ebd379c892)，
并准备该版本需要的 Cosmos-Reason2-2B tokenizer/processor 缓存或可用 Hub 连接。
该目录不是独立于 GR00T 运行环境的应用。

保存的 `config.json`、`processor_config.json`、`statistics.json` 与
`embodiment_id.json` 来自实际训练，不应替换为基础模型的对应配置。
本次检查验证文件与张量完整性，尚未执行发布包的端到端推理。

## 训练与数据

| 项目 | 实际配置 |
|---|---|
| 官方代码 | `dac54b313160c201c450f97df0fd74ebd379c892` |
| 全量可训练参数 | 3,144,016,000 / 3,144,016,000；无 LoRA |
| 数据 | 10 条专家种子 + 62 条增广轨迹，合计 72 条、87,075 帧 |
| 训练 / 留出 | 56 / 16 条，68,191 / 18,884 帧 |
| 分组 | G04/G05 及其全部变体只在 holdout，其余专家族只在 train |
| 环境 | 一份完整高斯场景，24 env 按 8 × 3 排列，共三轮亮度变化 |
| 随机化 | 初始位姿、原建模门材质的中性增益/粗糙度/UV、足端摩擦与三档灯光/曝光 |
| GPU / batch | 2 × RTX PRO 6000 Blackwell 96 GB；每卡 4，累积 8，有效 64 |
| 优化 | LR 2e-5、余弦调度、5% warmup；bf16、ZeRO-2、梯度检查点 |
| 步数 / 耗时 | 10,000 / 15 小时 28 分钟 |
| 最后三次训练 loss | 0.0211、0.0230、0.0223 |

72 次增广尝试中 62 条通过生成验收与独立回放；10 条失败结果不进入训练。
留出划分隔离专家来源，但仍是同一场馆，不能称作跨场馆泛化测试。
高斯保留采集时的辐射外观，动态灯光主要影响插入几何，曝光影响整幅图像，
不能将本批数据解释为对高斯进行了物理重光照。

`training_summary.json` 保存最终训练与参数验证摘要。
[复现训练入口](../../scripts/door/training/README.md) 使用同样的正式训练参数。

## 观测与动作约定

- 双路 RGB：`wrist`、`ego`，640 × 480、30 Hz。
- 状态 13D：EEF XYZ（m）、旋转向量（rad）、单侧手指开度（m）、6 轴臂关节角（rad）。
- 数据动作 7D：EEF XYZ + 旋转向量 + 单侧手指开度，手指范围 0–0.044 m。
  这是实际执行的 EEF 目标，不是关节角，也不是下一帧状态。
- 坐标系为当前 R5a 安装座；官方 processor 将 EEF 转为相对 SE(3) 动作，夹爪保持绝对。
  训练模态选取 16 个时间索引；网络配置中的 `action_horizon=40` 原样保留，
  应通过保存的 processor 处理时序与掩码，不自行覆盖。
- 30 Hz EEF 控制通过 120 Hz IK 执行；腿部使用独立站立 PD，不输出学习到的行走动作。

完整模态定义见 [b2arx_eef_n17.py](../../configs/dataset/b2arx_eef_n17.py)。
训练 loss 降低不等于开门成功率；应先完成离线和仿真闭环评估。

## 发布范围与许可

本目录仅发布最终推理权重和必要配置。配套机器人 URDF/USD、网格和门资产见
[资产入口](../../assets/README.md)。数据集、原始场馆扫描、失败轨迹、
视频、中间 checkpoint、优化器、训练日志、服务器环境与凭据不包含在 Git 中。
可续训的完整 checkpoint 仍保留在原训练服务器。

模型及其衍生权重遵循随附 [NVIDIA LICENSE](LICENSE)，限非商业研究或评估用途；
它们不适用仓库集成代码的 Apache-2.0 许可证。上游模型与许可来源：
[nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B)。
