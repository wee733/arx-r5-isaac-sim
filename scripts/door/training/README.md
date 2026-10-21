# B2ARX 全量微调与模型发布

最终模型与数据接口见 [模型说明](../../../models/groot-n17-b2arx-door/README.md)。
2026-10-21 完成的训练使用 NVIDIA Isaac-GR00T
`dac54b313160c201c450f97df0fd74ebd379c892`，不是 LoRA 或只训练 action head。

## 复现训练

1. 检出上述官方版本，按该版本的 `pyproject.toml` / `uv.lock` 安装依赖。
   已验证版本为 Python 3.12、torch 2.9.0+cu128、torchvision 0.24.0+cu128、
   torchcodec 0.8.0、transformers 4.57.3、flash-attn 2.8.3、deepspeed 0.17.6。
2. 准备 `nvidia/GR00T-N1.7-3B` 及其处理器依赖的 Hugging Face 本地缓存。
   `launch_b2arx_full.py` 将处理器 Hub 名称解析为缓存路径，避免离线 tokenizer
   仍发起元数据请求；缓存缺失时会直接报错。
3. 准备通过验收的 LeRobot v2.1 train 分区。56 条训练轨迹与 16 条留出轨迹
   按专家源分组，G04/G05 及其变体只能进入 holdout。
4. 在调度器已分配的两张 GPU 上运行：

```bash
export GR00T_REPO=/path/to/Isaac-GR00T
export GR00T_TRAIN_PYTHON=/path/to/training-venv/bin/python
export B2ARX_DATASET=/path/to/lerobot_v21/train
export B2ARX_OUTPUT=/path/to/new/checkpoints
bash scripts/door/training/train_b2arx_full.sh
```

若视频解码需要独立 FFmpeg 运行库，在调用前设置对应 `PATH` 与
`LD_LIBRARY_PATH`；本仓库不写入特定服务器的环境路径或登录配置。

该入口沿用正式训练参数：2 GPU、每卡 batch 4、累积 8、有效 batch 64、
学习率 2e-5、10,000 步、bf16、ZeRO-2、梯度检查点。官方训练器的默认
余弦调度与 5% warmup 保持不变。包装入口要求所有原始参数可训练且无 LoRA，
记录主要模块的首个反向梯度；第 10 步及其后每 200 步保存完整训练状态。
最终发布只保留推理文件，服务器保留最近三份完整 checkpoint 供续训。

## 导出推理权重

```bash
python3 scripts/door/training/export_inference_checkpoint.py \
  /path/to/checkpoint-10000 /path/to/new/inference-export
```

导出器按完整张量重新分片，每片小于 1.9 GB，读回并逐张量核对字节一致性。
不进行量化或精度转换。发布目录包含权重、索引、模型/处理器配置、归一化统计、
embodiment 映射和验证摘要；训练缓存、优化器、失败轨迹、视频及中间 checkpoint
不进入 Git。权重受 NVIDIA 模型许可证约束，必须同时保留模型目录的 LICENSE。

训练完成与权重导出检查不代表策略闭环成功。留出集模型评估和仿真闭环尚未完成。
LingBot 目前只有数据接口验证，未完成微调。
