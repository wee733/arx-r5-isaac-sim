# 单腕部门任务 LeRobot 导出

`scripts/export_door_lerobot.py` 接收一个或多个 `--attempt` 目录。每个目录必须有
`episode.json`、`trajectory.npz` 和已编码的 `wrist.mp4`。仅导出 `success: true`
且 `review_required: false` 的数据；脚本不改变源标签。未审核的尝试、失败轨迹不能充当专家示范。

```bash
generated/runtime/data_export/bin/python scripts/export_door_lerobot.py \
  --attempt generated/door_teaching/run01/approved_attempt \
  --output generated/door_dataset_v2
```

输出为 LeRobot v2.1 单腕部 RGB 视频、7D 绝对状态/绝对控制目标：前六维是机械臂关节弧度，
最后一维是 joint7 主动夹爪位移（米）。保留原始动作，不用反馈位置伪造动作。
`meta/modality.json` 把手臂和夹爪拆分为 `single_arm` 与 `gripper`。门状态和阶段只放在
`diagnostics/`，不进入模型输入。帧数、FPS、时间戳与视频逐一核对。

格式依据 [NVIDIA 官方数据准备文档](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)
与官方 v2.1 示例。**尚未运行官方 N1.7 loader。** 当前 `stats.json` 与
`episodes_stats.jsonl` 只有绝对状态/动作基础统计；训练前应使用固定版本 N1.7 工具重新生成
正式统计（包括所需相对动作统计），配置单腕部数据模态，并实际完成 loader 验收。
本导出器不能替代既有服务器端同步审计与完整训练验收。

目前可能没有通过审核的成功 episode。结构自检的合成输入和输出必须标明 `TEST_ONLY`，
使用 `--test-only` 且输出在 `generated/runtime/` 下；它们不是用户专家样本，不能用于训练。
