# 开门项目：按功能进入

后续数据生成统一沿 AutoData 主线推进，固定依赖与代码归属见
[依赖与开发结构](../../dependencies/README.md)。官方代码在 `third_party/AutoData`，
ARX 适配在 `arx_door/autodata`。旧 SkillGen 数据 v2 保留作迁移对照；当前尚未完成 AutoData 物理验收。

| 功能 | 命令目录 | 可复用实现 / 配置 |
|---|---|---|
| 环境安装与恢复 | `setup/` | `arx_door/runtime.py`、`configs/runtime/` |
| 场景与资产 | `scene/` | `arx_door/scene/`、`configs/scene/` |
| 专家示范采集与审核 | `demos/` | `arx_door/demos/` |
| 已有 SkillGen 数据增广 | `skillgen/` | `arx_door/skillgen/` |
| **AutoData / Arena 数据生成主线（迁移验证中）** | **[autodata/](autodata/README.md)** | `arx_door/autodata/`、`configs/autodata/`、`third_party/AutoData/` |
| LeRobot 导出与数据契约 | `dataset/` | `configs/dataset/`、`arx_door/dataset/` |
| **数据 v2**：直线规划、恢复示范、机身视角、阶段语言 | `skillgen/run_batch.sh --generation-profile`、`dataset/` | `configs/skillgen/data_v2.json`、方案见 `docs/vla/vla_self_improvement_and_locomanipulation_plan.md` |
| Gemini / VLM 视觉审核（与物理验收并列，不覆盖） | `vlm/review_door_episodes.py` | `arx_door/vlm/` |
| **微调模型推理、闭环与随机场景评估** | **[policy/](policy/README.md)** | `arx_door/policy/`、`configs/policy/` |
| 原生 cuMotion 规划服务 | `planner/` | 独立规划环境 |
| 运动学/规划诊断 | `diagnostics/` | 仅用于定位问题 |
| 早期弃用的试验 | `legacy/` | 不作为新评估入口 |

ROS bringup 和早期 VLA 示教场景入口分别在 `scripts/ros/` 与 `scripts/vla/`。
策略评估统一使用 `scripts/door/policy/`，避免和数据生成、遥操作采集混在一起。

大型环境、数据、模型和运行产物保存在 `generated/`，不放进源码目录。
新策略结果统一位于 `generated/door_policy/`；历史数据和训练结果继续保留原位，
通过该目录的引用和手册访问。`docs/` 和 `tests/` 遵循本项目现有本地保存约定。
