# 开门项目：按功能进入

数据增广直接使用固定在 `third_party/AutoData` 的官方生成器（`scripts/generate_dataset.py`）。
本仓库只提供 AutoData 要求的扩展件：门环境、任务/机器人/环境描述 YAML、规划器配置、
种子转换和物理验收。依赖固定方式见[依赖与开发结构](../../dependencies/README.md)。

| 功能 | 命令目录 | 实现 / 配置 |
|---|---|---|
| 环境安装与恢复 | `setup/` | `arx_door/runtime.py`、`configs/runtime/` |
| 场景布置 | `scene/` | `arx_door/scene/`、`configs/scene/`，门源资产 `assets/door/` |
| 专家示范（种子）采集与审核 | `demos/` | `arx_door/demos/`，原生 cuMotion 规划 `planner/` |
| **AutoData 增广** | **[autodata/](autodata/README.md)** | 门环境 `arx_door/envs/`，扩展件 `arx_door/autodata/`，描述 `configs/autodata/` |
| LeRobot 导出与数据契约 | `dataset/` | `configs/dataset/`、`arx_door/dataset/` |
| Gemini / VLM 视觉审核（与物理验收并列，不覆盖） | `vlm/` | `arx_door/vlm/` |
| 微调模型推理、闭环与随机场景评估 | [policy/](policy/README.md) | `arx_door/policy/`、`configs/policy/`，与增广共用 `arx_door/envs/door.py` |
| 运动学/规划诊断 | `diagnostics/` | 仅用于定位问题 |

ROS bringup 和早期 VLA 示教场景入口分别在 `scripts/ros/` 与 `scripts/vla/`。

资产全部在 `assets/`：`arx_r5a/`（机械臂 URDF/XRDF，cuMotion 与 cuRobo 共用）、
`b2arx_robot/`（B2 搭载机械臂整机）、`door/`（门源资产）。`assets/` 只存本地、不进 Git。
环境、数据、模型和运行产物在 `generated/`，同样不进 Git。
