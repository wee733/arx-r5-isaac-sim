# ARX 门前圆弧随机场景

后续任务与采数约束见 [圆把手开门数据生成契约](door_data_generation.md)：
圆把手逆时针旋转 45°、先拉门缝、松开换抓侧沿再拉开；只使用腕部相机
作为策略视觉输入，目标为 GR00T N1.7，采用程序化专家生成而不使用 RL。

本场景使用仓库内的 ARX 模型和 `assets/door` 门资产。文件已实际复制到
`/home/workspace/arx-r5-isaac-sim`，不依赖旧项目目录或软链接。

## 几何定义

参数在 `arx_r5_isaac_sim_bringup/config/door_scene.yaml`：

| 参数 | 值 | 定义 |
| --- | --- | --- |
| 水平半径 | 0.59 m | 基座原点 XY 到门把手参考点 XY 的距离 |
| 基座高度 | 0.63 m | `base_link` 安装底面相对地面的高度 |
| 采样角 | −30° 至 +30° | 以指向门外的法线为中心，角度均匀采样 |
| 默认朝向 | `face_handle` | 基座 +X 在水平面内始终指向把手 |
| 基座俯仰/横滚 | 0° | 暂按水平安装，未沿用桌面场景的倾斜角 |

世界坐标为 Z 向上，现有门正面朝 −Y。闭门时前侧把手抓取参考点为
`H=(0.313, −0.3752, 0.928)` m，对应
`/World/Door/door_handle/grasp_target`。把手转轴位于门内侧的 Y=−0.3082 m；
请确认实测 59 cm 是否从前侧抓握位置量取，若从转轴量取，应修改
`handle_reference_prim` 为 `/World/Door/door_handle`。

默认法线下，基座位置为：

```text
x = Hx + 0.59 sin(theta)
y = Hy - 0.59 cos(theta)
z = 0.63
theta ∈ [-30°, +30°]
```

端点相对把手横向偏移 ±0.295 m，法线方向间距约 0.510955 m。
59 cm 是水平圆弧半径，因此既不是三维直线距离，也不是所有位置都保持
59 cm 的门面法向距离。

**待实测确认**：把手高度沿用资产的 92.8 cm；63 cm 暂按安装底面高度；
默认基座随位置转向把手。支柱仅表示安装支撑，不代表尚未测量的移动底盘。
本场景每次初始化改变基座位置，单次操作中底座保持固定。

## 生成与预览

在安装 Isaac Sim 5.1 的 Python 环境中，从仓库根目录运行：

```bash
# 正中位置。ISAAC_SIM_PYTHON 指向安装了 isaacsim 的 Python。
ISAAC_SIM_PYTHON=/home/lbz/miniforge3/envs/isaaclab/bin/python \
  ./scripts/run_author_door_scene.sh --angle-deg 0

# 按 seed 和 episode_index 复现任意 episode 的随机位置。
ISAAC_SIM_PYTHON=/home/lbz/miniforge3/envs/isaaclab/bin/python \
  ./scripts/run_author_door_scene.sh --seed 42 --episode-index 7 \
  --output generated/door/episode_000007.usd

# 朝向固定垂直门面（只随机位置）的变体。
ISAAC_SIM_PYTHON=/home/lbz/miniforge3/envs/isaaclab/bin/python \
  ./scripts/run_author_door_scene.sh --seed 42 --yaw-mode fixed

# 独立预览；使用干净的 Isaac Sim 终端，不要 source ROS Jazzy。
/home/lbz/miniforge3/envs/isaaclab/bin/python scripts/preview_door_scene.py

# 有限帧物理检查和截图。
/home/lbz/miniforge3/envs/isaaclab/bin/python scripts/preview_door_scene.py \
  --headless --frames 120 --screenshot generated/door/preview.png
```

默认输出 `generated/door/scene.usd` 和同名 JSON。每个 JSON 记录随机种子、
episode 索引、实际角度、把手世界坐标和 `world -> base_link` 位姿；
四元数顺序遵循 ROS 的 `(x,y,z,w)`。生成目录被 Git 忽略。
USD 中 `/World/PlacementArc` 是无碰撞的 guide，可在查看器中切换其显示。

预览入口保留相机远程 payload 的未加载状态，因此几何预览不依赖下载相机资产。
门的铰链、把手、锁舌及碰撞体保留原资产设置；动力学参数仍是原资产未标定值。

## 采数接口边界

纯 Python 函数 `door_workcell.sample_base_pose()` 可由未来的 episode reset
调用，采样不依赖之前执行过多少个 episode。当前交付包括场景生成、随机位姿、
复现信息和独立预览。现有桌面抓放 episode driver 尚未接入门任务；
开门轨迹、把手操作、成功判据、动态碰撞场景更新及 VLA 采数需要按门任务另行接入。
不要直接用桌面场景的固定 TF 或 cuMotion 碰撞配置运行此场景。

## 本次验证

几何、朝向、种子复现及无效参数检查：11 项通过。
Isaac Sim 5.1 已完成中点和 +30° 端点各 120 帧预览。
端点基座在运行前后保持不变，与期望坐标的误差为约 `2.91e-8 m`；
结果记录在 `generated/door/right_endpoint.check.json`。
截图为 `generated/door/preview.png` 和 `generated/door/right_endpoint.png`。
这些检查验证场景加载和基座放置，不代表开门任务或可达性验证。
