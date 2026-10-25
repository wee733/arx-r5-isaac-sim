# GR00T 数据生成配套资产

本目录保存开门演示采集、AutoData 增广和回放使用的机器人与门源资产。
GR00T 微调读取导出的图像、状态和动作数据；URDF/USD 用于生成这些数据的仿真与规划环境。

| 资产 | 入口 | 配套内容 |
|---|---|---|
| B2 + ARX 整机 | [my_robot.urdf](b2arx_robot/my_robot.urdf)、[my_robot.usd](b2arx_robot/my_robot/my_robot.usd) | `meshes/`、`my_robot/configuration/` 四个 USD 子层 |
| ARX R5A 机械臂 | [r5a_cumotion.urdf](arx_r5a/urdf/r5a_cumotion.urdf)、[r5a.xrdf](arx_r5a/xrdf/r5a.xrdf) | 原始 URDF、STL 网格、SRDF、ROS 描述包及配置 |
| 开门对象 | [door.usda](door/door.usda) | `door_visuals.usdc` 和 `textures/`，含白玉色门面贴图 |

当前发布的 [B2ARX GR00T 模型](../models/groot-n17-b2arx-door/README.md) 使用整机演示。
机械臂描述包同时供单臂流程和整机规划器的 XRDF 配置使用。

## 下载

USD、STL、DAE 网格及门贴图通过 Git LFS 保存。已有仓库中运行：

```bash
git lfs install
git lfs pull --include='assets/arx_r5a/**,assets/b2arx_robot/**,assets/door/**'
```

保持目录结构不变：整机 URDF 使用相对 `meshes/` 路径，USD 需要配套子层；
机械臂 URDF 使用 ROS 包名 `isaac_ros_manipulation_arx_r5a_robot_description`，
其包根为 `assets/arx_r5a/`，由 ROS 包解析器或项目规划适配器解析网格路径。

## 源资产与运行配置

`b2arx_robot/my_robot/my_robot.usd` 是已有的浮动基座转换结果。
[prepare_b2arx.py](../scripts/door/autodata/prepare_b2arx.py) 在源资产基础上生成独立运行包，
移除工作台支撑，并可调整手指碰撞分解；控制增益、自碰撞和传感器等还由环境配置设置。
这些源文件不能单独替代完整训练场景或复现已验收的回放。

发布前已检查三份 URDF 的网格路径、整机五个 USD 文件、门的两个 USD 文件及六张贴图。
用 Blender 自带 OpenUSD 检查时，整机材质依赖 Isaac Sim 提供的 `OmniPBR.mdl`，
且源模型已有 `R5a_ee/visuals` 指向缺失 `/visuals/R5a_ee` prim 的警告；本次保留原始 USD，
没有修补该引用或重新进行物理回放验证。

无人院原始扫描 `wurenyuan/`、生成的运行包及轨迹 `generated/`、转换缓存和含本机路径的
转换配置不随本次资产发布。复现完整场馆数据生成仍需这些本地输入。

## 来源

资产来自本项目现有的本地机器人描述包、整机转换包和门模型，本次未重新转换几何或修改动力学。
ARX 描述包的上游信息与许可声明保留在 [package.xml](arx_r5a/package.xml)，
门模型的来源记录见 [door/README.md](door/README.md)。
第三方网格和贴图的原有权利归属保持不变；仓库代码的 Apache-2.0 许可不自动适用于这些素材。
