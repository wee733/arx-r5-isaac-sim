# ARX 开门的 AutoData／Arena 接入

后续数据生成采用 AutoData 主线。[固定依赖与开发结构](../../../dependencies/README.md)
定义源码、ARX 适配和产物的归属。当前交付为固定子模块、接入代码、K001 可重定位资产包和离线契约检查。
尚未完成官方 Franka 运行、ARX 单环境物理验收、四环境隔离或吞吐测量。旧数据生成入口继续保留作对照。

Arena 使用 `mimic=False`，任务子阶段由 `configs/autodata/knob_hold_open.yaml` 唯一定义。入口直接调用官方 `DataGenerator`、SkillGen 扩展和 `env_loop`，不会调用旧 Mimic 生成循环。首阶段只支持 `knob_hold_open`；`edge_regrasp`、16 环境及 LeRobot 新 schema 导出待单环境物理节点通过后继续。

## 先跑官方 Franka

源码已固定在 `third_party/AutoData`；不是 `generated/runtime` 中的临时下载目录。
首次恢复和 Git LFS 命令见依赖手册。先检查源码，再由用户手动启动项目包装的容器：

```bash
python3 scripts/door/autodata/preflight.py --source-only
bash scripts/door/autodata/run_container.sh --official --start
```

该模式不需要 ARX 模型和 native planner。它使用原版官方 Dockerfile，项目包装器补充父仓库和输出挂载；
与直接使用上游 launcher 的结果区分记录。去掉 `--start` 只打印构建/启动命令。
在容器内先生成一条官方 MimicGen 数据：

```bash
cd /workspaces/autodata
mkdir -p /outputs/official_franka/run001
/isaac-sim/python.sh scripts/generate_dataset.py \
  --env_name IsaacContrib-Stack-Cube-Franka-IK-Rel --alg mimicgen \
  --task_descriptor autodata_examples/tasks/franka_cube_stack.yaml \
  --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
  --input_file /workspaces/arx/third_party/AutoData/datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
  --output_file /outputs/official_franka/run001/mimicgen.hdf5 \
  --result_file /outputs/official_franka/run001/mimicgen.json \
  --generation_num_trials 1 --num_envs 1 --viz none
HDF5_USE_FILE_LOCKING=FALSE /isaac-sim/python.sh scripts/validate_dataset.py \
  /outputs/official_franka/run001/mimicgen.hdf5
```

通过后在新输出文件中运行 SkillGen：将算法改为 `skillgen`，task 改为
`autodata_examples/tasks/franka_cube_stack_skillgen.yaml`，embodiment 改为
`autodata_examples/embodiments/franka_ik_rel_skillgen.yaml`，输入改为
`dataset_franka_skillgen_annotated.hdf5`。先维持一环境；记录失败尝试和最终成功数。
结构校验不等于物理回放，后续还需检查动作回放和堆叠成功。
这里直接调用官方 Python 入口；镜像不包含 Git 元数据，不能在 `/workspaces/autodata`
运行依赖 `git submodule status` 的上游 shell helper。

## ARX 手动启动独立容器

官方例子通过后，再进入带 ARX 模型和 native planner 的模式：

```bash
bash scripts/door/autodata/run_container.sh \
  --model-dir /home/lbz/workspace/B2ARX_Train_12DOF/Isaac_Ros_CuMotion_ArxR5a/isaac_ros_manipulation_arx_r5a_robot_description \
  --start
```

运行产物统一为 `generated/door_autodata`。缺少子模块时可加 `--bootstrap-source`，
它恢复已声明的 gitlink，拒绝 reset 版本不符的既有 checkout。不带 `--start` 只预览 Docker 命令。
源码版本、基础镜像和项目镜像名称统一从 `dependencies/autodata.lock.json` 读取。

挂载关系：固定源码参考位于 `/workspaces/arx/third_party/AutoData`，父仓库 `.git/modules` 一同挂载，
保证子模块的 Git 路径可解析。执行的是镜像内 `/workspaces/autodata` 的固定代码；官方 entrypoint
可在那里创建运行时 symlink，不改宿主源码。项目只读位于 `/workspaces/arx`，输出读写位于 `/outputs`，
URDF／XRDF 模型只读位于 `/assets/arx_model`。另外保留 native planner venv 与其实际 Python 路径的只读挂载，
保证与 Isaac Sim Python 的库隔离。cuMotion 的 CUDA 13 依赖仍需在首次真实规划时验证；失败会写明原因，不能转用另一个规划器。

Wrapper 检查三个实际 source pins、源码无本地改动及 native planner 路径。`--start` 遇到 swap 耗尽时停止，须先解决内存压力；不会结束其他用户任务。镜像构建可能需要 NGC 访问，登录失败须由用户处理，不写入凭证。

## 容器内单环境生成

K001 已准备的静态资产：`/outputs/assets_k001_20261013_final/manifest.json`。源 USD 和模型不修改，资产内关节、mimic、材质关系及 602 个 physics／PhysX／drive 属性已检查。该静态结果不等于物理验收。

在容器内创建 native worker argv 配置（不会在官方 Python 环境安装包）：

```bash
mkdir -p /outputs/runtime
cat > /outputs/runtime/native_worker.json <<'JSON'
{
  "command": ["bash", "/workspaces/arx/scripts/door/planner/run_cumotion.sh"]
}
JSON
```

已从同一个 K001 源场景静态导出 `/outputs/assets_k001_20261013_final/cuboids.json`；后续更换场景须重新导出，不能拿另一扇门的碰撞世界。

```bash
/isaac-sim/python.sh /workspaces/arx/scripts/door/autodata/generate.py \
  --assets /outputs/assets_k001_20261013_final/manifest.json \
  --input /workspaces/arx/generated/door_teaching/fresh_experts_20261013_005036/seeds/candidate_N13_K001.hdf5 \
  --urdf /assets/arx_model/urdf/r5a_cumotion.urdf \
  --cuboids /outputs/assets_k001_20261013_final/cuboids.json \
  --worker-config /outputs/runtime/native_worker.json \
  --output /outputs/knob_1env_run001 --num-envs 1 --trials 1 --max-attempts 16 --headless
```

单环境节点通过后，在新的输出目录用 `--num-envs 4 --trials 4` 验证四副本，并分别检查复位、接触过滤和腕图像来源。
当前容器入口按 headless 使用；Kit 窗口需要另行配置显示转发，四副本总览录像尚未交付。
运行中只测得什么就记录什么，不由 `num_envs=4` 推断并行速度。

## 成功与失败产物

新接受 profile 为最终门角严格 `>29°`、采样轨迹 arm 接触峰值 `<=2N`、相机间隙 `>=0.01m`、双指接触各 `>=3N` 且力向量相反，并满足解锁／门缝顺序和从首次门缝至终态的持续握持。指标在 30Hz pre/post-action 样本检查，不宣称覆盖每个 120Hz 物理子步。旧生成数据的 `>=29°/40N` 配置没有改动。

规划过渡直接透传已验证的六关节目标，夹爪保持规划请求的实测状态；技能阶段才恢复种子夹爪动作。所有动作噪声为零。没有 FK／pose 一致证据、夹爪模型不匹配或出限时拒绝执行。

入口用 `generate(export_demo=False)`，最终化后才通过 recorder 公共接口设置标签和导出。日志分别记录官方 OR 成功值和领域 QC 接受结果；阶段锁存不能拼成物理成功。相机接受依据实际 pose／RGB／sensor frame 递增和动作时序协议；像素比对和人工审核状态仍独立记录，不冒称已审核。

成功／失败文件分别为 `generated.hdf5`、`generated_failed.hdf5`。每个 attempt 的 JSON 记录 `env_id/attempt_id/dataset_file/demo_key`，即使早期规划失败也有原因。只含 initial_state、没有 actions 的 attempt 不导出伪 episode；有动作的部分失败轨迹保留。导出／清理错误会导致整个运行失败。

## 当前验证与后续节点

离线检查覆盖坐标／ProxyArray／精确规划关节、阈值边界、连续接触、每环境 reset、相机协议反例、失败导出和 batch 停止握手。正式 cuRobo 基类接口的 6 项测试因容器未就绪跳过。集中运行命令记录在实施进度文档；没有运行过的单环境物理、CUDA 求解、视频、四环境与 LeRobot 导出均待验。
