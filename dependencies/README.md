# 依赖与 AutoData 开发结构

后续开门数据生成以 AutoData 为主线。官方框架以 Git submodule 固定在本仓库内；
ARX 的任务、控制、物理验收和数据导出留在本项目。旧 SkillGen 实现和数据保留作迁移对照，
这次目录接入不代表数据 v2 已移植完毕，也不代表 AutoData 已通过仿真验收。

## 目录与职责

```text
arx-r5-isaac-sim/
├── .gitmodules                        # 官方仓库地址；Git gitlink 保存固定提交
├── dependencies/
│   ├── autodata.lock.json             # 启动/检查共用的版本与镜像清单
│   ├── python/                        # 独立数据/历史仿真环境的依赖
│   └── patches/                       # 现有 ROS 补丁
├── third_party/AutoData/              # 官方框架；不直接改动、不要提交生成产物
│   └── submodules/IsaacLab-Arena/     # 沿用官方锁定的 Arena/Lab，不另外装一份
├── arx_door/
│   ├── autodata/                      # 机器人接口、Arena 场景接入、记录和物理验收
│   ├── scene/                         # 场景/相机共用代码
│   ├── demos/                         # 示范采集与审核
│   ├── skillgen/                      # 已有实现与共用组件；迁移完成前保留
│   ├── dataset/                       # 训练数据转换
│   └── policy/                        # VLA 推理与闭环评估
├── configs/autodata/                  # ARX 任务描述；新生成任务从这里扩展
├── scripts/door/autodata/             # 准备、环境检查、容器和生成命令
└── generated/door_autodata/           # 资产包、官方例子、生成轨迹、报告；不进 Git
```

流程为：示范 → AutoData 增广 → ARX 物理验收 → LeRobot 导出 → VLA 全量微调 → 闭环评估。
视觉审核与物理验收分别保留证据。先跑官方 Franka，再验证一条 ARX 开门轨迹，随后批量生产。
这一步不开展消融、论文或额外场景随机化。

## 固定版本与恢复

当前基线：AutoData `953cd37778ab`，Arena `c7edf3a9d5d8`，Lab `91a0906f9380`，
Isaac Sim Docker 6.1.0。三个源码版本均来自实际 Git 对象；上游 README 的 Arena/Lab
版本说明与 gitlink 不一致，暂不按 README 换版本。完整值在 `autodata.lock.json`。
这是源码版本固定；Docker 基础镜像目前按版本 tag 引用，安装过程仍需联网，不宣称离线或逐字节重现。

在新机器正常克隆本项目后，从项目根初始化依赖：

```bash
dependencies/bootstrap.sh            # 检出锁定的 AutoData/Arena/Lab
dependencies/bootstrap.sh --with-lfs # 另外下载官方示例数据集
dependencies/bootstrap.sh --check    # 只校验版本，不改动磁盘
python3 scripts/door/autodata/preflight.py --source-only
```

`bootstrap.sh` 只检出本仓库 gitlink 记录的提交，从不跟随上游 `main`；
它用命令级 HTTPS 改写处理嵌套子模块的 SSH URL，不改全局 Git 配置。
`bootstrap.sh --with-lfs` 下载上述官方示例；ARX 自己的种子和资产仍按项目备份恢复。
`--source-only` 只验证源码；不带该选项还检查项目镜像及其版本标签等启动条件。
该检查不是模拟器或物理验收。

### 版本校验的两个层面

`dependencies/verify_pins.py` 不检出、不联网，只比较两处记录：

| 来源 | 作用 | 何时校验 |
|---|---|---|
| 父仓库 gitlink（`git ls-files --stage` / `git ls-tree`） | `git clone` 和 `git submodule update` 实际检出的提交 | CI 每次推送；`bootstrap.sh` 初始化前 |
| `dependencies/autodata.lock.json` | 容器启动、Docker 镜像标签和文档使用的版本 | 容器启动、`bootstrap.sh` 初始化后 |

两者不一致时 CI 失败，`bootstrap.sh` 拒绝初始化。嵌套的 Arena/Lab 版本记录在
**拥有它们的那个仓库**（Arena 的 gitlink 在 AutoData 的树里），所以未检出
AutoData 时这两项只报告为跳过，不判为漂移。要升级版本，在同一个提交里同时
更新 gitlink 和 lock，再跑 `dependencies/verify_pins.py` 确认。

## 容器入口

先查看官方示例容器的构建和启动命令：

```bash
bash scripts/door/autodata/run_container.sh --official
```

加 `--start` 才实际构建和启动；首次缺源码可加 `--bootstrap-source`，它初始化本仓库
声明的子模块，不从 `main` 拉取最新版本。已有源码版本不符时拒绝自动 reset。
容器启动仍按项目约定由用户手动执行。

使用的是固定源码里的原版 Dockerfile；项目包装器负责 Git 子模块所需的父仓库挂载、
输出目录和可选 ARX 资产挂载。它不是原封不动的上游 launcher，不能将其运行结果称为
上游 launcher 已验证。镜像使用 `arx-autodata:*`，与上游 `autodata:*` 分开。

`/workspaces/autodata` 是镜像构建时安装的官方代码；`/workspaces/arx` 是只读的本项目，
包含子模块及父仓库 `.git/modules`；`/outputs` 是可写产物目录。
本地源码改动不会自动覆盖镜像中的官方框架代码，升级后必须重建镜像。
不要将单独的子模块目录挂载为源码参考后在其中执行 Git：它的 `.git` 文件需要父仓库元数据。
目前容器入口为 headless 用途，没有配置 Kit GUI/X11 转发。

ARX 模式与具体生成命令见 [接入手册](../scripts/door/autodata/README.md)。

## 后续升级与开源贡献

- 日常 ARX 开发只改本项目的适配、配置和入口；不要把业务代码塞进官方目录。
- 不使用 `git submodule update --remote` 自动追随上游。升级时选择具体上游提交，核对其
  嵌套版本，同步 gitlink 和 `autodata.lock.json`，重建镜像并通过官方示例、ARX 单条物理验收。
- 需要给上游修 bug 时，在独立开发 checkout 中创建分支。GitHub fork 是你的远程副本，
  不是另一套依赖版本来源；开发 checkout 的 `origin` 指向你的 fork，`upstream` 指向 NVIDIA。
- PR 提交最小通用修复和复现，不带 ARX 资产、轨迹、模型或凭证。合入后，本项目再固定到含修复的上游提交。
  若必须提前使用未合入的修复，显式记录临时 fork URL 和提交，并经过相同验证。

本次仅建立本地依赖结构，没有创建远程 fork、提交或推送。Git submodule 注册会将 `.gitmodules`
和 gitlink 加入暂存区；它们与本项目代码应在之后审查时一起提交。已有其他暂存内容保持原样。
