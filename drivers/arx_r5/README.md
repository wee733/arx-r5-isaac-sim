# ARX R5 双 Jazzy 远程遥操作

本仓库 `master` 分支用于 **ROS 2 Jazzy ↔ ROS 2 Jazzy** 的 R5 主从臂远程遥操作。
旧版 Foxy 主臂与 Jazzy 从臂的实现保留在 `foxy-jazzy` 分支。

当前版本只保留机械臂控制链路，不包含相机、图像解码或 RViz 自动启动功能。

## 主要功能

- 主臂和从臂统一使用 ROS 2 Jazzy。
- 使用系统安装的 Cyclone DDS，不再加载 Foxy 或源码 DDS overlay。
- 使用静态 DDS Peer，关闭组播发现，避免向整个子网广播发现报文。
- `--peer-ip` 可重复指定，从而连接一个或多个 Peer。
- 使用专用轻量消息 `arx5_arm_msg/msg/RemoteControl`。
- 控制 QoS 使用 Best Effort、Keep Last，默认深度为 1，优先使用最新控制帧。
- 从臂默认不反向发布状态，跨机器链路默认为单向控制。
- 启动脚本负责环境隔离、IP/CAN 检查、重复进程检查以及退出时清理子进程。

## 默认部署

| 角色 | ROS 2 | 本机 IP | CAN 接口 | ROS 节点 | 默认 Peer |
| --- | --- | --- | --- | --- | --- |
| 主臂 | Jazzy | `192.168.17.161/24` | `can0` | `arm_master_l` | `192.168.17.160` |
| 从臂 | Jazzy | `192.168.17.160/24` | `slcan1` | `arm_slave_l` | `192.168.17.161` |

两端固定使用：

- `ROS_DOMAIN_ID=17`
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`
- Cyclone DDS 静态 Peer
- `AllowMulticast=false`

本机 IP、CAN 接口、节点名和默认 Peer 当前定义在
[`00-sh/ROS2/remote_env.sh`](00-sh/ROS2/remote_env.sh) 中。

## 数据传输链路

```mermaid
flowchart LR
    A[主臂电机与编码器] -->|SocketCAN can0| B[主臂 R5Controller]
    B -->|读取 7 个关节位置及可选夹爪力矩| C[RemoteControl]
    C -->|/arm_master_l_status<br/>Best Effort / depth 1 / 默认 200 Hz| D[Cyclone DDS 静态 Peer]
    D --> E[从臂 R5Controller]
    E -->|位置控制命令| F[InterfacesThread]
    F -->|SocketCAN slcan1| G[从臂电机]
```

默认跨机器话题：

| 项目 | 值 |
| --- | --- |
| 话题 | `/arm_master_l_status` |
| 类型 | `arx5_arm_msg/msg/RemoteControl` |
| 方向 | 主臂 → 从臂 |
| 默认频率 | `200 Hz` |
| Reliability | `Best Effort` |
| History | `Keep Last` |
| 默认深度 | `1` |

轻量消息定义：

```text
float64[7] joint_pos
float64 gripper_torque
```

从臂使用前 6 个关节位置控制机械臂。位置夹爪模式使用
`joint_pos[6]`；力矩夹爪模式使用 `gripper_torque`。完整末端位姿、关节速度、
时间戳和无用电流不会在远程控制话题中传输。

> 多 Peer 控制的是 DDS 静态发现范围。若多个已配置 Peer 上存在类型和话题均匹配的
> 订阅者，它们都可以收到控制数据。请只添加需要且可信的地址。

## 环境依赖

两台电脑均需安装 ROS 2 Jazzy，以及 Cyclone DDS RMW：

```bash
sudo apt update
sudo apt install ros-jazzy-rmw-cyclonedds-cpp
```

确认安装：

```bash
source /opt/ros/jazzy/setup.bash
ros2 pkg prefix rmw_cyclonedds_cpp
```

## 编译

主从两台电脑都必须使用相同版本的代码和消息定义，并分别编译：

```bash
cd ROS2/R5_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to arx_r5_controller
```

编译结果应生成：

```text
ROS2/R5_ws/install/setup.bash
```

由于远程话题使用仓库自定义的 `RemoteControl` 消息，新旧版本不能混用。

## CAN 准备

安装仓库中的 udev 规则：

```bash
cd ARX_CAN
sudo cp arx_can.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

启动主臂 CAN：

```bash
cd ARX_CAN/arx_can
./arx_can0.sh
```

启动从臂 CAN：

```bash
cd ARX_CAN/arx_can
./arx_can1.sh
```

启动遥操作前确认接口存在：

```bash
ip link show can0     # 主臂
ip link show slcan1   # 从臂
```

## 最简启动方法

先确保两台电脑已经配置默认静态 IP，并能互相 `ping`。

主臂电脑运行：

```bash
cd /path/to/r5remote
./00-sh/ROS2/remote_teleop.sh master
```

从臂电脑运行：

```bash
cd /path/to/r5remote
./00-sh/ROS2/remote_teleop.sh slave
```

未提供 `--peer-ip` 时，脚本自动使用“默认部署”表中的对端地址。

建议先启动主臂，再启动从臂。按 `Ctrl-C` 时，脚本会依次停止 ROS launch 及其子进程。

## 指定一个或多个 Peer

指定一个 Peer：

```bash
./00-sh/ROS2/remote_teleop.sh master \
  --peer-ip 192.168.17.160
```

指定多个 Peer 时重复参数：

```bash
./00-sh/ROS2/remote_teleop.sh master \
  --peer-ip 192.168.17.160 \
  --peer-ip 192.168.17.162
```

重复的地址会自动去重，非法 IPv4 地址会在启动前报错。

从臂也使用相同语法：

```bash
./00-sh/ROS2/remote_teleop.sh slave \
  --peer-ip 192.168.17.161
```

静态 Peer 通常应在通信双方互相配置。例如主臂配置从臂 IP，从臂配置主臂 IP。

## 只加载远程环境

调试 ROS 2 话题时，可以只加载环境而不启动机械臂控制器。

默认 Peer：

```bash
source ./00-sh/ROS2/remote_env.sh master
```

通过逗号分隔的环境变量指定多个 Peer：

```bash
export R5_REMOTE_PEER_IPS=192.168.17.160,192.168.17.162
source ./00-sh/ROS2/remote_env.sh master
```

环境脚本会导出：

| 变量 | 含义 |
| --- | --- |
| `R5_REMOTE_ROLE` | `master` 或 `slave` |
| `R5_REMOTE_EXPECTED_IP` | 当前角色要求的本机 IP |
| `R5_REMOTE_PEER_IPS` | 去重后的逗号分隔 Peer 列表 |
| `R5_REMOTE_CAN_INTERFACE` | 当前角色使用的 SocketCAN 接口 |
| `R5_REMOTE_NODE_NAME` | 控制器节点名 |
| `R5_REMOTE_LAUNCH_FILE` | 当前角色的 launch 文件 |
| `ROS_DOMAIN_ID` | 固定为 `17` |
| `RMW_IMPLEMENTATION` | 固定为 `rmw_cyclonedds_cpp` |
| `CYCLONEDDS_URI` | 根据本机 IP 和 Peer 列表生成的 Cyclone DDS XML |

`remote_env.sh` 必须通过 `source` 加载，不能直接执行。

## 可设置参数

### `remote_teleop.sh` 参数

| 参数 | 是否必需 | 说明 |
| --- | --- | --- |
| `master` / `slave` | 是 | 第一个参数，选择当前电脑角色 |
| `--peer-ip IP` | 否，可重复 | 覆盖默认 Peer，可指定多个地址 |

查看用法：

```bash
./00-sh/ROS2/remote_teleop.sh
```

### 主臂 launch 参数

文件：`open_remote_master_single.launch.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `arm_name` | `arm_master_l` | ROS 节点名 |
| `arm_can_id` | `can0` | 主臂 CAN 接口 |
| `remote_publish_rate_hz` | `200` | 控制消息发布频率 |
| `remote_qos_depth` | `1` | DDS Keep Last 队列深度 |

示例：

```bash
source ./00-sh/ROS2/remote_env.sh master
ros2 launch arx_r5_controller open_remote_master_single.launch.py \
  arm_name:=arm_master_l \
  arm_can_id:=can0 \
  remote_publish_rate_hz:=100 \
  remote_qos_depth:=1
```

### 从臂 launch 参数

文件：`open_remote_slave_single.launch.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `arm_name` | `arm_slave_l` | ROS 节点名 |
| `arm_can_id` | `slcan1` | 从臂 CAN 接口 |
| `remote_publish_slave_status` | `false` | 是否额外发布从臂完整状态 |
| `remote_qos_depth` | `1` | 订阅及可选状态发布的队列深度 |

示例：

```bash
source ./00-sh/ROS2/remote_env.sh slave
ros2 launch arx_r5_controller open_remote_slave_single.launch.py \
  arm_name:=arm_slave_l \
  arm_can_id:=slcan1 \
  remote_publish_slave_status:=false \
  remote_qos_depth:=1
```

`remote_publish_slave_status=true` 会启用 `/arm_slave_l_status` 状态发布，并增加网络流量；
最小远程模式建议保持 `false`。

## 运行检查

加载对应角色环境后执行：

```bash
ros2 topic info /arm_master_l_status -v
ros2 topic hz /arm_master_l_status
ros2 topic type /arm_master_l_status
```

期望类型：

```text
arx5_arm_msg/msg/RemoteControl
```

检查相关进程：

```bash
pgrep -a R5Controller
```

检查网络和 CAN：

```bash
ip -4 -o addr show
ip -details link show can0
ip -details link show slcan1
```

## 启动脚本会自动检查什么

`remote_teleop.sh` 和 `remote_env.sh` 会检查：

1. 角色参数是否合法。
2. Peer 是否为合法 IPv4 地址，并自动去重。
3. `/opt/ros/jazzy/setup.bash` 是否存在。
4. R5 工作空间是否已经编译。
5. 本机是否配置了当前角色要求的静态 IP。
6. `rmw_cyclonedds_cpp` 是否可用。
7. 对应 CAN 接口是否在约 15 秒内出现。
8. 同名 `R5Controller` 节点是否已经运行。

## 常见问题

### 本机 IP 检查失败

脚本要求主臂为 `192.168.17.161`、从臂为 `192.168.17.160`。使用下面命令检查：

```bash
ip -4 -o addr show
```

如果现场网络地址不同，需要同步修改 `remote_env.sh` 中的本机地址和默认 Peer。

### 找不到 `rmw_cyclonedds_cpp`

```bash
sudo apt install ros-jazzy-rmw-cyclonedds-cpp
```

安装后重新打开终端，或重新加载 Jazzy 环境。

### 找不到 CAN 接口

先运行对应的 CAN 启动脚本，并确认 USB-CAN 设备、udev 规则及接口名称正确。

### 两端看不到话题

依次确认：

- 两端 `ROS_DOMAIN_ID` 都是 `17`。
- 两端都使用 `rmw_cyclonedds_cpp`。
- 两端代码和 `RemoteControl` 消息版本一致。
- 双方静态 Peer 地址互相配置正确。
- 主机防火墙允许 DDS UDP 通信。
- 两端网络可以互相 `ping`。

### 为什么配置多个 Peer 后多台机器都可能收到数据

静态 Peer 限制的是 DDS 的发现目标，不是单独的话题访问控制。只要配置的 Peer 上存在
同 Domain、同类型、同话题且 QoS 兼容的订阅者，就可能收到发布数据。严格的身份认证和
访问控制需要 DDS Security 或主机防火墙。

## 安全注意事项

- 启动从臂前，将主从臂放到兼容的初始姿态。
- 首次测试建议降低主臂发布频率，并确保急停可随时触发。
- 从臂收到第一帧主臂消息后会进入位置跟随控制。
- 不要把不可信电脑加入 Peer 列表。
- 修改 CAN 接口、夹爪模式或控制频率后，应先进行低速空载测试。

更精简的远程说明位于 [`00-sh/ROS2/REMOTE_TELEOP.md`](00-sh/ROS2/REMOTE_TELEOP.md)。
