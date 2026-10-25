# R5 ROS 2 Remote Teleoperation (Dual Jazzy / 双 Jazzy)

This repository supports the master and slave computers with one checkout.
This branch is the **dual-Jazzy (双 Jazzy)** deployment: both the master and slave computers run ROS 2 Jazzy.

## Roles

| Role | ROS 2 | IP | CAN device | SocketCAN | Node |
| --- | --- | --- | --- | --- | --- |
| Master | Jazzy | 192.168.17.161/24 | /dev/arxcan0 | can0 | arm_master_l |
| Slave | Jazzy | 192.168.17.160/24 | /dev/arxcan1 | slcan1 | arm_slave_l |

Both roles use `ROS_DOMAIN_ID=17` and the system Jazzy Cyclone DDS packages.
DDS multicast discovery is disabled; only configured static peers are contacted.

Install the DDS implementation on both computers:

```bash
sudo apt install ros-jazzy-rmw-cyclonedds-cpp
```

## CAN

Install `ARX_CAN/arx_can.rules` on both computers and reload udev rules. Start
CAN in a dedicated terminal before remote teleoperation.

Master:

```bash
cd ARX_CAN/arx_can
./arx_can0.sh
```

Slave:

```bash
cd ARX_CAN/arx_can
./arx_can1.sh
```

## Remote Mode

With no peer option, master targets `192.168.17.160` and slave targets
`192.168.17.161`:

```bash
./00-sh/ROS2/remote_teleop.sh master
./00-sh/ROS2/remote_teleop.sh slave
```

Repeat `--peer-ip` to contact multiple DDS peers:

```bash
./00-sh/ROS2/remote_teleop.sh master \
  --peer-ip 192.168.17.160 \
  --peer-ip 192.168.17.162
```

The same list can be used when sourcing the environment directly:

```bash
export R5_REMOTE_PEER_IPS=192.168.17.160,192.168.17.162
source ./00-sh/ROS2/remote_env.sh master
```

Static peers stop normal DDS discovery broadcasts. Any configured peer with a
matching topic can receive the topic, so configure only trusted addresses. Use
a host firewall or DDS Security if protection from an actively malicious host
is required.

The cross-computer control topic is `/arm_master_l_status`, with type
`arx5_arm_msg/msg/RemoteControl`:

- Master publishes at 200 Hz.
- QoS is Best Effort with depth 1, so stale state is dropped.
- Each sample contains seven joint positions and optional gripper torque.
- Slave status publication is disabled by default.

The control parameters can be changed without rebuilding:

```bash
ros2 launch arx_r5_controller open_remote_master_single.launch.py \
  remote_publish_rate_hz:=100 remote_qos_depth:=1

ros2 launch arx_r5_controller open_remote_slave_single.launch.py \
  remote_publish_slave_status:=false remote_qos_depth:=1
```

## Safety

Place both arms in compatible initial poses and keep the emergency stop
accessible before starting the slave. The slave enters position following when
it receives master control messages.
