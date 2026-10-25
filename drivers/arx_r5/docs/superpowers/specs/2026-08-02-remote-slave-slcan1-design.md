# Remote Slave SLCAN1 Configuration

## Scope

Configure the single remote-slave ROS 2 launch file for the USB-CAN adapter
currently connected to this computer. The detected device mapping is
`/dev/arxcan1 -> /dev/ttyACM0`, and `arx_can1.sh` exposes it as `slcan1`.

## Design

- Change only the default `arm_can_id` in
  `open_remote_slave_single.launch.py` from `slcan0` to `slcan1`.
- Keep `remote_slave.yaml` unchanged because it also supplies parameters for
  the dual-arm launch path.
- Keep the launch argument overridable so another CAN interface can still be
  selected explicitly.

## Data Path

`/dev/arxcan1` is opened by `slcand`, which creates the SocketCAN interface
`slcan1`. The single remote-slave launch passes `slcan1` to `R5Controller` as
the `arm_can_id` parameter.

## Verification

1. Before the change, the installed launch reports `slcan0` as its default.
2. After the change and rebuild, it reports `slcan1`.
3. `slcan1` reaches the UP and ERROR-ACTIVE state.
4. The remote-slave node initializes without publishing a master status or
   sending an explicit motion command during the test.

## Safety

Do not publish `/arm_master_l_status` during initialization testing. A remote
slave starts position control when it receives that topic, so live following
must be tested only after both arms are placed in compatible poses and an
operator has access to the emergency stop.
