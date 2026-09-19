# VLA scene reset protocol

`/vla/scene_command` is a `std_msgs/msg/Int32` sent from the episode driver to
Isaac Sim. It contains one reset request, not merely a seed:

```
bits 30..16: request token (1..32767)
bits 15..0 : seed + 1 (seed 0..65534)
```

Zero is reserved because Isaac Sim's `ROS2Subscriber` reports zero before it
has received a ROS message. Each ROS driver process starts from a random
non-zero token and increments it for every request, including same-seed
retries. This also keeps a restarted ROS launch from repeating the final
command held by an Isaac Sim process that was intentionally left running.

After it has released any held object, teleported the block, and updated the
placement marker, Isaac Sim publishes the exact packed integer on
`/vla/scene_reset_ack` (`std_msgs/msg/Int32`). The driver proceeds only after a
new ACK equal to its request; `/vla/object_attached` is exclusively the grasp
attachment/release state and is not a reset acknowledgement.

For a manual reset, run `scripts/request_vla_scene_reset.sh [seed] [token]` and
inspect both the Isaac Sim log and `/vla/scene_reset_ack`.
