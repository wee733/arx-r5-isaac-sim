#pragma once

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <memory>

#include "arx_r5_src/interfaces/InterfacesThread.hpp"

#include "arx5_arm_msg/msg/robot_cmd.hpp"
#include "arx5_arm_msg/msg/remote_control.hpp"
#include "arx5_arm_msg/msg/robot_status.hpp"
#include "arm_control/msg/pos_cmd.hpp"

namespace arx::r5 {
class R5Controller : public rclcpp::Node {
 public:
  R5Controller();
  void cleanup() {
    interfaces_ptr_.reset();
  }

  void CmdCallback(const arx5_arm_msg::msg::RobotCmd::SharedPtr msg);
  void PubState();
  void PubRemoteState();

  void VrCmdCallback(const arm_control::msg::PosCmd::SharedPtr msg);
  void VrPubState();

  void FollowCmdCallback(const arx5_arm_msg::msg::RemoteControl::SharedPtr msg);

 private:
  std::shared_ptr<InterfacesThread> interfaces_ptr_;
  enum class CatchControlMode {
    kPosition,
    kTorque
  } catch_control_mode_;

  // 通常、VR 和 remote主机模式下
  rclcpp::Publisher<arx5_arm_msg::msg::RobotStatus>::SharedPtr joint_state_publisher_;
  rclcpp::Publisher<arx5_arm_msg::msg::RemoteControl>::SharedPtr remote_control_publisher_;
  // vr模式下
  rclcpp::Publisher<arm_control::msg::PosCmd>::SharedPtr vr_joint_state_publisher_;

  // 通常模式下
  rclcpp::Subscription<arx5_arm_msg::msg::RobotCmd>::SharedPtr joint_state_subscriber_;
  // vr模式下
  rclcpp::Subscription<arm_control::msg::PosCmd>::SharedPtr vr_joint_state_subscriber_;
  // remote从机模式下
  rclcpp::Subscription<arx5_arm_msg::msg::RemoteControl>::SharedPtr follow_joint_state_subscriber_;

  rclcpp::TimerBase::SharedPtr timer_;
};
}