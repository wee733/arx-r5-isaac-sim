#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <arx5_arm_msg/msg/robot_cmd.hpp>
#include <arx5_arm_msg/msg/robot_status.hpp>
#include <pico_r5_bridge/preview.hpp>
#include <kinematic_solver.hpp>
#include <arx_r5_src/interfaces/InterfacesThread.hpp>
#include <array>
#include <chrono>

using pico_r5::Json;
using pico_r5::Pose;
using Status = arx5_arm_msg::msg::RobotStatus;
using Command = arx5_arm_msg::msg::RobotCmd;

namespace {
double steady_now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
Pose from_transform(const Eigen::Isometry3d &t) { return {t.translation(), Eigen::Quaterniond(t.rotation())}; }
Pose from_json(const Json &j) {
  const auto p = j.at("target_xyz").get<std::array<double,3>>();
  const auto q = j.at("target_xyzw").get<std::array<double,4>>();
  Pose result{Eigen::Vector3d(p[0],p[1],p[2]), Eigen::Quaterniond(q[3],q[0],q[1],q[2])};
  if (!result.p.allFinite() || !result.q.coeffs().allFinite() || std::abs(result.q.norm()-1)>0.01)
    throw std::runtime_error("invalid_preview_pose");
  result.q.normalize();
  return result;
}
}

class PicoIK : public rclcpp::Node {
 public:
  PicoIK() : Node("pico_ik_bridge") {
    execute_ = declare_parameter<bool>("execute",false);
    const auto mode=declare_parameter<std::string>("translation_mode","pose");
    if (mode!="pose" && mode!="joystick") throw std::runtime_error("invalid translation_mode");
    joystick_=mode=="joystick";
    declare_parameter<bool>("orientation_enabled",false);
    declare_parameter<bool>("gripper_enabled",false);
    max_translation_ = declare_parameter<double>("max_clutch_translation_m",0.03);
    max_rotation_ = declare_parameter<double>("max_clutch_rotation_rad",0.2);
    max_joint_speed_ = declare_parameter<double>("max_joint_speed_rad_s",0.2);
    for (double v : {max_translation_,max_rotation_,max_joint_speed_})
      if (!std::isfinite(v) || v<=0) throw std::runtime_error("invalid motion limits");
    arm_sub_ = create_subscription<Status>("/pico_r5/arm_status",rclcpp::SensorDataQoS().keep_last(1),
      [this](Status::ConstSharedPtr s) {
        for (double q:s->joint_pos) if (!std::isfinite(q)) return;
        for (double q:s->end_pos) if (!std::isfinite(q)) return;
        feedback_ = *s; have_feedback_=true; feedback_time_=steady_now();
      });
    preview_sub_ = create_subscription<std_msgs::msg::String>("/pico_preview/status",rclcpp::QoS(1),
      [this](std_msgs::msg::String::ConstSharedPtr m) {
        try {
          auto j=Json::parse(m->data);
          if (j.value("translation_mode",std::string("pose"))!=(joystick_?"joystick":"pose"))
            throw std::runtime_error("preview_control_mode_mismatch");
          stick_active_=j.value("stick_active",false);
          rotation_active_=j.value("rotation_active",false);
          input_pose_=from_json(j);
          input_active_=j.at("active").get<bool>() && j.at("receive_age_ms").get<double>()>=0
                        && j.at("receive_age_ms").get<double>()<=200;
          trigger_=j.at("trigger").get<double>();
          if (!std::isfinite(trigger_) || trigger_<0 || trigger_>1) throw std::runtime_error("invalid trigger");
          input_time_=steady_now();
          if (!input_active_) release_seen_=true;
        } catch (...) { input_active_=false; release_seen_=false; }
      });
    debug_pub_=create_publisher<Command>("/pico_ik/robot_cmd_preview",rclcpp::QoS(1));
    // No real command publisher exists in inspection mode.
    if (execute_) command_pub_=create_publisher<Command>("/pico_r5/arm_cmd",rclcpp::QoS(1));
    report_pub_=create_publisher<std_msgs::msg::String>("/pico_ik/status",rclcpp::QoS(1));
    timer_=create_wall_timer(std::chrono::milliseconds(20),[this]{tick();});
    RCLCPP_WARN(get_logger(),"execute=%s; native ARX IK, measured-joint anchor, translation first; release grip before engaging",
                execute_?"true":"false");
  }

 private:
  Pose forward(const std::array<double,6> &q) {
    auto input=q;
    double xyzrpy[6]; solver_.computeForwardKinematics(input.data(),xyzrpy);
    for (double v:xyzrpy) if (!std::isfinite(v)) throw std::runtime_error("invalid_forward_kinematics");
    return from_transform(arx::solve::Xyzrpy2Isometry(xyzrpy));
  }
  std::array<double,6> measured() const {
    std::array<double,6> q; std::copy_n(feedback_.joint_pos.begin(),6,q.begin()); return q;
  }
  Command command(const std::array<double,6> &q,double gripper) {
    Command m; m.header.stamp=now(); m.joint_pos=q;
    m.mode=arx::r5::InterfacesThread::state::POSITION_CONTROL;
    m.gripper=gripper;
    return m;
  }
  void hold(const std::string &reason,double t) {
    if (engaged_ && have_feedback_ && t-feedback_time_<0.2) {
      // Existing POSITION_CONTROL: replace outstanding goal with measured joints on release.
      const auto m=command(measured(),gripper_cmd_);
      debug_pub_->publish(m); if (execute_) command_pub_->publish(m);
    }
    engaged_=false;
    last_stick_=false; last_rotation_=false;
    state_=reason;
  }
  void tick() {
    const double t=steady_now();
    const double dt=last_tick_>0?std::clamp(t-last_tick_,0.0,0.04):0;
    last_tick_=t;
    try {
      if (!have_feedback_ || t-feedback_time_>0.2) {
        hold("waiting_for_arm_feedback",t); release_seen_=false;
      } else if (t-input_time_>0.2) {
        hold("preview_timeout",t); release_seen_=false;
      } else if (!input_active_) {
        hold("released",t);
      } else if (!engaged_ && !release_seen_) {
        state_="release_grip_to_arm";
      } else {
        if (!engaged_) {
          robot_anchor_=forward(measured());
          input_anchor_=input_pose_;
          last_command_=measured();
          // Same measured gripper scaling as the repository's FollowCmdCallback.
          gripper_cmd_=std::clamp(feedback_.joint_pos[6]*5.0,0.0,5.0);
          start_gripper_=gripper_cmd_; start_trigger_=trigger_;
          orientation_=get_parameter("orientation_enabled").as_bool();
          gripper_=get_parameter("gripper_enabled").as_bool();
          engaged_=true; release_seen_=false;
        }
        Pose wanted=robot_anchor_;
        if (joystick_) {
          const Pose actual=forward(measured());
          if (stick_active_!=last_stick_) {
            // Center stops translation even while the orientation clutch stays held.
            robot_anchor_.p=actual.p; input_anchor_.p=input_pose_.p;
          }
          if (rotation_active_ && !last_rotation_) {
            robot_anchor_.q=actual.q; input_anchor_.q=input_pose_.q;
            orientation_=get_parameter("orientation_enabled").as_bool();
            gripper_=get_parameter("gripper_enabled").as_bool();
            start_gripper_=gripper_cmd_; start_trigger_=trigger_;
          }
          last_stick_=stick_active_; last_rotation_=rotation_active_;
          wanted=robot_anchor_;
        }
        wanted.p+=input_pose_.p-input_anchor_.p;
        if (orientation_) wanted.q=(input_pose_.q*input_anchor_.q.conjugate()*robot_anchor_.q).normalized();
        if ((wanted.p-robot_anchor_.p).norm()>max_translation_ ||
            wanted.q.angularDistance(robot_anchor_.q)>max_rotation_) throw std::runtime_error("clutch_range_exceeded");
        Eigen::Isometry3d transform=Eigen::Isometry3d::Identity();
        transform.translation()=wanted.p; transform.linear()=wanted.q.toRotationMatrix();
        auto xyzrpy=arx::solve::Isometry2Xyzrpy(transform);
        auto solution=last_command_;
        solver_.computeInverseKinematics(xyzrpy.data(),solution.data());
        const auto current=measured();
        for (size_t i=0;i<6;++i) {
          if (!std::isfinite(solution[i])) throw std::runtime_error("nonfinite_ik");
          solution[i]=last_command_[i]+std::remainder(solution[i]-last_command_[i],2*M_PI);
          if (std::abs(solution[i]-last_command_[i])>0.25 || std::abs(solution[i]-current[i])>0.3)
            throw std::runtime_error("ik_branch_or_tracking_gap");
        }
        const auto reconstructed=forward(solution);
        position_error_=(reconstructed.p-wanted.p).norm();
        rotation_error_=reconstructed.q.angularDistance(wanted.q);
        if (position_error_>0.001 || rotation_error_>0.01) throw std::runtime_error("ik_residual_exceeded");
        auto limited=solution;
        for (size_t i=0;i<6;++i) limited[i]=last_command_[i]+std::clamp(solution[i]-last_command_[i],-max_joint_speed_*dt,max_joint_speed_*dt);
        if (gripper_ && (!joystick_ || rotation_active_)) {
          // Relative trigger mapping avoids a jump when gripper mode is first engaged.
          const double desired=std::clamp(start_gripper_-(trigger_-start_trigger_)*5.0,0.0,5.0);
          gripper_cmd_+=std::clamp(desired-gripper_cmd_,-1.0*dt,1.0*dt);
        }
        auto m=command(limited,gripper_cmd_);
        std::copy_n(xyzrpy.begin(),6,m.end_pos.begin());
        debug_pub_->publish(m); if (execute_) command_pub_->publish(m);
        last_command_=limited; target_=wanted; ++commands_;
        state_=execute_?"executing":"ik_preview";
      }
    } catch (const std::exception &e) {
      hold(e.what(),t); release_seen_=false;
    }
    if (t-last_report_>1 || state_!=last_state_) {
      last_report_=t; last_state_=state_;
      const Json j={{"execute",execute_},{"state",state_},{"commands",commands_},
        {"feedback_age_ms",have_feedback_?(t-feedback_time_)*1000:-1},
        {"q_actual",feedback_.joint_pos},{"q_command",last_command_},
        {"actual_xyzrpy",feedback_.end_pos},{"target_xyz",{target_.p.x(),target_.p.y(),target_.p.z()}},
        {"ik_position_error_m",position_error_},{"ik_rotation_error_rad",rotation_error_},
        {"gripper_command",gripper_cmd_},{"orientation_enabled",orientation_},{"gripper_enabled",gripper_}};
      // Target orientation is reported as well as measured RPY for axis verification.
      auto report=j;
      report["translation_mode"]=joystick_?"joystick":"pose";
      report["stick_active"]=stick_active_; report["rotation_active"]=rotation_active_;
      report["target_xyzw"]={target_.q.x(),target_.q.y(),target_.q.z(),target_.q.w()};
      std_msgs::msg::String message; message.data=report.dump(); report_pub_->publish(message);
      RCLCPP_INFO(get_logger(),"%s",message.data.c_str());
    }
  }
  arx::KinematicSolver solver_;
  Status feedback_;
  Pose input_pose_,input_anchor_,robot_anchor_,target_;
  std::array<double,6> last_command_{};
  bool execute_=false,have_feedback_=false,input_active_=false,release_seen_=false,engaged_=false;
  bool orientation_=false,gripper_=false;
  bool joystick_=false,stick_active_=false,rotation_active_=false,last_stick_=false,last_rotation_=false;
  double input_time_=0,feedback_time_=0,last_tick_=0,last_report_=0;
  double max_translation_=0.03,max_rotation_=0.2,max_joint_speed_=0.2;
  double trigger_=0,start_trigger_=0,gripper_cmd_=0,start_gripper_=0,position_error_=0,rotation_error_=0;
  uint64_t commands_=0;
  std::string state_,last_state_;
  rclcpp::Subscription<Status>::SharedPtr arm_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr preview_sub_;
  rclcpp::Publisher<Command>::SharedPtr command_pub_,debug_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr report_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc,char **argv) {
  rclcpp::init(argc,argv);
  try { rclcpp::spin(std::make_shared<PicoIK>()); }
  catch (const std::exception &e) {
    RCLCPP_ERROR(rclcpp::get_logger("pico_ik_bridge"),"%s",e.what()); rclcpp::shutdown(); return 1;
  }
  rclcpp::shutdown(); return 0;
}
