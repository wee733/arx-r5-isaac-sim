// SPDX-License-Identifier: Apache-2.0
#include <arx_vla_control/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <arx5_arm_msg/msg/robot_cmd.hpp>
#include <arx5_arm_msg/msg/robot_status.hpp>
#include <map>
#include <optional>
#include <unistd.h>
using namespace vla;
using Status=arx5_arm_msg::msg::RobotStatus;
using Command=arx5_arm_msg::msg::RobotCmd;
using String=std_msgs::msg::String;
struct Observation { double time; Pose pose; double grip; };

class Executor:public rclcpp::Node {
 public:
  Executor():Node("arx_vla_executor"),config_(declare_parameter<std::string>("config","")) {
    execute_=declare_parameter<bool>("execute",false);
    session_=std::to_string(getpid())+"-"+std::to_string(steady());
    feedback_sub_=create_subscription<Status>("/vla_r5/arm_status",rclcpp::SensorDataQoS().keep_last(1),
      [this](Status::ConstSharedPtr s){
        for(double v:s->joint_pos) if(!std::isfinite(v)) return;
        feedback_=*s; feedback_time_=steady(); have_feedback_=true;
      });
    candidate_=create_publisher<Command>(execute_?"/vla/candidate":"/vla/preview",rclcpp::QoS(1));
    observation_=create_publisher<String>("/vla/observation",rclcpp::QoS(1));
    report_=create_publisher<String>("/vla/status",rclcpp::QoS(1));
    chunk_sub_=create_subscription<String>("/vla/chunk",rclcpp::QoS(1),[this](String::ConstSharedPtr m){
      try {
        require(enabled_,"executor_disabled");
        require(m->data.size()<=32768,"oversized_chunk");
        auto j=Json::parse(m->data);
        require(j.at("observation_id").is_number_unsigned(),"invalid_observation_id");
        auto id=j.at("observation_id").get<uint64_t>();
        require(id>=last_observation_,"observation_reordered");
        const auto it=history_.find(id); require(it!=history_.end(),"unknown_observation");
        auto c=parse_chunk(j,config_,session_,last_sequence_,it->second.time,steady(),it->second.pose,it->second.grip);
        last_sequence_=c.sequence; last_observation_=id;clipped_grippers_+=c.clipped_grippers;
        chunk_=std::move(c); reason_="chunk_accepted";
      }catch(const std::exception& e){ stop(std::string("rejected:")+e.what()); }
    });
    enable_=create_service<std_srvs::srv::SetBool>("/vla/enable",
      [this](std_srvs::srv::SetBool::Request::SharedPtr r,std_srvs::srv::SetBool::Response::SharedPtr s){
        try {
          if(!r->data) { stop("disabled"); s->success=true; s->message=reason_; return; }
          require(fresh(),"feedback_unavailable");
          if(execute_) require(count_publishers("/vla/candidate")==1,"multiple_executors");
          last_q_=measured(); config_.limits(last_q_);
          servo_.reset(last_q_);
          gripper_=config_.grip_hold(feedback_.joint_pos[6]);
          // Every re-arm invalidates old observations and in-flight model replies.
          session_=std::to_string(getpid())+"-"+std::to_string(steady());
          history_.clear(); last_sequence_=0;last_observation_=0;chunk_.reset();
          enabled_=true;reason_="holding";s->success=true;s->message=session_;
        }catch(const std::exception& e){s->success=false;s->message=e.what();}
      });
    timer_=create_wall_timer(std::chrono::milliseconds(20),[this]{tick();});
  }
 private:
  Q measured() const {Q q;std::copy_n(feedback_.joint_pos.begin(),6,q.begin());return q;}
  bool fresh() const {return have_feedback_&&steady()-feedback_time_<config_.feedback_timeout;}
  void publish() {
    Command m;m.header.stamp=now();m.joint_pos=last_q_;m.gripper=gripper_;m.mode=5;
    const auto p=arx::solve::Isometry2Xyzrpy(kin_.forward(last_q_));
    std::copy_n(p.begin(),6,m.end_pos.begin());candidate_->publish(m);++commands_;
  }
  void stop(const std::string& reason) {
    if(enabled_&&fresh()) {last_q_=measured();publish();}
    enabled_=false;chunk_.reset();servo_.reset(last_q_);reason_=reason;
  }
  void tick() {
    const double t=steady(),dt=last_tick_?std::clamp(t-last_tick_,0.0,0.04):0;
    last_tick_=t;
    try {
      if(!fresh()) {if(enabled_) stop("feedback_timeout");}
      else {
        const auto actual=kin_.forward(measured());
        const double measured_grip=config_.grip_hold(feedback_.joint_pos[6]);
        const auto model=vector(config_.to_model(actual));
        const auto id=++observation_id_;
        history_.emplace(id,Observation{t,actual,measured_grip});
        while(history_.size()>128) history_.erase(history_.begin());
        Json j={{"schema","arx_vla_observation_v1"},{"session_id",session_},{"observation_id",id},
          {"received_monotonic_s",feedback_time_},{"sample_monotonic_s",t},
          {"driver_stamp_s",rclcpp::Time(feedback_.header.stamp).seconds()},
          {"feedback_age_s",t-feedback_time_},{"calibrated",config_.frames_verified&&config_.gripper_verified},
          {"state",{{"eef",model},{"gripper",{config_.grip_unit(feedback_.joint_pos[6])*config_.finger_max}},{"joints",measured()}}},
          {"gripper_opening_reference",config_.raw.value("gripper_opening_reference",std::string("unverified"))},
          {"sdk_eef",vector(actual)},{"sdk_gripper_raw",feedback_.joint_pos[6]},
          {"driver_eef_xyzrpy",feedback_.end_pos}};
        String message;message.data=j.dump();observation_->publish(message);
        if(enabled_) {
          config_.limits(measured());
          if(execute_) require(count_publishers("/vla/candidate")==1,"multiple_executors");
          if(chunk_&&t>=chunk_->start+chunk_->duration()) {
            chunk_.reset();last_q_=measured();servo_.reset(last_q_);reason_="chunk_exhausted_hold";
          }
          if(chunk_) {
            auto [target,g]=chunk_->sample(t);
            const auto solution=kin_.inverse(target,servo_.nominal,measured(),config_);
            last_q_=servo_.step(solution,measured(),dt,config_);
            gripper_+=std::clamp(g-gripper_,-config_.gripper_speed*dt,config_.gripper_speed*dt);
            reason_=execute_?"executing":"preview";
          }
          publish();
        }
      }
    }catch(const std::exception& e){stop(e.what());}
    // Status at 10 Hz, observations/execution at 50 Hz. Never block on model inference.
    if(t-last_report_>=0.1) {
      last_report_=t;String m;m.data=Json({{"enabled",enabled_},{"execute",execute_},
        {"state",reason_},{"session_id",session_},{"commands",commands_},
        {"last_sequence",last_sequence_},{"feedback_age_s",have_feedback_?t-feedback_time_:-1},
        {"q_command",last_q_},{"q_nominal",servo_.nominal},{"q_correction",servo_.integral},
        {"q_actual",feedback_.joint_pos},{"gripper_command",gripper_},{"gripper_clipped_samples",clipped_grippers_}}).dump();
      report_->publish(m);
      if(reason_!=last_reason_) {RCLCPP_INFO(get_logger(),"%s",m.data.c_str());last_reason_=reason_;}
    }
  }
  Config config_;Kinematics kin_;JointServo servo_;Status feedback_;
  bool execute_=false,enabled_=false,have_feedback_=false;
  double feedback_time_=0,last_tick_=0,last_report_=0,gripper_=0;
  uint64_t observation_id_=0,last_observation_=0,last_sequence_=0,commands_=0,clipped_grippers_=0;
  Q last_q_{};std::string session_,reason_="disabled",last_reason_;
  std::map<uint64_t,Observation> history_;std::optional<Chunk> chunk_;
  rclcpp::Subscription<Status>::SharedPtr feedback_sub_;
  rclcpp::Subscription<String>::SharedPtr chunk_sub_;
  rclcpp::Publisher<Command>::SharedPtr candidate_;
  rclcpp::Publisher<String>::SharedPtr observation_,report_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_;
  rclcpp::TimerBase::SharedPtr timer_;
};
int main(int argc,char** argv) {
  rclcpp::init(argc,argv);
  try {rclcpp::spin(std::make_shared<Executor>());}
  catch(const std::exception& e){RCLCPP_ERROR(rclcpp::get_logger("executor"),"%s",e.what());return 1;}
  rclcpp::shutdown();return 0;
}
