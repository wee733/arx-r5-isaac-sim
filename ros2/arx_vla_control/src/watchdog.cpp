// SPDX-License-Identifier: Apache-2.0
// Separate process: retains a hold if the executor or model process disappears.
#include <arx_vla_control/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <arx5_arm_msg/msg/robot_cmd.hpp>
#include <arx5_arm_msg/msg/robot_status.hpp>
using namespace vla;
using Command=arx5_arm_msg::msg::RobotCmd;
using Status=arx5_arm_msg::msg::RobotStatus;
class Watchdog:public rclcpp::Node {
 public:
  Watchdog():Node("arx_vla_watchdog"),config_(declare_parameter<std::string>("config","")) {
    driver_name_=declare_parameter<std::string>("driver_node_name","arm_vla");
    output_=create_publisher<Command>("/vla_r5/arm_cmd",rclcpp::QoS(1));
    report_=create_publisher<std_msgs::msg::String>("/vla/watchdog/status",rclcpp::QoS(1));
    feedback_sub_=create_subscription<Status>("/vla_r5/arm_status",rclcpp::SensorDataQoS().keep_last(1),
      [this](Status::ConstSharedPtr s){for(double v:s->joint_pos) if(!std::isfinite(v)) return;
        feedback_=*s;feedback_time_=steady();have_feedback_=true;});
    input_=create_subscription<Command>("/vla/candidate",rclcpp::QoS(1),[this](Command::ConstSharedPtr m){
      if(!enabled_) return;
      try {
        require(fresh(),"feedback_timeout");exclusive();
        const double age=(now()-rclcpp::Time(m->header.stamp)).seconds();
        require(age>=-0.01&&age<0.1,"stale_candidate");
        require(m->mode==5,"invalid_control_mode");config_.limits(m->joint_pos);
        require(std::isfinite(m->gripper)&&m->gripper>=0&&m->gripper<=5,"invalid_gripper");
        const double dt=std::clamp(steady()-last_received_,0.0,0.04);
        for(size_t i=0;i<6;++i) {
          require(std::abs(m->joint_pos[i]-feedback_.joint_pos[i])<=config_.tracking,"tracking_gap");
          require(std::abs(m->joint_pos[i]-last_.joint_pos[i])<=config_.speed*dt+0.002,"command_speed");
        }
        require(std::abs(m->gripper-last_.gripper)<=config_.gripper_speed*dt+0.01,"gripper_speed");
        for(double v:m->end_pos) require(std::isfinite(v),"invalid_end_pose");
        last_=*m;last_received_=steady();last_.header.stamp=now();output_->publish(last_);++forwarded_;
      }catch(const std::exception& e){hold(e.what());}
    });
    enable_=create_service<std_srvs::srv::SetBool>("/vla/watchdog/enable",
      [this](std_srvs::srv::SetBool::Request::SharedPtr r,std_srvs::srv::SetBool::Response::SharedPtr s){
        try {
          if(!r->data){hold("disabled");s->success=true;s->message=reason_;return;}
          require(fresh(),"feedback_unavailable");exclusive();
          std::copy_n(feedback_.joint_pos.begin(),6,last_.joint_pos.begin());config_.limits(last_.joint_pos);
          last_.gripper=config_.grip_hold(feedback_.joint_pos[6]);last_.mode=5;
          last_received_=steady();enabled_=true;reason_="armed";s->success=true;s->message=reason_;
        }catch(const std::exception& e){s->success=false;s->message=e.what();}
      });
    timer_=create_wall_timer(std::chrono::milliseconds(20),[this]{
      if(enabled_) {
        if(!fresh()) hold("feedback_timeout");
        else if(steady()-last_received_>0.15) hold("executor_timeout");
        else {try{exclusive();}catch(const std::exception& e){hold(e.what());}}
      }
      if(steady()-last_report_>0.1){last_report_=steady();std_msgs::msg::String m;
        m.data=Json({{"enabled",enabled_},{"state",reason_},{"forwarded",forwarded_}}).dump();report_->publish(m);
        if(reason_!=last_reason_){RCLCPP_INFO(get_logger(),"%s",m.data.c_str());last_reason_=reason_;}}
    });
  }
 private:
  bool fresh()const{return have_feedback_&&steady()-feedback_time_<config_.feedback_timeout;}
  void exclusive(){
    require(count_publishers("/vla_r5/arm_cmd")==1,"multiple_driver_publishers");
    require(count_publishers("/vla/candidate")==1,"executor_missing_or_multiple");
    // Recorders/diagnostics may subscribe without owning hardware. Count the
    // intended driver endpoint, not every read-only consumer of this topic.
    size_t drivers=0;
    for(const auto& endpoint:get_subscriptions_info_by_topic("/vla_r5/arm_cmd"))
      if(endpoint.node_name()==driver_name_) ++drivers;
    require(drivers==1,"driver_missing_or_multiple");
  }
  void hold(const std::string& reason){
    if(enabled_){
      // With lost feedback freeze the last bounded command; never invent a zero/home pose.
      if(fresh())std::copy_n(feedback_.joint_pos.begin(),6,last_.joint_pos.begin());
      last_.header.stamp=now();last_.mode=5;output_->publish(last_);
    }
    enabled_=false;reason_=reason;
  }
  Config config_;Status feedback_;Command last_;
  bool have_feedback_=false,enabled_=false;
  double feedback_time_=0,last_received_=0,last_report_=0;
  uint64_t forwarded_=0;std::string reason_="disabled",last_reason_;
  std::string driver_name_;
  rclcpp::Publisher<Command>::SharedPtr output_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr report_;
  rclcpp::Subscription<Command>::SharedPtr input_;
  rclcpp::Subscription<Status>::SharedPtr feedback_sub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_;
  rclcpp::TimerBase::SharedPtr timer_;
};
int main(int argc,char** argv){rclcpp::init(argc,argv);
  try{rclcpp::spin(std::make_shared<Watchdog>());}
  catch(const std::exception& e){RCLCPP_ERROR(rclcpp::get_logger("watchdog"),"%s",e.what());return 1;}
  rclcpp::shutdown();return 0;
}
