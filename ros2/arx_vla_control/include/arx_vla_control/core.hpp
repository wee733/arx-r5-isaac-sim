// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <kinematic_solver.hpp>
#include <arx_r5_src/interfaces/InterfacesThread.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace vla {
using Json = nlohmann::json;
using Q = std::array<double,6>;
using Pose = Eigen::Isometry3d;
inline double steady() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void require(bool b, const char* reason) { if (!b) throw std::runtime_error(reason); }
template<size_t N> std::array<double,N> numbers(const Json& j) {
  require(j.is_array() && j.size()==N,"invalid_vector_shape");
  std::array<double,N> a{};
  for(size_t i=0;i<N;++i) { a[i]=j.at(i).get<double>(); require(std::isfinite(a[i]),"nonfinite_vector"); }
  return a;
}
inline Pose pose(const Q& a) {
  Pose t=Pose::Identity(); t.translation()=Eigen::Vector3d(a[0],a[1],a[2]);
  const Eigen::Vector3d r(a[3],a[4],a[5]);
  if(r.norm()>1e-12) t.linear()=Eigen::AngleAxisd(r.norm(),r.normalized()).toRotationMatrix();
  return t;
}
inline Q vector(const Pose& t) {
  Eigen::AngleAxisd a(t.rotation()); const auto r=a.axis()*a.angle();
  return {t.translation().x(),t.translation().y(),t.translation().z(),r.x(),r.y(),r.z()};
}
inline Pose blend(const Pose& a,const Pose& b,double u) {
  Pose t=Pose::Identity(); t.translation()=(1-u)*a.translation()+u*b.translation();
  t.linear()=Eigen::Quaterniond(a.rotation()).slerp(u,Eigen::Quaterniond(b.rotation())).toRotationMatrix();
  return t;
}
struct Config {
  Json raw;
  Pose base,tool;
  Q lower,upper;
  bool frames_verified,gripper_verified;
  double speed,tracking,branch,translation,rotation,feedback_timeout,observation_age;
  double finger_max,feedback_open,feedback_closed,gripper_speed;
  double tracking_ki,correction_limit;
  std::vector<std::array<double,2>> grip_response;
  explicit Config(const std::string& path) {
    std::ifstream in(path); require(in.good(),"missing_config"); in>>raw;
    require(raw.at("schema")=="arx_vla_config_v1","config_schema");
    base=pose(numbers<6>(raw.at("sdk_base_from_model_mount")));
    tool=pose(numbers<6>(raw.at("model_link6_from_sdk_tool")));
    lower=numbers<6>(raw.at("joint_lower")); upper=numbers<6>(raw.at("joint_upper"));
    for(size_t i=0;i<6;++i) require(lower[i]<upper[i],"invalid_joint_limits");
    frames_verified=raw.at("frames_verified"); gripper_verified=raw.at("gripper_verified");
    speed=raw.at("max_joint_speed_rad_s"); tracking=raw.at("max_tracking_error_rad");
    branch=raw.at("max_ik_jump_rad"); translation=raw.at("max_chunk_translation_m");
    rotation=raw.at("max_chunk_rotation_rad"); feedback_timeout=raw.at("feedback_timeout_s");
    observation_age=raw.at("max_observation_age_s"); finger_max=raw.at("finger_max_m");
    feedback_open=raw.at("gripper_feedback_open"); feedback_closed=raw.at("gripper_feedback_closed");
    gripper_speed=raw.at("max_gripper_command_speed_s");
    tracking_ki=raw.value("tracking_integral_gain_s",0.0);
    correction_limit=raw.value("max_tracking_correction_rad",0.02);
    require(std::isfinite(tracking_ki)&&tracking_ki>=0&&tracking_ki<=2,"invalid_tracking_gain");
    require(std::isfinite(correction_limit)&&correction_limit>=0&&correction_limit<=0.03,"invalid_tracking_correction");
    for(double v:{speed,tracking,branch,translation,rotation,feedback_timeout,observation_age,finger_max,gripper_speed})
      require(std::isfinite(v)&&v>0,"invalid_config_limit");
    require(std::isfinite(feedback_open)&&std::isfinite(feedback_closed)&&feedback_open!=feedback_closed,"invalid_gripper_calibration");
    if(raw.contains("gripper_response_raw_command")) {
      for(const auto& point:raw.at("gripper_response_raw_command"))grip_response.push_back(numbers<2>(point));
    } else grip_response={{{feedback_closed,0}},{{feedback_open,5}}};
    require(grip_response.size()>=2&&grip_response.front()[1]==0&&grip_response.back()[1]==5,"invalid_gripper_response");
    require(std::abs(grip_response.front()[0]-feedback_closed)<1e-6&&std::abs(grip_response.back()[0]-feedback_open)<1e-6,"gripper_endpoint_mismatch");
    for(size_t i=1;i<grip_response.size();++i)
      require(grip_response[i][0]>grip_response[i-1][0]&&grip_response[i][1]>grip_response[i-1][1],"nonmonotonic_gripper_response");
  }
  Pose to_sdk(const Pose& p) const { return base*p*tool; }
  Pose to_model(const Pose& p) const { return base.inverse()*p*tool.inverse(); }
  double grip_unit(double raw_value) const {
    double u=(raw_value-feedback_closed)/(feedback_open-feedback_closed);
    require(std::isfinite(u)&&u>=-0.05&&u<=1.05,"gripper_feedback_out_of_range");
    return std::clamp(u,0.0,1.0);
  }
  double grip_command(double metres) const {
    require(std::isfinite(metres)&&metres>=-0.001&&metres<=0.045,"gripper_target_out_of_range");
    return grip_hold(feedback_closed+std::clamp(metres,0.0,finger_max)/finger_max*(feedback_open-feedback_closed));
  }
  double grip_hold(double raw_value) const {
    grip_unit(raw_value); // finite/range validation, then saturate only near measured endpoints
    const double value=std::clamp(raw_value,feedback_closed,feedback_open);
    for(size_t i=1;i<grip_response.size();++i)if(value<=grip_response[i][0]) {
      const auto& a=grip_response[i-1];const auto& b=grip_response[i];
      return std::clamp(a[1]+(value-a[0])/(b[0]-a[0])*(b[1]-a[1]),0.0,5.0);
    }
    return 5;
  }
  void limits(const Q& q) const {
    for(size_t i=0;i<6;++i) require(std::isfinite(q[i])&&q[i]>=lower[i]&&q[i]<=upper[i],"joint_limit");
  }
};
// Bounded slow position-error trim. Retains vendor gains, native IK, all joint
// limits and the independent watchdog. Anti-windup on output slew/joint limits.
struct JointServo {
  Q nominal{},output{},integral{};
  void reset(const Q& measured){nominal=output=measured;integral.fill(0);}
  Q step(const Q& target,const Q& measured,double dt,const Config& c){
    require(dt>=0&&dt<=0.04,"invalid_servo_dt");c.limits(target);
    for(size_t i=0;i<6;++i){
      nominal[i]+=std::clamp(target[i]-nominal[i],-c.speed*dt,c.speed*dt);
      const double error=nominal[i]-measured[i];
      require(std::abs(error)<=c.tracking,"servo_tracking_gap");
      const double old=integral[i];
      if(c.tracking_ki==0)integral[i]=0;
      else if(std::abs(error)>0.0008)
        integral[i]=std::clamp(old+c.tracking_ki*error*dt,-c.correction_limit,c.correction_limit);
      double wanted=nominal[i]+integral[i];
      const double bounded=std::clamp(wanted,c.lower[i],c.upper[i]);
      const double sent=output[i]+std::clamp(bounded-output[i],-c.speed*dt,c.speed*dt);
      if((wanted-sent)*error>1e-10)integral[i]=old;
      output[i]=sent;
    }
    return output;
  }
};
// The same ARX native FK/IK and branch/residual checks used by pico_ik_bridge.
class Kinematics {
  arx::KinematicSolver solver_;
 public:
  Pose forward(Q q) {
    double p[6]; solver_.computeForwardKinematics(q.data(),p);
    for(double v:p) require(std::isfinite(v),"invalid_fk");
    return arx::solve::Xyzrpy2Isometry(p);
  }
  Q inverse(const Pose& target,const Q& previous,const Q& measured,const Config& c) {
    auto p=arx::solve::Isometry2Xyzrpy(target); Q q=previous;
    solver_.computeInverseKinematics(p.data(),q.data());
    for(size_t i=0;i<6;++i) {
      require(std::isfinite(q[i]),"nonfinite_ik");
      q[i]=previous[i]+std::remainder(q[i]-previous[i],2*M_PI);
      require(std::abs(q[i]-previous[i])<=c.branch,"ik_branch_jump");
      require(std::abs(q[i]-measured[i])<=c.tracking,"tracking_gap");
    }
    c.limits(q); const auto actual=forward(q);
    require((actual.translation()-target.translation()).norm()<=0.001,"ik_position_residual");
    require(Eigen::Quaterniond(actual.rotation()).angularDistance(Eigen::Quaterniond(target.rotation()))<=0.01,"ik_rotation_residual");
    return q;
  }
};
struct Chunk {
  uint64_t sequence=0;
  size_t clipped_grippers=0;
  double start=0,dt=0;
  std::vector<Pose> poses;
  std::vector<double> grippers;
  Pose anchor;
  double anchor_grip=0;
  double duration() const {return poses.size()*dt;}
  std::pair<Pose,double> sample(double t) const {
    require(!poses.empty()&&t>=start&&t<start+duration(),"chunk_expired");
    const double index=(t-start)/dt;
    const auto i=std::min(poses.size()-1,static_cast<size_t>(std::floor(index)));
    const double u=std::clamp(index-i,0.0,1.0);
    const auto next=std::min(i+1,poses.size()-1);
    return {blend(poses[i],poses[next],u),
      (1-u)*grippers[i]+u*grippers[next]};
  }
};
inline Chunk parse_chunk(const Json& j,const Config& c,const std::string& session,
                         uint64_t last_sequence,double observed,double now,
                         const Pose& anchor,double anchor_grip) {
  require(j.at("schema")=="arx_vla_chunk_v1","chunk_schema");
  require(j.at("session_id")==session,"wrong_session");
  require(j.at("semantics")=="absolute_xyz_rotvec","relative_or_unnormalized_action");
  require(j.at("sequence").is_number_unsigned(),"invalid_sequence");
  Chunk chunk; chunk.sequence=j.at("sequence").get<uint64_t>();
  require(chunk.sequence>last_sequence,"out_of_order_chunk");
  require(now>=observed && now-observed<=c.observation_age,"stale_observation");
  chunk.start=observed; chunk.dt=j.at("dt_s").get<double>();
  require(std::isfinite(chunk.dt)&&chunk.dt>=0.01&&chunk.dt<=0.1,"invalid_action_dt");
  const auto& eef=j.at("eef");const auto& grip=j.at("gripper");
  require(eef.is_array()&&eef.size()>0&&eef.size()<=40&&grip.is_array()&&grip.size()==eef.size(),"invalid_chunk_shape");
  const std::string frame=j.at("frame");
  require(frame=="model_mount_link6"||frame=="sdk_base_tool","invalid_frame");
  if(frame=="model_mount_link6") require(c.frames_verified,"model_frames_unverified");
  const bool move_gripper=j.at("control_gripper").get<bool>();
  const std::string grip_units=j.value("gripper_units",std::string("single_finger_m"));
  require(grip_units=="single_finger_m"||grip_units=="sdk_command","invalid_gripper_units");
  if(move_gripper&&grip_units=="single_finger_m") require(c.gripper_verified,"gripper_calibration_unverified");
  if(grip_units=="sdk_command") require(frame=="sdk_base_tool","sdk_gripper_requires_sdk_frame");
  chunk.anchor=anchor;chunk.anchor_grip=anchor_grip;
  for(size_t i=0;i<eef.size();++i) {
    Pose target=pose(numbers<6>(eef[i])); if(frame=="model_mount_link6") target=c.to_sdk(target);
    require((target.translation()-anchor.translation()).norm()<=c.translation,"chunk_translation_range");
    require(Eigen::Quaterniond(anchor.rotation()).angularDistance(Eigen::Quaterniond(target.rotation()))<=c.rotation,"chunk_rotation_range");
    const double g=grip[i].get<double>(); require(std::isfinite(g),"nonfinite_gripper");
    double command=anchor_grip;
    if(move_gripper) {
      if(grip_units=="sdk_command") {require(g>=0&&g<=5,"sdk_gripper_out_of_range");command=g;}
      else {command=c.grip_command(g);if(g<0||g>c.finger_max)++chunk.clipped_grippers;}
    }
    chunk.poses.push_back(target); chunk.grippers.push_back(command);
  }
  require(now<chunk.start+chunk.duration(),"chunk_expired");
  return chunk;
}
} // namespace vla
