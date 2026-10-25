#pragma once
#include <pico_r5_bridge/preview.hpp>

namespace pico_r5 {
// Joystick velocity integration is independent of the orientation clutch.
class JoystickPreview {
 public:
  explicit JoystickPreview(Settings settings = {}) : settings_(settings) {}
  const Pose &target() const { return target_; }
  const Sample &sample() const { return sample_; }
  bool active() const { return ready_ && (stick_active_ || rotation_active_); }
  bool stick_active() const { return ready_ && stick_active_; }
  bool rotation_active() const { return ready_ && rotation_active_; }
  const std::string &reason() const { return reason_; }
  uint64_t frames() const { return frames_; }
  uint64_t rejected() const { return rejected_; }
  double age(double now) const { return have_ ? std::max(0.0,now-received_) : -1; }
  void fault(const std::string &why) {
    ready_=false; stick_active_=false; rotation_active_=false;
    velocity_.setZero(); reason_=why; ++rejected_;
  }
  void ingest(const Sample &s,double now) {
    if (have_ && s.stamp<=sample_.stamp) {
      fault("source_timestamp_not_advancing"); sample_.stamp=s.stamp; return;
    }
    const bool gap=have_ && now-received_>settings_.timeout;
    const bool jump=have_ && s.pose.q.angularDistance(sample_.pose.q)>settings_.jump_rotation;
    sample_=s; have_=true; received_=now; ++frames_;
    if (!s.focus) { fault("pico_not_focused"); return; }
    if (gap || jump) { fault(gap?"input_gap":"rotation_jump"); return; }
    if (!s.left_stick.allFinite() || !s.right_stick.allFinite() ||
        s.left_stick.cwiseAbs().maxCoeff()>1 || s.right_stick.cwiseAbs().maxCoeff()>1) {
      fault("invalid_sticks"); return;
    }
    const bool centered=s.left_stick.norm()<=settings_.stick_deadzone &&
                          s.right_stick.norm()<=settings_.stick_deadzone;
    if (!ready_) {
      if (!centered || s.grip>settings_.grip_off) {
        reason_="center_sticks_and_release_grip"; return;
      }
      ready_=true;
    }
    auto axis=[this](double x) {
      return std::copysign(std::max(0.0,std::abs(x)-settings_.stick_deadzone)/
                           (1-settings_.stick_deadzone),x);
    };
    // Forward +X, left +Y, up +Z in robot base frame. Right-stick X unused.
    velocity_=Eigen::Vector3d(axis(s.left_stick.y()),-axis(s.left_stick.x()),axis(s.right_stick.y()));
    if (velocity_.norm()>1) velocity_.normalize();
    const bool moving=velocity_.norm()>0;
    if (moving && !stick_active_) translation_anchor_=target_.p;
    stick_active_=moving;
    if (s.grip<=settings_.grip_off) rotation_active_=false;
    else if (s.grip>=settings_.grip_on && !rotation_active_) {
      rotation_input_=s.pose.q; rotation_target_=target_.q; rotation_active_=true;
    }
    reason_=active()?"joystick_following":"joystick_idle";
  }
  void step(double now) {
    const double dt=last_step_<0?0:std::clamp(now-last_step_,0.0,0.04);
    if (last_step_>=0 && now-last_step_>settings_.timeout) fault("preview_loop_stall");
    last_step_=now;
    if (!have_) return;
    if (now-received_>settings_.timeout) {
      if (reason_!="input_timeout") fault("input_timeout");
      return;
    }
    if (!ready_) return;
    const Eigen::Vector3d next=target_.p+velocity_*settings_.linear_speed*dt;
    if ((next-translation_anchor_).norm()>settings_.max_offset && stick_active_) {
      fault("joystick_translation_range"); return;
    }
    Eigen::Quaterniond next_q=target_.q;
    if (rotation_active_) {
      Eigen::Quaterniond delta(settings_.axes*(sample_.pose.q*rotation_input_.conjugate()).toRotationMatrix()*settings_.axes.transpose());
      Eigen::AngleAxisd aa(delta);
      const double angle=aa.angle()*settings_.rotation_scale;
      if (angle>settings_.max_rotation) { fault("joystick_rotation_range"); return; }
      Eigen::Quaterniond wanted=Eigen::Quaterniond(Eigen::AngleAxisd(angle,aa.axis()))*rotation_target_;
      const double distance=target_.q.angularDistance(wanted);
      next_q=target_.q.slerp(distance>0?std::min(1.0,settings_.angular_speed*dt/distance):1,wanted).normalized();
    }
    target_.p=next; target_.q=next_q;
  }
 private:
  Settings settings_;
  Sample sample_;
  Pose target_;
  Eigen::Vector3d velocity_=Eigen::Vector3d::Zero(),translation_anchor_=Eigen::Vector3d::Zero();
  Eigen::Quaterniond rotation_input_=Eigen::Quaterniond::Identity(),rotation_target_=Eigen::Quaterniond::Identity();
  bool have_=false,ready_=false,stick_active_=false,rotation_active_=false;
  double received_=0,last_step_=-1;
  uint64_t frames_=0,rejected_=0;
  std::string reason_="waiting_for_tracking";
};

// Common node-facing API keeps the existing pose mode unchanged.
class InputPreview {
 public:
  InputPreview(Settings s={},bool joystick=false) : joystick_(joystick),pose_(s),stick_(s) {}
  const Pose &target() const { return joystick_?stick_.target():pose_.target(); }
  const Sample &sample() const { return joystick_?stick_.sample():pose_.sample(); }
  const std::string &reason() const { return joystick_?stick_.reason():pose_.reason(); }
  bool active() const { return joystick_?stick_.active():pose_.active(); }
  bool stick_active() const { return joystick_ && stick_.stick_active(); }
  bool rotation_active() const { return joystick_?stick_.rotation_active():pose_.active(); }
  uint64_t frames() const { return joystick_?stick_.frames():pose_.frames(); }
  uint64_t rejected() const { return joystick_?stick_.rejected():pose_.rejected(); }
  double age(double t) const { return joystick_?stick_.age(t):pose_.age(t); }
  void fault(const std::string &s) { if(joystick_) stick_.fault(s); else pose_.fault(s); }
  void ingest(const Sample &s,double t) { if(joystick_) stick_.ingest(s,t); else pose_.ingest(s,t); }
  void step(double t) { if(joystick_) stick_.step(t); else pose_.step(t); }
 private:
  bool joystick_;
  Preview pose_;
  JoystickPreview stick_;
};
}
