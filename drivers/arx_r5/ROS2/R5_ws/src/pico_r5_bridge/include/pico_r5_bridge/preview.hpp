#pragma once
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pico_r5 {
using Json = nlohmann::json;
struct Pose {
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};
struct Sample {
  Pose pose;
  Eigen::Vector2d left_stick = Eigen::Vector2d::Zero();
  Eigen::Vector2d right_stick = Eigen::Vector2d::Zero();
  double grip = 0, trigger = 0;
  uint64_t stamp = 0;
  bool focus = false;
};

// The installed SDK delivers an outer JSON object with a JSON-encoded value string.
// Non-tracking messages must be filtered by the caller before parsing.
inline Sample parse(const Json &outer, const std::string &hand, bool require_sticks = false) {
  const auto &value = outer.at("value");
  const Json body = value.is_string() ? Json::parse(value.get<std::string>()) : value;
  if (body.at("Input").get<int>() != 1) throw std::runtime_error("not_controller_input");
  Sample s;
  s.focus = body.at("appState").at("focus").get<bool>();
  const auto &stamp = body.at("timeStampNs");
  if (!stamp.is_number_integer() || stamp.get<double>() <= 0)
    throw std::runtime_error("invalid_source_timestamp");
  s.stamp = stamp.get<uint64_t>();
  const auto &c = body.at("Controller").at(hand);
  if (require_sticks) {
    const auto &left=body.at("Controller").at("left");
    const auto &right=body.at("Controller").at("right");
    s.left_stick={left.at("axisX").get<double>(),left.at("axisY").get<double>()};
    s.right_stick={right.at("axisX").get<double>(),right.at("axisY").get<double>()};
    if (!s.left_stick.allFinite() || !s.right_stick.allFinite() ||
        s.left_stick.cwiseAbs().maxCoeff()>1 || s.right_stick.cwiseAbs().maxCoeff()>1)
      throw std::runtime_error("invalid_sticks");
  }
  s.grip = c.at("grip").get<double>();
  s.trigger = c.at("trigger").get<double>();
  if (!std::isfinite(s.grip) || !std::isfinite(s.trigger) ||
      s.grip < 0 || s.grip > 1 || s.trigger < 0 || s.trigger > 1)
    throw std::runtime_error("invalid_buttons");
  std::string raw = c.at("pose").get<std::string>();
  std::replace(raw.begin(), raw.end(), ',', ' ');
  std::istringstream input(raw);
  double v[7];
  for (double &x : v) {
    if (!(input >> x) || !std::isfinite(x)) throw std::runtime_error("invalid_pose");
  }
  std::string extra;
  if (input >> extra) throw std::runtime_error("invalid_pose_length");
  s.pose.p = Eigen::Vector3d(v[0], v[1], v[2]);
  s.pose.q = Eigen::Quaterniond(v[6], v[3], v[4], v[5]);
  if (std::abs(s.pose.q.norm() - 1.0) > 0.1) throw std::runtime_error("invalid_quaternion");
  s.pose.q.normalize();
  return s;
}

struct Settings {
  double scale = 0.3, timeout = 0.2;
  double rotation_scale = 1.0;
  double stick_deadzone = 0.15;
  double linear_speed = 0.05, angular_speed = 0.3;
  double max_offset = 0.1, max_rotation = 0.5;
  double grip_on = 0.6, grip_off = 0.3;
  double jump_distance = 0.2, jump_rotation = 1.0;
  // Provisional OpenXR right/up/back -> robot forward/left/up. Verify physically.
  Eigen::Matrix3d axes = (Eigen::Matrix3d() << 0,0,-1, -1,0,0, 0,1,0).finished();
};

class Preview {
 public:
  explicit Preview(Settings settings = {}) : settings_(settings) {}
  const Pose &target() const { return target_; }
  const Sample &sample() const { return sample_; }
  bool active() const { return active_; }
  const std::string &reason() const { return reason_; }
  uint64_t frames() const { return frames_; }
  uint64_t rejected() const { return rejected_; }
  double age(double now) const { return have_ ? std::max(0.0, now - received_) : -1; }

  void fault(const std::string &reason) {
    active_ = false;
    ready_ = false;
    reason_ = reason;
    ++rejected_;
  }

  void ingest(const Sample &s, double now) {
    if (have_ && s.stamp <= sample_.stamp) {
      fault("source_timestamp_not_advancing");
      // Rebase only after the restart has forced a release-to-rearm cycle.
      sample_.stamp = s.stamp;
      return;
    }
    const bool gap = have_ && now - received_ > settings_.timeout;
    const bool jump = have_ && ((s.pose.p - sample_.pose.p).norm() > settings_.jump_distance ||
      s.pose.q.angularDistance(sample_.pose.q) > settings_.jump_rotation);
    sample_ = s;
    received_ = now;
    have_ = true;
    ++frames_;
    if (!s.focus) { fault("pico_not_focused"); return; }
    if (gap || jump) { fault(gap ? "input_gap" : "pose_jump"); return; }
    if (s.grip <= settings_.grip_off) {
      active_ = false;
      ready_ = true;
      reason_ = "released";
    } else if (s.grip >= settings_.grip_on && !active_) {
      if (!ready_) { reason_ = "release_grip_to_arm"; return; }
      input_anchor_ = s.pose;
      target_anchor_ = target_;
      active_ = true;
      ready_ = false;
      reason_ = "following_preview";
    }
  }

  void step(double now) {
    const double dt = last_step_ < 0 ? 0 : std::clamp(now - last_step_, 0.0, 0.04);
    if (last_step_ >= 0 && now - last_step_ > settings_.timeout) fault("preview_loop_stall");
    last_step_ = now;
    if (!have_) return;
    if (now - received_ > settings_.timeout) {
      if (reason_ != "input_timeout") fault("input_timeout");
      return;
    }
    if (!active_) return;
    Eigen::Vector3d offset = settings_.scale * settings_.axes * (sample_.pose.p - input_anchor_.p);
    Eigen::Quaterniond delta(settings_.axes * (sample_.pose.q * input_anchor_.q.conjugate()).toRotationMatrix()
                             * settings_.axes.transpose());
    const Eigen::AngleAxisd raw_rotation(delta);
    const double mapped_angle = raw_rotation.angle() * settings_.rotation_scale;
    if (offset.norm() > settings_.max_offset || mapped_angle > settings_.max_rotation) {
      fault("preview_workspace_exceeded");
      return;
    }
    const Eigen::Vector3d wanted = target_anchor_.p + offset;
    delta = Eigen::Quaterniond(Eigen::AngleAxisd(mapped_angle, raw_rotation.axis()));
    const Eigen::Vector3d change = wanted - target_.p;
    const double distance = change.norm();
    target_.p += change * (distance > 0 ? std::min(1.0, settings_.linear_speed * dt / distance) : 0);
    const Eigen::Quaterniond rotation = (delta * target_anchor_.q).normalized();
    const double angle = target_.q.angularDistance(rotation);
    target_.q = target_.q.slerp(angle > 0 ? std::min(1.0, settings_.angular_speed * dt / angle) : 1,
                              rotation).normalized();
  }

 private:
  Settings settings_;
  Pose target_, input_anchor_, target_anchor_;
  Sample sample_;
  double received_ = 0, last_step_ = -1;
  bool have_ = false, active_ = false, ready_ = false;
  uint64_t frames_ = 0, rejected_ = 0;
  std::string reason_ = "waiting_for_tracking";
};
}  // namespace pico_r5
