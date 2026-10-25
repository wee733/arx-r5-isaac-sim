#include <cstdint>
#include <cstring>
#include <PXREARobotSDK.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <pico_r5_bridge/joystick.hpp>
#include <chrono>
#include <mutex>

namespace {
double monotonic() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}

class PicoPreview : public rclcpp::Node {
 public:
  PicoPreview() : Node("pico_preview") {
    hand_ = declare_parameter<std::string>("hand", "right");
    if (hand_ != "right" && hand_ != "left") throw std::runtime_error("hand must be right or left");
    device_ = declare_parameter<std::string>("device_id", "");
    pico_r5::Settings s;
    const auto mode=declare_parameter<std::string>("translation_mode","pose");
    if (mode!="pose" && mode!="joystick") throw std::runtime_error("invalid translation_mode");
    joystick_=mode=="joystick";
    if (joystick_ && hand_!="right") throw std::runtime_error("joystick mode requires right orientation hand");
    s.stick_deadzone=declare_parameter<double>("joystick_deadzone",0.15);
    if (!std::isfinite(s.stick_deadzone) || s.stick_deadzone<0 || s.stick_deadzone>=1)
      throw std::runtime_error("invalid joystick deadzone");
    const auto axes=declare_parameter<std::vector<int64_t>>("rotation_axis_map",{-3,-1,2});
    if (axes.size()!=3) throw std::runtime_error("rotation_axis_map needs 3 signed axes");
    Eigen::Matrix3d rotation_axes=Eigen::Matrix3d::Zero();
    for (size_t i=0;i<3;++i) {
      if (axes[i]==0 || std::abs(axes[i])>3) throw std::runtime_error("invalid rotation axis");
      rotation_axes(i,std::abs(axes[i])-1)=axes[i]>0?1:-1;
    }
    if (std::abs(rotation_axes.determinant())<0.9) throw std::runtime_error("rotation axes must be unique");
    // Only joystick mode uses configurable rotation axes; pose mapping stays legacy.
    if (joystick_) s.axes=rotation_axes;
    s.scale = declare_parameter<double>("position_scale", 0.3);
    s.rotation_scale = declare_parameter<double>("rotation_scale", 1.0);
    s.timeout = declare_parameter<double>("timeout_s", 0.2);
    s.linear_speed = declare_parameter<double>("linear_speed_m_s", 0.05);
    s.angular_speed = declare_parameter<double>("angular_speed_rad_s", 0.3);
    s.max_offset = declare_parameter<double>("max_clutch_translation_m", 0.1);
    s.max_rotation = declare_parameter<double>("max_clutch_rotation_rad", 0.5);
    for (double v : {s.scale, s.rotation_scale, s.timeout, s.linear_speed, s.angular_speed, s.max_offset, s.max_rotation})
      if (!std::isfinite(v) || v <= 0) throw std::runtime_error("preview limits must be positive and finite");
    preview_ = pico_r5::InputPreview(s,joystick_);
    raw_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/pico_preview/controller_pose", rclcpp::SensorDataQoS().keep_last(1));
    target_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/pico_preview/target_pose", rclcpp::SensorDataQoS().keep_last(1));
    status_pub_ = create_publisher<std_msgs::msg::String>("/pico_preview/status", rclcpp::QoS(1));
    timer_ = create_wall_timer(std::chrono::milliseconds(20), [this] { tick(); });
    RCLCPP_WARN(get_logger(), "PREVIEW ONLY: no CAN, no R5 commands, no IK. Target origin is synthetic. Release grip once, then hold to preview.");
    sdk_started_ = true;
    const int result = PXREAInit(this, &PicoPreview::callback, PXREAFullMask);
    RCLCPP_INFO(get_logger(), "PXREAInit returned %d; waiting for tracking callbacks", result);
  }

  ~PicoPreview() override {
    timer_->cancel();
    if (sdk_started_) PXREADeinit();
  }

 private:
  static void callback(void *ctx, PXREAClientCallbackType type, int, void *data) {
    auto *self = static_cast<PicoPreview *>(ctx);
    // Never allow exceptions to cross the SDK's C callback boundary.
    try { self->receive(type, data); }
    catch (...) {
      std::lock_guard<std::mutex> lock(self->mutex_);
      self->preview_.fault("callback_error");
    }
  }

  void receive(PXREAClientCallbackType type, void *data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (type == PXREAServerDisconnect || type == PXREADeviceMissing) {
      preview_.fault("sdk_or_device_disconnected");
      return;
    }
    if (type != PXREADeviceStateJson || !data) return;
    const auto &message = *static_cast<const PXREADevStateJson *>(data);
    const std::string device(message.devID, strnlen(message.devID, sizeof(message.devID)));
    if (!device_.empty() && device != device_) return;
    try {
      const auto outer = pico_r5::Json::parse(message.stateJson,
        message.stateJson + strnlen(message.stateJson, sizeof(message.stateJson)));
      if (outer.value("functionName", "") != "Tracking") return;
      auto sample = pico_r5::parse(outer, hand_,joystick_);
      if (device_.empty()) device_ = device;
      preview_.ingest(sample, monotonic());
    } catch (const std::exception &) {
      preview_.fault("invalid_tracking_packet");
    }
  }

  geometry_msgs::msg::PoseStamped pose_message(const pico_r5::Pose &p, const char *frame) {
    geometry_msgs::msg::PoseStamped m;
    // ROS header is local publication time, not Pico capture time. See status.source_timestamp_ns.
    m.header.stamp = now();
    m.header.frame_id = frame;
    m.pose.position.x = p.p.x(); m.pose.position.y = p.p.y(); m.pose.position.z = p.p.z();
    m.pose.orientation.x = p.q.x(); m.pose.orientation.y = p.q.y();
    m.pose.orientation.z = p.q.z(); m.pose.orientation.w = p.q.w();
    return m;
  }

  void tick() {
    std::lock_guard<std::mutex> lock(mutex_);
    const double t = monotonic();
    preview_.step(t);
    if (preview_.frames() != last_raw_frame_) {
      raw_pub_->publish(pose_message(preview_.sample().pose, "pico_tracking_unverified"));
      last_raw_frame_ = preview_.frames();
    }
    target_pub_->publish(pose_message(preview_.target(), "r5_preview_relative_origin"));
    const auto &sample = preview_.sample();
    const auto &p = preview_.target().p;
    const auto &q = preview_.target().q;
    pico_r5::Json report = {
      {"preview_only", true}, {"hand", hand_}, {"device_id", device_},
      {"state", preview_.reason()}, {"active", preview_.active()},
      {"translation_mode",joystick_?"joystick":"pose"},
      {"stick_active",preview_.stick_active()}, {"rotation_active",preview_.rotation_active()},
      {"left_stick",{sample.left_stick.x(),sample.left_stick.y()}},
      {"right_stick",{sample.right_stick.x(),sample.right_stick.y()}},
      {"frames", preview_.frames()}, {"rejected", preview_.rejected()},
      {"receive_age_ms", preview_.age(t) * 1000}, {"source_timestamp_ns", sample.stamp},
      {"focus", sample.focus}, {"grip", sample.grip}, {"trigger", sample.trigger},
      {"target_xyz", {p.x(), p.y(), p.z()}}, {"target_xyzw", {q.x(), q.y(), q.z(), q.w()}}
    };
    std_msgs::msg::String status;
    status.data = report.dump();
    status_pub_->publish(status);
    if (t - last_status_ >= 1.0 || preview_.reason() != last_reason_) {
      last_status_ = t;
      last_reason_ = preview_.reason();
      RCLCPP_INFO(get_logger(), "%s", status.data.c_str());
    }
  }

  std::mutex mutex_;
  pico_r5::InputPreview preview_;
  std::string hand_, device_, last_reason_;
  bool sdk_started_ = false;
  bool joystick_ = false;
  uint64_t last_raw_frame_ = 0;
  double last_status_ = 0;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr raw_pub_, target_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try { rclcpp::spin(std::make_shared<PicoPreview>()); }
  catch (const std::exception &e) {
    RCLCPP_ERROR(rclcpp::get_logger("pico_preview"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
