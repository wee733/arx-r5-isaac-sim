#include <pico_r5_bridge/preview.hpp>
#include <iostream>
#include <stdexcept>

using namespace pico_r5;
void check(bool result, const char *message) { if (!result) throw std::runtime_error(message); }
Sample sample(uint64_t n, double grip = 0) {
  Sample s; s.stamp = n; s.grip = grip; s.focus = true; return s;
}

int main() {
  try {
    Json body = {{"Input",1}, {"appState",{{"focus",true}}}, {"timeStampNs",uint64_t(1789138190375224064ULL)},
      {"Controller",{{"right",{{"pose","1,2,3,0,0,0,1"},{"grip",0.7},{"trigger",0.2}}}}}};
    Json outer = {{"functionName","Tracking"},{"value",body.dump()}};
    auto parsed = parse(outer, "right");
    check(parsed.pose.p == Eigen::Vector3d(1,2,3) && parsed.stamp == 1789138190375224064ULL, "SDK JSON/uint64 parsing");
    body["Controller"]["right"]["pose"] = "1,2,3,0,0,0,0";
    bool bad = false;
    try { parse(Json{{"value",body}}, "right"); } catch (...) { bad = true; }
    check(bad, "zero quaternion must fail");

    Preview p;
    p.ingest(sample(1,1),0); p.step(0);
    check(!p.active(), "startup held grip must not arm");
    p.ingest(sample(2),0.01);
    p.ingest(sample(3,1),0.02); p.step(0.02);
    check(p.active() && p.target().p.norm() == 0, "engage without jump");
    Sample moved = sample(4,1); moved.pose.p.z() = -0.1;
    p.ingest(moved,0.04); p.step(0.04);
    check(std::abs(p.target().p.x()-0.001) < 1e-9, "axis mapping and linear speed bound");
    check(p.target().p.y() == 0 && p.target().p.z() == 0, "other axes unchanged");
    auto hold = p.target();
    p.step(0.3);
    check(!p.active() && p.target().p == hold.p, "timeout cancels old target");
    moved.stamp = 5; p.ingest(moved,0.31);
    moved.stamp = 6; p.ingest(moved,0.32);
    check(!p.active(), "reconnect held grip must not rearm");
    moved.stamp = 7; moved.grip = 0; p.ingest(moved,0.33);
    moved.stamp = 8; moved.grip = 1; p.ingest(moved,0.34); p.step(0.34);
    check(p.active() && p.target().p == hold.p, "reclutch preserves target");
    p.ingest(moved,0.35);
    check(!p.active(), "duplicate timestamp must disarm");

    Preview rotation;
    rotation.ingest(sample(1),0); rotation.step(0);
    rotation.ingest(sample(2,1),0.02); rotation.step(0.02);
    Sample turned = sample(3,1);
    turned.pose.q = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
    rotation.ingest(turned,0.04); rotation.step(0.04);
    Eigen::AngleAxisd aa(rotation.target().q);
    check(std::abs(aa.angle()-0.006)<1e-9 && aa.axis().z()>0.999, "rotation axes and speed limit");
    turned.stamp = 4; turned.focus = false; rotation.ingest(turned,0.05);
    check(!rotation.active(), "focus loss must disarm");

    Settings scaled;
    scaled.scale = 1.0; scaled.rotation_scale = 0.5;
    scaled.linear_speed = 100; scaled.angular_speed = 100;
    scaled.max_rotation = 0.15;
    Preview mapping(scaled);
    mapping.ingest(sample(1),0); mapping.step(0);
    mapping.ingest(sample(2,1),0.02); mapping.step(0.02);
    auto mapped = sample(3,1); mapped.pose.p.z() = -0.04;
    mapped.pose.q = Eigen::Quaterniond(Eigen::AngleAxisd(0.2,Eigen::Vector3d::UnitY()));
    mapping.ingest(mapped,0.04); mapping.step(0.04);
    check(mapping.active(), "rotation limit must apply after scaling");
    check(std::abs(mapping.target().p.x()-0.04)<1e-9, "1:1 translation mapping");
    check(std::abs(Eigen::AngleAxisd(mapping.target().q).angle()-0.1)<1e-9, "half rotation mapping");
    mapped.stamp=4;
    mapped.pose.q = Eigen::Quaterniond(Eigen::AngleAxisd(0.4,Eigen::Vector3d::UnitY()));
    mapping.ingest(mapped,0.06); mapping.step(0.06);
    check(!mapping.active(), "mapped rotation range must disarm");

    Preview jump;
    jump.ingest(sample(1),0); jump.ingest(sample(2,1),0.01);
    auto teleported = sample(3,1); teleported.pose.p.x() = 1;
    jump.ingest(teleported,0.02);
    check(!jump.active() && jump.reason() == "pose_jump", "relocalization must disarm");
    std::cout << "PASS: parser, arming, re-clutch, translation/rotation limits, timeout, timestamps, focus, pose jump\n";
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
