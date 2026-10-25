#include <pico_r5_bridge/joystick.hpp>
#include <iostream>
using namespace pico_r5;
void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
Sample s(uint64_t n) { Sample a; a.focus=true; a.stamp=n; return a; }
int main() {
 try {
  Settings cfg; cfg.linear_speed=0.1; cfg.angular_speed=100;
  cfg.rotation_scale=0.5; cfg.max_rotation=0.5;
  JoystickPreview p(cfg);
  auto a=s(1); a.left_stick.y()=1; p.ingest(a,0); p.step(0);
  check(!p.active(),"startup deflected stick must not move");
  a=s(2); p.ingest(a,.02); p.step(.02);
  a=s(3); a.left_stick.y()=1; p.ingest(a,.04); p.step(.04);
  check(p.active() && !p.rotation_active(),"translation does not require grip");
  check(std::abs(p.target().p.x()-.002)<1e-9,"forward velocity integration");
  a=s(4); a.pose.p.x()=5; p.ingest(a,.06); p.step(.06);
  check(!p.active() && std::abs(p.target().p.x()-.002)<1e-9,"center stops; physical translation ignored");
  a=s(5); a.right_stick.y()=1; p.ingest(a,.08); p.step(.08);
  check(std::abs(p.target().p.z()-.002)<1e-9,"right stick up");
  a=s(6); a.left_stick.x()=1; p.ingest(a,.10); p.step(.10);
  check(std::abs(p.target().p.y()+.002)<1e-9,"left stick right maps negative Y");
  a=s(7); a.grip=1; p.ingest(a,.12); p.step(.12);
  a=s(8); a.grip=1; a.pose.q=Eigen::Quaterniond(Eigen::AngleAxisd(.2,Eigen::Vector3d::UnitY()));
  p.ingest(a,.14); p.step(.14);
  auto held=p.target().q;
  check(held.angularDistance(Eigen::Quaterniond(Eigen::AngleAxisd(.1,Eigen::Vector3d::UnitZ())))<1e-9,"yaw scale and mapping");
  a.stamp=9; a.grip=0; a.left_stick.y()=1; p.ingest(a,.16); p.step(.16);
  check(p.active() && !p.rotation_active() && held.angularDistance(p.target().q)<1e-9,"release holds orientation, translation continues");
  a.stamp=10; a.grip=1; p.ingest(a,.18); p.step(.18);
  check(held.angularDistance(p.target().q)<1e-9,"reclutch no angular jump");
  auto stopped=p.target().p; p.step(.5);
  check(!p.active() && p.target().p==stopped,"timeout freezes integrated target");
  a.stamp=11; p.ingest(a,.51); p.step(.51);
  check(!p.active(),"reconnect deflected sticks must not resume");
  a=s(12); p.ingest(a,.53); p.step(.53);
  a=s(13); a.left_stick.x()=.1; p.ingest(a,.55); p.step(.55);
  check(!p.active(),"deadzone");
  for(int axis=0;axis<3;++axis) {
    JoystickPreview r(cfg); auto b=s(1); r.ingest(b,0); r.step(0);
    b=s(2); b.grip=1; b.pose.q=Eigen::Quaterniond(Eigen::AngleAxisd(.7,Eigen::Vector3d::UnitZ()));
    r.ingest(b,.02); r.step(.02); auto anchor=b.pose.q;
    b.stamp=3; b.pose.q=Eigen::Quaterniond(Eigen::AngleAxisd(.2,Eigen::Vector3d::Unit(axis)))*anchor;
    r.ingest(b,.04); r.step(.04);
    const auto expected=Eigen::Quaterniond(Eigen::AngleAxisd(.1,cfg.axes.col(axis)));
    check(r.target().q.angularDistance(expected)<1e-9,"each rotation axis with nonidentity anchor");
  }
  Settings small=cfg; small.max_offset=.001;
  JoystickPreview limited(small); limited.ingest(s(1),0); limited.step(0);
  a=s(2); a.left_stick.y()=1; limited.ingest(a,.02); limited.step(.02);
  check(!limited.active() && limited.target().p.norm()==0,"range fault does not integrate");
  Json controller={{"axisX",0.2},{"axisY",-0.3},{"grip",0.0},{"trigger",0.0},{"pose","0,0,0,0,0,0,1"}};
  Json body={{"Input",1},{"appState",{{"focus",true}}},{"timeStampNs",123},
    {"Controller",{{"left",controller},{"right",controller}}}};
  auto parsed=parse(Json{{"value",body.dump()}},"right",true);
  check(parsed.left_stick.x()==.2 && parsed.right_stick.y()==-.3,"observed SDK axisX/Y fields");
  body["Controller"]["left"].erase("axisX"); bool failed=false;
  try { parse(Json{{"value",body}},"right",true); } catch(...) { failed=true; }
  check(failed,"missing left stick cannot silently become zero");
  std::cout<<"PASS joystick integration, center, deadzone, grip independence, timeout, axes, parser, range\n";
 } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
