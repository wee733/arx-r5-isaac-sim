#include <arx_vla_control/core.hpp>
#include <iostream>
using namespace vla;
void check(bool ok){require(ok,"test_failed");}
template<class F> void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught);}
int main(){
  Config c(DEFAULT_CONFIG);Kinematics kin;
  c.frames_verified=false;c.gripper_verified=false;
  const Q q{0,0.5,0.8,0,0.4,0};const auto p=kin.forward(q);
  auto ik=kin.inverse(p,q,q,c);check((kin.forward(ik).matrix()-p.matrix()).norm()<0.002);
  // Noncommuting base/tool transforms and a near-pi rotation must round-trip.
  c.base=pose({0.1,-0.2,0.3,0.1,0.2,0.3});c.tool=pose({0.02,0,0,0.4,0,0});
  auto t=pose({0.3,0.1,0.2,0,3.141592-1e-7,0});
  check((c.to_model(c.to_sdk(t)).matrix()-t.matrix()).norm()<1e-9);
  check((pose(vector(t)).matrix()-t.matrix()).norm()<1e-8);
  const auto a=vector(p);Json j={{"schema","arx_vla_chunk_v1"},{"session_id","test"},
    {"semantics","absolute_xyz_rotvec"},{"sequence",1},{"dt_s",1.0/30},
    {"frame","sdk_base_tool"},{"control_gripper",false},
    {"eef",Json::array({a,a,a,a})},{"gripper",{0.01,0.01,0.01,0.01}}};
  auto parse=[&](const Json& x){return parse_chunk(Json::parse(x.dump()),c,"test",0,10,10.02,p,2.0);};
  const auto chunk=parse(j);check(std::abs(chunk.duration()-4.0/30)<1e-9);
  check((chunk.sample(10.05).first.matrix()-p.matrix()).norm()<1e-9);
  check(chunk.sample(10.05).second==2.0);
  rejects([&]{chunk.sample(10.14);});
  auto bad=j;bad["frame"]="model_mount_link6";rejects([&]{parse(bad);});
  bad=j;bad["control_gripper"]=true;rejects([&]{parse(bad);});
  bad=j;bad["eef"][0][0]=a[0]+0.5;rejects([&]{parse(bad);});
  bad=j;bad["dt_s"]=-1;rejects([&]{parse(bad);});
  bad=j;bad["sequence"]=0;rejects([&]{parse(bad);});
  bad=j;bad["sequence"]=-1;rejects([&]{parse(bad);});
  bad=j;bad["session_id"]="old";rejects([&]{parse(bad);});
  bad=j;bad["semantics"]="relative";rejects([&]{parse(bad);});
  bad=j;bad["eef"][0][0]=nullptr;rejects([&]{parse(bad);});
  rejects([&]{parse_chunk(j,c,"test",0,9,10,p,2);});
  rejects([&]{c.grip_command(0.05);});
  check(c.grip_command(0.044)==5);
  check(c.grip_command(0.04)==5);check(c.grip_unit(c.feedback_open)==1);
  for(const auto& point:c.grip_response)check(std::abs(c.grip_hold(point[0])-point[1])<1e-10);
  bad=j;bad["control_gripper"]=true;bad["gripper_units"]="sdk_command";
  check(parse(bad).grippers[0]==0.01);
  bad["gripper"][0]=6;rejects([&]{parse(bad);});
  // Verify actual time interpolation (not one target per execution tick).
  Chunk time;time.start=1;time.dt=0.1;time.anchor=Pose::Identity();time.anchor_grip=0;
  time.poses={pose({0.01,0,0,0,0,0.2}),pose({0.02,0,0,0,0,0.4})};time.grippers={1,2};
  auto halfway=time.sample(1.05);check(std::abs(halfway.first.translation().x()-0.015)<1e-10);
  check(std::abs(vector(halfway.first)[5]-0.3)<1e-10);
  check(std::abs(time.sample(1.15).second-2)<1e-10);
  // Replanning an unchanged absolute target must not interpolate backwards
  // from delayed measured feedback at the beginning of every new chunk.
  check((chunk.sample(10.0001).first.matrix()-p.matrix()).norm()<1e-9);
  JointServo servo;servo.reset(q);c.tracking_ki=0.6;c.correction_limit=0.02;
  Q actual=q,target=q;target[0]+=0.015;Q previous=q;
  for(int i=0;i<1000;++i){
    auto output=servo.step(target,actual,.02,c);
    for(size_t k=0;k<6;++k)check(std::abs(output[k]-previous[k])<=c.speed*.02+1e-10);
    actual=output;actual[0]-=.006;previous=output;
    check(std::abs(servo.integral[0])<=.02);
  }
  check(std::abs(actual[0]-target[0])<.001);
  servo.reset(actual);check(servo.integral[0]==0);
  std::cout<<"PASS: native IK, SE3 transforms, physical units, timing, stale/session/shape/range rejection\n";
}
