// Standalone offline probe: only constructs the existing kinematics solver, never a CAN interface.
#include <kinematic_solver.hpp>
#include <arx_r5_src/interfaces/InterfacesThread.hpp>
#include <arx_r5_src/interfaces/InterfaceTools.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
  arx::KinematicSolver solver;
  arx::r5::InterfacesTools controller_tools(0);
  std::array<std::array<double,6>,3> cases{{
    {{0,-0.5,0.8,0,0.4,0}}, {{0.1,0.2,-0.3,0.1,0.2,-0.1}}, {{0,0,0,0,0,0}}
  }};
  int passed = 0;
  for (auto joints : cases) {
    double pose[6], answer[6], reconstructed[6];
    std::fill_n(pose,6,std::numeric_limits<double>::quiet_NaN());
    std::fill_n(answer,6,std::numeric_limits<double>::quiet_NaN());
    solver.computeForwardKinematics(joints.data(),pose);
    const auto controller_pose = controller_tools.ForwardKinematicsRpy(std::vector<double>(joints.begin(),joints.end()));
    solver.computeInverseKinematics(pose,answer);
    bool finite = true;
    for (int i=0;i<6;++i) finite = finite && std::isfinite(pose[i]) && std::isfinite(answer[i]);
    std::cout << "q:"; for (double v:joints) std::cout << ' ' << v;
    std::cout << " pose:"; for (double v:pose) std::cout << ' ' << v;
    std::cout << " ik:"; for (double v:answer) std::cout << ' ' << v;
    std::cout << " controller_fk:"; for (double v:controller_pose) std::cout << ' ' << v;
    if (!finite) { std::cout << " NONFINITE\n"; continue; }
    solver.computeForwardKinematics(answer,reconstructed);
    auto expected = arx::solve::Xyzrpy2Isometry(pose);
    auto actual = arx::solve::Xyzrpy2Isometry(reconstructed);
    const double position_error = (expected.translation()-actual.translation()).norm();
    const double rotation_error = Eigen::Quaterniond(expected.rotation()).angularDistance(Eigen::Quaterniond(actual.rotation()));
    std::cout << " position_error=" << position_error << " rotation_error=" << rotation_error << '\n';
    if (position_error < 0.001 && rotation_error < 0.01) ++passed;
  }
  std::cout << "FK-IK-FK passed " << passed << '/' << cases.size() << '\n';
  return passed == static_cast<int>(cases.size()) ? 0 : 1;
}
