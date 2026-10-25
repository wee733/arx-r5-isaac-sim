#!/bin/bash
source ~/.bashrc
workspace=$(pwd)
ROS_DISTRO="${ROS_DISTRO:-jazzy}"

if [ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    echo "未找到 /opt/ros/${ROS_DISTRO}/setup.bash，请检查 ROS_DISTRO。"
    exit 1
fi

#x7s
gnome-terminal -t "L" -x  bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; cd ${workspace}; cd ../..; cd ROS2/R5_ws; rm -rf build install log .catkin_workspace src/CMakeLists.txt; colcon build; exec bash;"
sleep 0.5

#VR
gnome-terminal -t "vr" -x  bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; cd ${workspace}; cd ../..; cd ARX_VR_SDK/ROS2; rm -rf build install log .catkin_workspace; ./port.sh; exec bash;"
