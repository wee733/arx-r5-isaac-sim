cmake_minimum_required(VERSION 3.8)
project(arx_r5_controller)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(arm_control REQUIRED)
find_package(arx5_arm_msg REQUIRED)

find_package(tf2_msgs REQUIRED)
find_package(tf2 REQUIRED)

include_directories(include)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Target architecture is ARM")
    set(DEPEND_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/arx_r5_src/libarx_r5_src-arm64.so)
else ()
    set(DEPEND_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/arx_r5_src/libarx_r5_src.so)
endif ()

# 添加可执行文件
add_executable(R5Controller src/R5Controller.cpp)
target_include_directories(R5Controller PUBLIC lib/arx_hardware_interface/include lib/arx_r5_src/include)
target_link_libraries(R5Controller ${DEPEND_LIB})
ament_target_dependencies(R5Controller rclcpp ament_index_cpp std_msgs arx5_arm_msg arm_control tf2 tf2_msgs)

# 添加可执行文件
add_executable(R5Visualization src/R5Visualization.cpp)
target_include_directories(R5Visualization PUBLIC lib/arx_hardware_interface/include lib/arx_r5_src/include)
target_link_libraries(R5Visualization ${DEPEND_LIB})
ament_target_dependencies(R5Visualization rclcpp std_msgs arx5_arm_msg arm_control tf2 tf2_msgs)

add_executable(KeyBoard src/KeyBoard.cpp)
ament_target_dependencies(KeyBoard rclcpp std_msgs arx5_arm_msg)

install(TARGETS
    R5Controller
    R5Visualization
    KeyBoard
  DESTINATION lib/${PROJECT_NAME})

install(FILES ${DEPEND_LIB}
  DESTINATION lib/
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py")

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml")

install(FILES X5liteaa0.urdf
        DESTINATION share/${PROJECT_NAME}
)

install(FILES R5_master.urdf
        DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()