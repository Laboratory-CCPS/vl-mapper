cmake_minimum_required(VERSION 3.16)
project(vl_mapper_interface)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.27)
    cmake_policy(SET CMP0148 OLD)
endif()

# --- ROS DEPENDENCIES ---
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# --- INTERFACES ---
set(MSG_FILES
  "msg/Labels.msg"
)

set(ACTION_FILES
  "action/Detect.action"
)

set(SRV_FILES
  "srv/DetectAtStamp.srv"
  "srv/TokenizeLabels.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES}
  ${ACTION_FILES}
  ${SRV_FILES}
  DEPENDENCIES std_msgs vision_msgs sensor_msgs
)

ament_package()
