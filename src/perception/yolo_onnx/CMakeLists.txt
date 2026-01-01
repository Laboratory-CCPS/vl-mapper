cmake_minimum_required(VERSION 3.8)
project(yolo_onnx)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vl_mapper_interface REQUIRED)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
find_library(ONNXRUNTIME_LIB onnxruntime)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "ONNX Runtime headers or library not found. Set ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIB")
endif()

add_executable(yolo_onnx_node src/yolo_onnx_node.cpp)
target_include_directories(yolo_onnx_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_INCLUDE_DIR}
)
target_compile_features(yolo_onnx_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_include_directories(yolo_onnx_node PUBLIC ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(yolo_onnx_node rclcpp sensor_msgs vision_msgs builtin_interfaces vl_mapper_interface)
target_link_libraries(yolo_onnx_node ${ONNXRUNTIME_LIB} ${OpenCV_LIBS})

install(TARGETS yolo_onnx_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
