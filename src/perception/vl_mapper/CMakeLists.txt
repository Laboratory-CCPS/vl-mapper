cmake_minimum_required(VERSION 3.8)
project(vl_mapper)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenMP REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(vl_mapper_interface REQUIRED)

add_executable(vl_mapper_node 
    src/vl_mapper_node.cpp
)

target_include_directories(vl_mapper_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(vl_mapper_node PRIVATE
  -O3
  -march=native
)
target_compile_features(vl_mapper_node PUBLIC 
    c_std_99 
    cxx_std_17)
ament_target_dependencies(
    vl_mapper_node
    "rclcpp"
    "sensor_msgs"
    "vision_msgs"
    "nav_msgs"
    "message_filters"
    "cv_bridge"
    "pcl_ros"
    "tf2_eigen"
    "vl_mapper_interface"
)
target_link_libraries(vl_mapper_node 
    ${OpenCV_LIBS}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)


install(TARGETS vl_mapper_node
    DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
endif()

ament_package()
