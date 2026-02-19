cmake_minimum_required(VERSION 3.16)
project(mesh_vo_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Add the include directory for the mesh_vo library
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/src
    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/HLSLinearAlgebra/include
    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty/Sophus
    ${OpenCV_INCLUDE_DIRS}
)

# --- (Optional) Build your VO core as a library ---
# If VisualOdometryThreaded has source files, list them here and link them to the node.
# add_library(mesh_vo_core
#   src/visualOdometryThreaded.cpp
#   src/whatever_else.cpp
# )
# target_include_directories(mesh_vo_core PUBLIC include)
# ament_target_dependencies(mesh_vo_core
#   rclcpp
#   sensor_msgs
#   geometry_msgs
#   visualization_msgs
#   cv_bridge
# )
# target_link_libraries(mesh_vo_core ${OpenCV_LIBRARIES})

add_executable(mesh_vo_ros_node src/mesh_vo_ros_node.cpp)

ament_target_dependencies(mesh_vo_ros_node
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
)

target_link_libraries(mesh_vo_ros_node
  ${OpenCV_LIBRARIES}
  mesh_vo_core
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
  mesh_vo_ros_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
