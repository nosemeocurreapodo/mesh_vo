cmake_minimum_required(VERSION 3.16)
project(mesh_vo_ros)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  add_compile_options(-Wall -Wextra -Wpedantic)
#endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LAPACK REQUIRED)               
find_library(LAPACKE_LIB lapacke REQUIRED)

# Add the include directory for the mesh_vo library
set(MESH_VO_SRC_DIR ../../../)

include_directories(
    ${MESH_VO_SRC_DIR}/src
    ${MESH_VO_SRC_DIR}/MultiPlatformDiffRenderer/src
    ${MESH_VO_SRC_DIR}/MultiPlatformDiffRenderer/HLSLinearAlgebra/include
    ${MESH_VO_SRC_DIR}/MultiPlatformDiffRenderer/thirdparty/Sophus
    ${OpenCV_INCLUDE_DIRS}
)

set(VISUALODOMETRY_SRCS
  ${MESH_VO_SRC_DIR}/src/params.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/poseOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/poseExpOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/poseVelOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/poseVelExpOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/depthOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/depthExpOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/rayDepthOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/vertexOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/poseDepthOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/poseExpDepthOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/poseVelMapOptimizer.cpp
#  ${MESH_VO_SRC_DIR}/src/optimizers/poseVelExpMapOptimizer.cpp
)

add_library(mesh_vo_core STATIC ${VISUALODOMETRY_SRCS})

#target_include_directories(mesh_vo_core
#  PUBLIC
#    ${PROJECT_SOURCE_DIR}/src
#    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/src
#    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/HLSLinearAlgebra/include
#    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty/Sophus
#    ${EIGEN3_INCLUDE_DIR}
#    ${Pangolin_INCLUDE_DIRS}
#    ${CSPARSE_INCLUDE_DIR}
#    ${CHOLMOD_INCLUDE_DIR}
#  PRIVATE
#    ${OpenCV_INCLUDE_DIRS}
#)

target_link_libraries(mesh_vo_core
  PUBLIC
    Eigen3::Eigen
    ${Pangolin_LIBRARIES}
  PRIVATE
    ${OpenCV_LIBS}
    LAPACK::LAPACK ${LAPACKE_LIB}
)

# Make mesh_vo depend on glad so glad's PUBLIC includes propagate
if(COMPILE_GL OR COMPILE_GLES2)
  target_link_libraries(mesh_vo_core
    PUBLIC
      glad
  )
endif()

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

#ament_target_dependencies(mesh_vo_ros_node
#  rclcpp
#  sensor_msgs
#  geometry_msgs
#  visualization_msgs
#  cv_bridge
#)

target_link_libraries(mesh_vo_ros_node PUBLIC
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  cv_bridge::cv_bridge
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs_library
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
