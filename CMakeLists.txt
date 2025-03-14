cmake_minimum_required(VERSION 3.16.3)
project(mesh_vo)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

if(NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

include_directories("src")
#include_directories("include")
include_directories(
        ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        )

include_directories( ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 

#file(GLOB VISUALODOMETRY_SRCS "src/*.cpp"
#                              "src/common/*.cpp"
#                              "src/cpu/*.cpp"
#                              "src/scene/*.cpp"
#                              "src/utils/*.cpp"
#)

set(VISUALODOMETRY_SRCS
      "src/params.cpp"
      "src/common/DelaunayTriangulation.cpp"
      "src/optimizers/baseOptimizerCPU.cpp"
      "src/optimizers/poseOptimizerCPU.cpp"
      "src/optimizers/poseVelOptimizerCPU.cpp"
      "src/optimizers/mapOptimizerCPU.cpp"
      "src/optimizers/poseMapOptimizerCPU.cpp"
      "src/optimizers/intrinsicPoseMapOptimizerCPU.cpp"
      "src/visualOdometry.cpp")

#file(GLOB TESTAPP_SRCS "src/test/*.cpp"
#                       "src/test/utils/*.cpp")

#file(GLOB TUMDATASET_SRCS "src/tum_dataset/*.cpp")

set(TESTAPP_SRCS
      "src/apps/test/main.cpp"
      "src/apps/test/utils/convertAhandaPovRayToStandard.cpp"
)

set(TUMDATASET_SRCS
      "src/apps/tum_dataset/main_on_images.cpp"
      "src/apps/tum_dataset/Undistorter.cpp"
)

set(KITTI_SRCS
      "src/apps/kitti/main_on_images.cpp"
)

set(KITTI_BINOCULAR_SRCS
      "src/apps/kitti_binocular/main_on_images.cpp"
)

set(SYFRA_SRCS
      "src/apps/syfra/main_on_images.cpp"
)

set(RGBD_SRCS
      "src/apps/rgbd_dataset/main_on_images.cpp"
)

#file(GLOB SHADERS  "src/shaders/*")

#foreach(SHADER ${SHADERS})
#  file(COPY ${SHADER} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
#endforeach(SHADER)

add_library(mesh_vo STATIC ${VISUALODOMETRY_SRCS})
#target_precompile_headers(mesh_vo PRIVATE "include/pch.h")
target_include_directories(mesh_vo PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(testapp ${TESTAPP_SRCS})
add_executable(tum_dataset ${TUMDATASET_SRCS})
add_executable(rgbd_dataset ${RGBD_SRCS})
#add_executable(kitti_dataset ${KITTI_SRCS} ${VISUALODOMETRY_SRCS})
#add_executable(kitti_binocular ${KITTI_BINOCULAR_SRCS} ${VISUALODOMETRY_SRCS})
#add_executable(syfra ${SYFRA_SRCS})

#target_link_libraries(testapp ${OpenCV_LIBS} glfw OpenGL::GL ${CMAKE_DL_LIBS})
target_link_libraries(testapp mesh_vo ${OpenCV_LIBS} ${Pangolin_LIBRARIES} pthread)
target_link_libraries(tum_dataset mesh_vo ${OpenCV_LIBS} ${Pangolin_LIBRARIES} pthread)
target_link_libraries(rgbd_dataset mesh_vo ${OpenCV_LIBS} ${Pangolin_LIBRARIES} pthread)
#target_link_libraries(kitti_dataset ${OpenCV_LIBS} pthread)
#target_link_libraries(kitti_binocular ${OpenCV_LIBS} pthread)
#target_link_libraries(syfra mesh_vo ${OpenCV_LIBS} pthread)





# --- Testing Setup ---

# Enable testing
enable_testing()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


# Create test executable(s)
add_executable(mesh_vo_tests
    tests/rendererCPU_test.cpp
)

# Link your test executable with Google Test and your project library
target_link_libraries(mesh_vo_tests PRIVATE gtest_main mesh_vo ${OpenCV_LIBS} ${Pangolin_LIBRARIES} pthread)

# Pass the absolute path to the test data directory as a compile definition
target_compile_definitions(mesh_vo_tests PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

# Register the test with CTest
add_test(NAME mesh_vo_tests COMMAND mesh_vo_tests)
