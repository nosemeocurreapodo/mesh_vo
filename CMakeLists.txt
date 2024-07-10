cmake_minimum_required(VERSION 3.22)

project(mesh_vo)

set(CMAKE_BUILD_TYPE Debug)

#set(CMAKE_CXX_FLAGS " -Wall -Wextra -g")
#set(CMAKE_CXX_FLAGS_DEBUG "-pg")
#set(CMAKE_CXX_FLAGS_Debug "-O3")

#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

find_package(Boost COMPONENTS system thread) 
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(glfw3 REQUIRED)
#include_directories(${GLFW3_INCLUDE_DIR})
#find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
#include_directories(${GLM_INCLUDE_DIRS})

include_directories("src")
include_directories("include")
include_directories(
        ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
        )

#file(GLOB VISUALODOMETRY_SRCS "src/*.cpp"
#                              "src/common/*.cpp"
#                              "src/cpu/*.cpp"
#                              "src/scene/*.cpp"
#                              "src/utils/*.cpp"
#)

set(VISUALODOMETRY_SRCS
      "src/cpu/MeshCPU.cpp"
      "src/cpu/renderCPU.cpp"
      "src/common/common.cpp"
      "src/common/DelaunayTriangulation.cpp"
      "src/optimizers/meshOptimizerCPU.cpp"
      "src/visualOdometry.cpp")

#file(GLOB TESTAPP_SRCS "src/test/*.cpp"
#                       "src/test/utils/*.cpp")

#file(GLOB TUMDATASET_SRCS "src/tum_dataset/*.cpp")

set(TESTAPP_SRCS
      "src/test/main.cpp"
      "src/test/utils/convertAhandaPovRayToStandard.cpp"
)

set(TUMDATASET_SRCS
      "src/tum_dataset/main_on_images.cpp"
      "src/tum_dataset/Undistorter.cpp"
)

file(GLOB SHADERS  "src/shaders/*"
                   )
foreach(SHADER ${SHADERS})
  file(COPY ${SHADER} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach(SHADER)

# build example app
add_executable(testapp ${TESTAPP_SRCS} ${VISUALODOMETRY_SRCS})
#target_link_libraries(testapp ${OpenCV_LIBS} glfw OpenGL::GL ${CMAKE_DL_LIBS})
target_link_libraries(testapp boost_system boost_thread ${OpenCV_LIBS} glfw ${CMAKE_DL_LIBS})

#add_executable(tum_dataset ${TUMDATASET_SRCS} ${DepthEstimator_SRCS} ${SHADERS})
#target_link_libraries(tum_dataset ${OpenCV_LIBS} glfw ${CMAKE_DL_LIBS})
