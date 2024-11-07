cmake_minimum_required(VERSION 3.22.1)

project(mesh_vo)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS
#    "-O3"
#   "-O3 -ffast-math -fno-math-errno -DNDEBUG"
#"-O3 -ffast-math -march=native -fopenmp -DNDEBUG"
"-g -DEBUG -fno-omit-frame-pointer -DDEBUG"
)

include_directories("src")
include_directories("include")
include_directories(
        ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
        ${EIGEN3_INCLUDE_DIR}
        )

include_directories( ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 

#file(GLOB VISUALODOMETRY_SRCS "src/*.cpp"
#                              "src/common/*.cpp"
#                              "src/cpu/*.cpp"
#                              "src/scene/*.cpp"
#                              "src/utils/*.cpp"
#)

set(VISUALODOMETRY_SRCS
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

set(KITTI_SRCS
      "src/kitti/main_on_images.cpp"
)

set(KITTI_BINOCULAR_SRCS
      "src/kitti_binocular/main_on_images.cpp"
)

file(GLOB SHADERS  "src/shaders/*"
                   )
foreach(SHADER ${SHADERS})
  file(COPY ${SHADER} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach(SHADER)

# build example app
add_executable(testapp ${TESTAPP_SRCS} ${VISUALODOMETRY_SRCS})
#target_link_libraries(testapp ${OpenCV_LIBS} glfw OpenGL::GL ${CMAKE_DL_LIBS})
target_link_libraries(testapp ${OpenCV_LIBS} glfw ${CMAKE_DL_LIBS})

add_executable(tum_dataset ${TUMDATASET_SRCS} ${VISUALODOMETRY_SRCS})
target_link_libraries(tum_dataset ${OpenCV_LIBS} glfw ${CMAKE_DL_LIBS})

add_executable(kitti_dataset ${KITTI_SRCS} ${VISUALODOMETRY_SRCS})
target_link_libraries(kitti_dataset ${OpenCV_LIBS} glfw ${CMAKE_DL_LIBS})

add_executable(kitti_binocular ${KITTI_BINOCULAR_SRCS} ${VISUALODOMETRY_SRCS})
target_link_libraries(kitti_binocular ${OpenCV_LIBS} glfw ${CMAKE_DL_LIBS})