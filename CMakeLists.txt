cmake_minimum_required (VERSION 2.6)
project(mesh_vo)

set(CMAKE_BUILD_TYPE release) 

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#find_package(OpenCV2.4.13 REQUIRED)
find_package(OpenCV REQUIRED )

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories("src")
include_directories("include")
include_directories(
        ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
        )

file(GLOB DepthEstimator_SRCS "src/*.cpp"
                              "src/Utils/*.cpp"
)

file(GLOB TESTAPP_SRCS "src/test/*.cpp"
                       "src/test/utils/*.cpp")

# build shared library.
add_library(mesh_vo SHARED ${DepthEstimator_SRCS})
target_link_libraries(mesh_vo ${OpenCV_LIBS})

# build example app
add_executable (testapp src/test/main.cpp ${TESTAPP_SRCS})
target_link_libraries(testapp mesh_vo ${OpenCV_LIBS})
