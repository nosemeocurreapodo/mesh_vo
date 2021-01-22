cmake_minimum_required (VERSION 2.6)
project(mesh_vo)

set(CMAKE_BUILD_TYPE debug)

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)
#set(EIGEN3_INCLUDE_DIR "C:/Emanuel/eigen-3.3.8")
include_directories(${EIGEN3_INCLUDE_DIR})
#message(STATUS "Eigen dir: ${EIGEN3_INCLUDE_DIR}")

find_package(glfw3 REQUIRED)
#include_directories(${GLFW3_INCLUDE_DIR})

#find_package(OpenGL REQUIRED)

find_package(glm REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

include_directories("src")
include_directories("include")
include_directories(
        ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
        )

file(GLOB DepthEstimator_SRCS "src/*.cpp"
                              "src/*.c"
                              "src/Utils/*.cpp"
)

file(GLOB TESTAPP_SRCS "src/test/*.cpp"
                       "src/test/utils/*.cpp")

file(GLOB TUMDATASET_SRCS "src/tum_dataset/*.cpp")

file(GLOB SHADERS  "src/shaders/*"
                   )
foreach(SHADER ${SHADERS})
  file(COPY ${SHADER} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach(SHADER)

# build example app
add_executable(testapp src/test/main.cpp ${TESTAPP_SRCS} ${DepthEstimator_SRCS} ${SHADERS})
#target_link_libraries(testapp ${OpenCV_LIBS} glfw OpenGL::GL ${CMAKE_DL_LIBS})
target_link_libraries(testapp ${OpenCV_LIBS} glfw ${CMAKE_DL_LIBS})

add_executable(tum_dataset src/tum_dataset/main_on_images.cpp ${TUMDATASET_SRCS} ${DepthEstimator_SRCS} ${SHADERS})
target_link_libraries(tum_dataset ${OpenCV_LIBS} glfw ${CMAKE_DL_LIBS})
