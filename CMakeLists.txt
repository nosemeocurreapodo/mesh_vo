cmake_minimum_required(VERSION 3.16.3)
project(mesh_vo)

option(COMPILE_CPU "Enable CPU compilation" ON)
option(COMPILE_GL "Enable OpenGL compilation" ON)
option(COMPILE_GLES2 "Enable GLES2 compilation" OFF)
option(COMPILE_HLS "Enable HLS synthesis" OFF)
option(COMPILE_XRT "Enable XRT compilation" OFF)
option(COMPILE_DEMO "Enable Demo compilation" OFF)
option(COMPILE_TESTS "Enable Tests compilation" ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

if(NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

include_directories(
      "src"
      "MultiPlatformDiffRenderer/src"
      "MultiPlatformDiffRenderer/HLSLinearAlgebra/include"
      ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
      ${EIGEN3_INCLUDE_DIR}
      ${Pangolin_INCLUDE_DIRS}
      ${CSPARSE_INCLUDE_DIR} 
      ${CHOLMOD_INCLUDE_DIR}
)

if(COMPILE_GL AND COMPILE_GLES2)
    message(WARNING "Both GL and GLES2 are selected, plase just pick one, disabling both for nor")
    set(COMPILE_GL OFF)
    set(COMPILE_GLES2 OFF)
endif()

if(COMPILE_GL)
      #include_directories(
      #      "MultiPlatformDiffRenderer/thirdparty/glad2"
      #)
      add_library(glad STATIC
            MultiPlatformDiffRenderer/thirdparty/glad2/src/gl.c
            MultiPlatformDiffRenderer/thirdparty/glad2/src/egl.c
      )
      target_include_directories(glad PUBLIC 
            thirdparty
            MultiPlatformDiffRenderer/thirdparty/glad2/include
      )
      target_include_directories(glad PRIVATE thirdparty)
endif()

# Build glad library
if(COMPILE_GLES2)
      #include_directories(
      #      "MultiPlatformDiffRenderer/thirdparty/glad2/include"
      #)
      add_library(glad STATIC
            MultiPlatformDiffRenderer/thirdparty/glad2/src/gles2.c
            MultiPlatformDiffRenderer/thirdparty/glad2/src/egl.c
      )
      target_include_directories(glad PUBLIC 
            thirdparty
            MultiPlatformDiffRenderer/thirdparty/glad2/include
      )
      target_include_directories(glad PRIVATE thirdparty)
endif()

set(VISUALODOMETRY_SRCS
      "src/params.cpp"
      "src/optimizers/poseOptimizer.cpp"
      # "src/optimizers/mapOptimizer.cpp"
)

add_library(mesh_vo STATIC ${VISUALODOMETRY_SRCS})
#target_precompile_headers(mesh_vo PRIVATE "include/pch.h")
target_include_directories(mesh_vo PRIVATE ${OpenCV_INCLUDE_DIRS})

# --- Testing Setup ---

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.11.0)

FetchContent_MakeAvailable(googletest)

if(COMPILE_TESTS)
      include(CTest)
      include(GoogleTest)
      enable_testing()

      # Unified typed backend tests
      add_executable(poseEstimation_tests tests/poseEstimation_test.cpp)
      target_link_libraries(poseEstimation_tests 
            mesh_vo
            gtest_main 
            ${OpenCV_LIBS}
            Eigen3::Eigen
            ${Pangolin_LIBRARIES} 
            pthread
      )
      target_include_directories(poseEstimation_tests PRIVATE 
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty
            ${OpenCV_INCLUDE_DIRS}
      )
      target_compile_definitions(poseEstimation_tests PRIVATE
            TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
      )

      # Add GL-specific libs/defs when enabled
      if(COMPILE_GL)
            target_link_libraries(poseEstimation_tests 
                  ${OPENGL_LIBRARIES}
                  EGL
                  glad
            )
            target_compile_definitions(poseEstimation_tests PRIVATE COMPILE_GL)
      endif()

      gtest_discover_tests(poseEstimation_tests)
endif()
