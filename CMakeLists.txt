cmake_minimum_required(VERSION 3.16.3)

project(mesh_vo LANGUAGES C CXX)

# --- Options ------------------------------------------------------
option(COMPILE_CPU     "Enable CPU compilation"           ON)
option(COMPILE_GL      "Enable OpenGL compilation"        ON)
option(COMPILE_GLES2   "Enable GLES2 compilation"         OFF)
option(COMPILE_HLS     "Enable HLS synthesis"             OFF)
option(COMPILE_XRT     "Enable XRT compilation"           OFF)
option(COMPILE_DEMO    "Enable Demo compilation"          OFF)
option(COMPILE_TUM     "Enable Tum app compilation"       ON)
option(COMPILE_TESTS   "Enable Tests compilation"         ON)
option(COMPILE_ROS2    "Enable ROS2 node compilation"     ON)

# --- C++ Standard -------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Dependencies -------------------------------------------------
find_package(Eigen3      REQUIRED)
find_package(OpenCV      REQUIRED)
#find_package(Pangolin    REQUIRED)
find_package(OpenGL      REQUIRED)
find_package(Threads     REQUIRED)

find_package(LAPACK REQUIRED)                 # provides LAPACK::LAPACK :contentReference[oaicite:1]{index=1}
# LAPACKE is commonly separate:
find_library(LAPACKE_LIB lapacke REQUIRED)


# Some projects set these (e.g. from parent CMake). Keep them as-is if defined.
#   CSPARSE_INCLUDE_DIR
#   CHOLMOD_INCLUDE_DIR

# Provide Eigen3::Eigen if the FindEigen3 module did not
if(NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
  )
endif()

# --- GL / GLES2 sanity check --------------------------------------
if(COMPILE_GL AND COMPILE_GLES2)
  message(WARNING
    "Both COMPILE_GL and COMPILE_GLES2 are ON. "
    "Please pick one; disabling both for now."
  )
  set(COMPILE_GL OFF)
  set(COMPILE_GLES2 OFF)
endif()

# --- GLAD (GL / GLES2) --------------------------------------------
if(COMPILE_GL OR COMPILE_GLES2)
  add_library(glad STATIC)

  target_sources(glad PRIVATE
    $<$<BOOL:${COMPILE_GL}>:
      ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty/glad2/src/gl.c>
    $<$<BOOL:${COMPILE_GLES2}>:
      ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty/glad2/src/gles2.c>
    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty/glad2/src/egl.c
  )

  target_include_directories(glad
    PUBLIC
      ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty/glad2/include
  )
endif()

# --- Core Library -------------------------------------------------
set(VISUALODOMETRY_SRCS
  src/params.cpp
  src/optimizers/poseOptimizer.cpp
  src/optimizers/poseExpOptimizer.cpp
#  src/optimizers/poseVelOptimizer.cpp
#  src/optimizers/poseVelExpOptimizer.cpp
  src/optimizers/depthOptimizer.cpp
  src/optimizers/depthExpOptimizer.cpp
  src/optimizers/rayDepthOptimizer.cpp
  src/optimizers/vertexOptimizer.cpp
  src/optimizers/poseDepthOptimizer.cpp
  src/optimizers/poseExpDepthOptimizer.cpp
#  src/optimizers/poseVelMapOptimizer.cpp
#  src/optimizers/poseVelExpMapOptimizer.cpp
)

add_library(mesh_vo STATIC ${VISUALODOMETRY_SRCS})

target_include_directories(mesh_vo
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/src
    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/HLSLinearAlgebra/include
    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
#    ${CSPARSE_INCLUDE_DIR}
#    ${CHOLMOD_INCLUDE_DIR}
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(mesh_vo
  PUBLIC
    Eigen3::Eigen
    ${Pangolin_LIBRARIES}
  PRIVATE
    ${OpenCV_LIBS}
    LAPACK::LAPACK ${LAPACKE_LIB}
)

# Make mesh_vo depend on glad so glad's PUBLIC includes propagate
if(COMPILE_GL OR COMPILE_GLES2)
  target_link_libraries(mesh_vo
    PUBLIC
      glad
  )
endif()

if(COMPILE_TUM)
add_subdirectory(src/apps/tum_dataset)
endif()

#if(COMPILE_DEMO)
#add_subdirectory(src/apps/tum_dataset)
#endif()

# --- Tests --------------------------------------------------------
if(COMPILE_TESTS)
  include(CTest)
  include(FetchContent)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
  )

  # Avoid changing parent's cache unless you really want to
  # set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # for MSVC if needed

  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)

  add_executable(poseEstimation_tests
    tests/poseEstimation_test.cpp
  )

  target_link_libraries(poseEstimation_tests
    PRIVATE
      mesh_vo
      gtest_main
      Eigen3::Eigen
      ${OpenCV_LIBS}
      ${Pangolin_LIBRARIES}
      Threads::Threads
  )

  target_include_directories(poseEstimation_tests
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty
      ${OpenCV_INCLUDE_DIRS}
  )

  target_compile_definitions(poseEstimation_tests
    PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
  )

  if(COMPILE_GL)
    target_link_libraries(poseEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(poseEstimation_tests PRIVATE COMPILE_GL)
  elseif(COMPILE_GLES2)
    target_link_libraries(poseEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(poseEstimation_tests PRIVATE COMPILE_GLES2)
  endif()

  gtest_discover_tests(poseEstimation_tests)


  add_executable(depthEstimation_tests
    tests/depthEstimation_test.cpp
  )

  target_link_libraries(depthEstimation_tests
    PRIVATE
      mesh_vo
      gtest_main
      Eigen3::Eigen
      ${OpenCV_LIBS}
      ${Pangolin_LIBRARIES}
      Threads::Threads
  )

  target_include_directories(depthEstimation_tests
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty
      ${OpenCV_INCLUDE_DIRS}
  )

  target_compile_definitions(depthEstimation_tests
    PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
  )

  if(COMPILE_GL)
    target_link_libraries(depthEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(depthEstimation_tests PRIVATE COMPILE_GL)
  elseif(COMPILE_GLES2)
    target_link_libraries(depthEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(depthEstimation_tests PRIVATE COMPILE_GLES2)
  endif()

  gtest_discover_tests(depthEstimation_tests)




  add_executable(rayDepthEstimation_tests
    tests/rayDepthEstimation_test.cpp
  )

  target_link_libraries(rayDepthEstimation_tests
    PRIVATE
      mesh_vo
      gtest_main
      Eigen3::Eigen
      ${OpenCV_LIBS}
      ${Pangolin_LIBRARIES}
      Threads::Threads
  )

  target_include_directories(rayDepthEstimation_tests
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty
      ${OpenCV_INCLUDE_DIRS}
  )

  target_compile_definitions(rayDepthEstimation_tests
    PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
  )

  if(COMPILE_GL)
    target_link_libraries(rayDepthEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(rayDepthEstimation_tests PRIVATE COMPILE_GL)
  elseif(COMPILE_GLES2)
    target_link_libraries(rayDepthEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(rayDepthEstimation_tests PRIVATE COMPILE_GLES2)
  endif()

  gtest_discover_tests(rayDepthEstimation_tests)





  add_executable(vertexEstimation_tests
    tests/vertexEstimation_test.cpp
  )

  target_link_libraries(vertexEstimation_tests
    PRIVATE
      mesh_vo
      gtest_main
      Eigen3::Eigen
      ${OpenCV_LIBS}
      ${Pangolin_LIBRARIES}
      Threads::Threads
  )

  target_include_directories(vertexEstimation_tests
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty
      ${OpenCV_INCLUDE_DIRS}
  )

  target_compile_definitions(vertexEstimation_tests
    PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
  )

  if(COMPILE_GL)
    target_link_libraries(vertexEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(vertexEstimation_tests PRIVATE COMPILE_GL)
  elseif(COMPILE_GLES2)
    target_link_libraries(vertexEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(vertexEstimation_tests PRIVATE COMPILE_GLES2)
  endif()

  gtest_discover_tests(vertexEstimation_tests)

  add_executable(poseDepthEstimation_tests
    tests/poseDepthEstimation_test.cpp
  )

  target_link_libraries(poseDepthEstimation_tests
    PRIVATE
      mesh_vo
      gtest_main
      Eigen3::Eigen
      ${OpenCV_LIBS}
      ${Pangolin_LIBRARIES}
      Threads::Threads
  )

  target_include_directories(poseDepthEstimation_tests
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty
      ${OpenCV_INCLUDE_DIRS}
  )

  target_compile_definitions(poseDepthEstimation_tests
    PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
  )

  if(COMPILE_GL)
    target_link_libraries(poseDepthEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(poseDepthEstimation_tests PRIVATE COMPILE_GL)
  elseif(COMPILE_GLES2)
    target_link_libraries(poseDepthEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(poseDepthEstimation_tests PRIVATE COMPILE_GLES2)
  endif()

  gtest_discover_tests(poseDepthEstimation_tests)
endif()

# --- ROS 2 Node ---------------------------------------------------
if(COMPILE_ROS2)
  add_subdirectory(src/mesh_vo_ros)
endif()