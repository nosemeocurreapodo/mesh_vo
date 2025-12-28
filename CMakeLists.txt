cmake_minimum_required(VERSION 3.16.3)

project(mesh_vo LANGUAGES C CXX)

# --- Options ------------------------------------------------------
option(COMPILE_CPU     "Enable CPU compilation"           ON)
option(COMPILE_GL      "Enable OpenGL compilation"        ON)
option(COMPILE_GLES2   "Enable GLES2 compilation"         OFF)
option(COMPILE_HLS     "Enable HLS synthesis"             OFF)
option(COMPILE_XRT     "Enable XRT compilation"           OFF)
option(COMPILE_DEMO    "Enable Demo compilation"          OFF)
option(COMPILE_TESTS   "Enable Tests compilation"         ON)

# --- C++ Standard -------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Dependencies -------------------------------------------------
find_package(Eigen3      REQUIRED)
find_package(OpenCV      REQUIRED)
find_package(Pangolin    REQUIRED)
find_package(OpenGL      REQUIRED)
find_package(Threads     REQUIRED)

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

# --- Core Library -------------------------------------------------
set(VISUALODOMETRY_SRCS
  src/params.cpp
  src/optimizers/poseOptimizer.cpp
  src/optimizers/poseExpOptimizer.cpp
  src/optimizers/poseVelOptimizer.cpp
  src/optimizers/poseVelExpOptimizer.cpp
  src/optimizers/mapOptimizer.cpp
  src/optimizers/mapExpOptimizer.cpp
  src/optimizers/poseMapOptimizer.cpp
  src/optimizers/poseExpMapOptimizer.cpp
  src/optimizers/poseVelMapOptimizer.cpp
  src/optimizers/poseVelExpMapOptimizer.cpp
)

add_library(mesh_vo STATIC ${VISUALODOMETRY_SRCS})

target_include_directories(mesh_vo
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/src
    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/HLSLinearAlgebra/include
    ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty/Sophus
    ${Pangolin_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${CHOLMOD_INCLUDE_DIR}
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(mesh_vo
  PUBLIC
    Eigen3::Eigen
    ${Pangolin_LIBRARIES}
  PRIVATE
    ${OpenCV_LIBS}
)

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

# Make mesh_vo depend on glad so glad's PUBLIC includes propagate
if(COMPILE_GL OR COMPILE_GLES2)
  target_link_libraries(mesh_vo
    PUBLIC
      glad
  )
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


  add_executable(mapEstimation_tests
    tests/mapEstimation_test.cpp
  )

  target_link_libraries(mapEstimation_tests
    PRIVATE
      mesh_vo
      gtest_main
      Eigen3::Eigen
      ${OpenCV_LIBS}
      ${Pangolin_LIBRARIES}
      Threads::Threads
  )

  target_include_directories(mapEstimation_tests
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty
      ${OpenCV_INCLUDE_DIRS}
  )

  target_compile_definitions(mapEstimation_tests
    PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
  )

  if(COMPILE_GL)
    target_link_libraries(mapEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(mapEstimation_tests PRIVATE COMPILE_GL)
  elseif(COMPILE_GLES2)
    target_link_libraries(mapEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(mapEstimation_tests PRIVATE COMPILE_GLES2)
  endif()

  gtest_discover_tests(mapEstimation_tests)

  add_executable(poseMapEstimation_tests
    tests/poseMapEstimation_test.cpp
  )

  target_link_libraries(poseMapEstimation_tests
    PRIVATE
      mesh_vo
      gtest_main
      Eigen3::Eigen
      ${OpenCV_LIBS}
      ${Pangolin_LIBRARIES}
      Threads::Threads
  )

  target_include_directories(poseMapEstimation_tests
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/MultiPlatformDiffRenderer/thirdparty
      ${OpenCV_INCLUDE_DIRS}
  )

  target_compile_definitions(poseMapEstimation_tests
    PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
  )

  if(COMPILE_GL)
    target_link_libraries(poseMapEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(poseMapEstimation_tests PRIVATE COMPILE_GL)
  elseif(COMPILE_GLES2)
    target_link_libraries(poseMapEstimation_tests
      PRIVATE
        glad
        ${OPENGL_LIBRARIES}
        EGL
    )
    target_compile_definitions(poseMapEstimation_tests PRIVATE COMPILE_GLES2)
  endif()

  gtest_discover_tests(poseMapEstimation_tests)
endif()
