add_executable(tum_app main_on_images.cpp Undistorter.cpp)

target_include_directories(tum_app
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

target_link_libraries(tum_app
  PRIVATE
    mesh_vo_core
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    Threads::Threads
)

if(COMPILE_CPU)
  target_compile_definitions(tum_app PRIVATE COMPILE_CPU)
endif()
#if(COMPILE_GL)
#  target_compile_definitions(tum_app PRIVATE COMPILE_GL)
#  target_link_libraries(tum_app EGL glad)
#endif()
if(COMPILE_GLES2)
  target_compile_definitions(tum_app PRIVATE COMPILE_GLES2)
  target_link_libraries(tum_app EGL glad)
endif()
if(COMPILE_XRT)
  target_compile_definitions(tum_app PRIVATE COMPILE_XRT)
  target_link_libraries(tum_app xilinxopencl rt stdc++ uuid xrt_coreutil)
endif()


