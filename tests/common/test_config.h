#pragma once

#include <string>

// Test configuration constants
namespace TestConfig {
    // Performance thresholds (milliseconds)
    constexpr double CPU_DEPTH_THRESHOLD_MS = 1000.0;
    constexpr double GL_DEPTH_THRESHOLD_MS = 500.0;
    constexpr double XRT_DEPTH_THRESHOLD_MS = 200.0;
    
    constexpr double CPU_IMAGE_THRESHOLD_MS = 2000.0;
    constexpr double GL_IMAGE_THRESHOLD_MS = 1000.0;
    
    // Error tolerances
    constexpr double DEPTH_ERROR_TOLERANCE = 1e-3;
    constexpr double IMAGE_ERROR_TOLERANCE = 1e-2;
    constexpr double CROSS_BACKEND_ERROR_TOLERANCE = 0.5;
    
    // Performance ratios
    constexpr double MAX_PERFORMANCE_RATIO = 5.0;
    constexpr double MIN_GL_SPEEDUP = 2.0;
    
    // Memory limits (MB)
    constexpr size_t MAX_MEMORY_PER_TEST_MB = 500;
    constexpr size_t MAX_TOTAL_MEMORY_MB = 2048;
    
    // Test data configuration
    const std::string DEFAULT_TEST_DATA_SUBDIR = "rgbd_dataset_freiburg1_floor_part";
    constexpr int DEFAULT_TEST_WIDTH = 640;
    constexpr int DEFAULT_TEST_HEIGHT = 480;
    
    // Debug output settings
    constexpr bool SAVE_DEBUG_IMAGES = true;
    constexpr bool VERBOSE_OUTPUT = false;
    
    // Test execution settings
    constexpr int DEFAULT_PERFORMANCE_RUNS = 5;
    constexpr int DEFAULT_PRECISION_RUNS = 3;
    constexpr int MEMORY_TEST_ITERATIONS = 10;
}
