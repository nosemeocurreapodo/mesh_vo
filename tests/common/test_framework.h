#pragma once

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <chrono>
// #include <memory>
// #include <string>
// #include <vector>
// #include <functional>
// #include <fstream>
// #include <sstream>

#include "mpdr/common/types.h"
#include "mpdr/common/mesh_helpers.h"
#include "mpdr/common/helpers.h"
#include "mpdr/backends/base/renderref.h"

#include "mpdr/backends/cpu/texturecpu.h"
#include "mpdr/backends/cpu/meshcpu.h"
#include "mpdr/backends/cpu/renderercpu.h"
//#include "backends/cpu/deferredreducecpu.h"
#include "mpdr/backends/cpu/deferredrenderercpu.h"

#ifdef COMPILE_GL
#include "mpdr/backends/gl/devicegl_glad.h"
#include "mpdr/backends/gl/texturegl.h"
#include "mpdr/backends/gl/meshgl.h"
#include "mpdr/backends/gl/renderergl.h"
#include "mpdr/backends/gl/deferredreducegl.h"
#include "mpdr/backends/gl/reducergl.h"
#include "mpdr/backends/gl/nvdiffrastgl.h"
#endif

#ifdef COMPILE_HLS
// #include "backends/xrt/hls/devicegl_glad.h"
#include "mpdr/backends/xrt/hls/texturehls.h"
#include "mpdr/backends/xrt/hls/meshhls.h"
#include "mpdr/backends/xrt/hls/rendererhls.h"
#endif

#include "test_config.h"
#include "loaddataset.h"

// Performance measurement utilities
class PerformanceTimer
{
public:
    void Start()
    {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double Stop()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
        return duration.count() / 1000.0; // Return milliseconds
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

// Test result structure for validation
template <typename T>
struct TestResult
{
    cv::Mat output;
    double execution_time_ms;
    std::string backend_name;
    bool success;
    std::string error_message;

    TestResult() : execution_time_ms(0.0), success(false) {}
};

// Base test fixture for all renderer tests
class RendererTestBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Load test dataset
        // dataset_ = std::make_unique<LoadDatasetIclNuim>(std::string(TEST_DATA_DIR));
        dataset_ = std::make_unique<LoadDesktopDataset>(std::string(TEST_DATA_DIR));
        // dataset_ = std::make_unique<LoadDatasetTumRgbd>(std::string(TEST_DATA_DIR));

        image_files_ = dataset_->GetImageFiles();
        depth_files_ = dataset_->GetDepthFiles();
        poses_ = dataset_->GetPoses();
        cam_ = dataset_->GetCamera();
        w_ = dataset_->GetWidth();
        h_ = dataset_->GetHeight();
        depth_factor_ = dataset_->GetDepthFactor();
    }

    void TearDown() override
    {
        // Cleanup
        dataset_.reset();
    }

protected:
    // Dataset and test data
    // std::unique_ptr<LoadDatasetIclNuim> dataset_;
    std::unique_ptr<LoadDesktopDataset> dataset_;
    // std::unique_ptr<LoadDatasetTumRgbd> dataset_;

    std::vector<std::string> image_files_, depth_files_;
    std::vector<SE3<float>> poses_;
    PinholeCamera<float> cam_;
    int w_, h_;
    float depth_factor_;

    // Test frames
    // cv::Mat image_src_cv_, depth_src_cv_, image_dst_cv_, depth_dst_cv_;
    // linalg::SE3<float> pose_src_, pose_dst_;

    // Mesh data
    // std::vector<float> vertices_, texcoords_, weights_;
    // std::vector<int> indices_;

    // Performance timer
    PerformanceTimer timer_;
};

class TwoViewTests : public RendererTestBase
{
protected:
    void SetUp() override
    {
        RendererTestBase::SetUp();

        float scale = 1.0f / depth_factor_;

        image_src_cv_ = cv::imread(image_files_[0], cv::IMREAD_GRAYSCALE);
        depth_src_cv_ = cv::imread(depth_files_[0], cv::IMREAD_GRAYSCALE);
        depth_src_cv_.convertTo(depth_src_cv_, CV_32FC1);
        depth_src_cv_ = depth_src_cv_ * scale;

        pose_src_ = poses_[0];

        image_dst_cv_ = cv::imread(image_files_[50], cv::IMREAD_GRAYSCALE);
        depth_dst_cv_ = cv::imread(depth_files_[50], cv::IMREAD_GRAYSCALE);
        depth_dst_cv_.convertTo(depth_dst_cv_, CV_32FC1);
        depth_dst_cv_ = depth_dst_cv_ * scale;

        pose_dst_ = poses_[50];

        // TextureCPU<float> depth_src_cpu(depth_src_cv_.cols, depth_src_cv_.rows, 0.0f);
        // UploadMatToTexture(depth_src_cpu, 0, depth_src_cv_);

        // CreateMesh(depth_src_cpu, cam_, 24, vertex_, indices_, true, false, false);
        //  float meanDepth = cv::mean(depth_src_cv_)[0];
        //  CreateFlatMesh(meanDepth * 0.5, meanDepth * 1.5, cam_, 16, vertex_, indices_, true, true, true);

        // CreateScreenQuad(screen_vertex_, screen_indices_);
    }

    cv::Mat image_src_cv_, depth_src_cv_, image_dst_cv_, depth_dst_cv_;

    SE3<float> pose_src_, pose_dst_;

    // std::vector<float> vertex_;
    // std::vector<int> indices_;

    // std::vector<float> screen_vertex_;
    // std::vector<int> screen_indices_;
};

// Test result reporting utilities
class TestReporter
{
public:
    struct TestResult
    {
        std::string test_name;
        std::string backend;
        double execution_time_ms;
        double memory_usage_mb;
        double error_metric;
        bool passed;
        std::string failure_reason;
    };

    static void RecordResult(const TestResult &result)
    {
        results_.push_back(result);
    }

    static void GenerateReport(const std::string &output_file = "test_report.json")
    {
        std::ofstream file(output_file);
        file << "{\n";
        file << "  \"test_results\": [\n";

        for (size_t i = 0; i < results_.size(); ++i)
        {
            const auto &r = results_[i];
            file << "    {\n";
            file << "      \"test_name\": \"" << r.test_name << "\",\n";
            file << "      \"backend\": \"" << r.backend << "\",\n";
            file << "      \"execution_time_ms\": " << r.execution_time_ms << ",\n";
            file << "      \"memory_usage_mb\": " << r.memory_usage_mb << ",\n";
            file << "      \"error_metric\": " << r.error_metric << ",\n";
            file << "      \"passed\": " << (r.passed ? "true" : "false") << ",\n";
            file << "      \"failure_reason\": \"" << r.failure_reason << "\"\n";
            file << "    }" << (i < results_.size() - 1 ? "," : "") << "\n";
        }

        file << "  ],\n";
        file << "  \"summary\": {\n";
        file << "    \"total_tests\": " << results_.size() << ",\n";
        file << "    \"passed_tests\": " << CountPassed() << ",\n";
        file << "    \"failed_tests\": " << CountFailed() << ",\n";
        file << "    \"pass_rate\": " << (results_.empty() ? 0.0 : (double)CountPassed() / results_.size() * 100.0) << "\n";
        file << "  }\n";
        file << "}\n";
    }

private:
    static std::vector<TestResult> results_;

    static size_t CountPassed()
    {
        return std::count_if(results_.begin(), results_.end(),
                             [](const TestResult &r)
                             { return r.passed; });
    }

    static size_t CountFailed()
    {
        return results_.size() - CountPassed();
    }
};

// Memory usage monitoring
class MemoryMonitor
{
public:
    static size_t GetCurrentMemoryUsage()
    {
        // Simple implementation - in production, use more sophisticated memory tracking
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line))
        {
            if (line.substr(0, 6) == "VmRSS:")
            {
                std::istringstream iss(line);
                std::string key, value, unit;
                iss >> key >> value >> unit;
                return std::stoul(value) / 1024; // Convert KB to MB
            }
        }
        return 0;
    }

    static void CheckMemoryLimits()
    {
        size_t current_mb = GetCurrentMemoryUsage();
        EXPECT_LT(current_mb, TestConfig::MAX_MEMORY_PER_TEST_MB)
            << "Memory usage exceeded limit: " << current_mb << "MB";
    }
};

// Validation thresholds
struct ValidationThresholds
{
    double gt_max_depth_error = 0.172;
    double gt_max_image_error = 6.1;

    double ref_gt_max_depth_error = 0.171;
    double ref_gt_max_image_error = 14.3;

    double ref_max_depth_error = 0.178;
    double ref_max_image_error = 12.1;
    double ref_max_residual_error = 0.0034;
    double ref_max_jtra_error = 18.31;
    double ref_max_jrot_error = 27.2;
    double ref_max_jdepth_error = 2.4;
    double ref_max_jv0_error = 2.4;
    double ref_max_jv1_error = 2.4;
    double ref_max_jv2_error = 2.4;

    int cr_max_valid_diff = 200;
    double cr_max_mipmap_error = 0.00015;
    double cr_max_fpos_error = 0.00015;
    double cr_max_kfpos_error = 0.00015;
    double cr_max_bcid_error = 0.00015;
    double cr_max_depth_error = 1.11e-6;
    double cr_max_image_error = 0.63;
    double cr_max_residual_error = 0.63;
    double cr_max_l2_error = 79.0;
    double cr_max_didxy_error = 0.0073;
    double cr_max_jtra_error = 0.19;
    double cr_max_jrot_error = 0.34;
    double cr_max_jexp_error = 0.0069;
    double cr_max_jdepth_error = 0.063;
    double cr_max_jv0_error = 0.063;
    double cr_max_jv1_error = 0.063;
    double cr_max_jv2_error = 0.063;
    double cr_max_pids_error = 0.11;
};

/*
// Test result validation
class TestValidator
{
public:
    static void ValidateAgainstGroundTruth(const cv::Mat &result, const cv::Mat &ground_truth,
                                           const ValidationThresholds &thresholds)
    {
        double error = ComputeImageError<float>(result, ground_truth, 0.0f);
        EXPECT_LT(error, thresholds.max_l2_error)
            << "Ground truth validation failed with L2 error: " << error;
    }

    static void ValidateCrossBackend(const cv::Mat &cpu_result, const cv::Mat &gl_result,
                                     const ValidationThresholds &thresholds)
    {
        double error = ComputeImageError<float>(cpu_result, gl_result, 0.0f);
        EXPECT_LT(error, thresholds.max_cross_backend_error)
            << "Cross-backend validation failed with L2 error: " << error;
    }

    static void ValidatePerformance(double cpu_time, double gl_time,
                                    const ValidationThresholds &thresholds)
    {
        EXPECT_LT(cpu_time, thresholds.max_cpu_time_ms)
            << "CPU execution time exceeded threshold: " << cpu_time << "ms";
        EXPECT_LT(gl_time, thresholds.max_gl_time_ms)
            << "GL execution time exceeded threshold: " << gl_time << "ms";

        if (gl_time > 0)
        {
            double ratio = cpu_time / gl_time;
            EXPECT_LT(ratio, thresholds.max_performance_ratio)
                << "Performance ratio CPU/GL exceeded threshold: " << ratio;
        }
    }
};
*/