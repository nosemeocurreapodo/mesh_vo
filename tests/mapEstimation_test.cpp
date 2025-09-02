#include <gtest/gtest.h>
#include "common/test_framework.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/common.h"
#include "backends/cpu/renderercpu.h"
#include "optimizers/poseOptimizer.h"

// Test to ensure PoseEstimator correctly computes the pose
TEST(RendererTestBase, ComputeMapFromInitial)
{
    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accError = 0;
    int framesProcessedCounter = 0;

    mapOptimizerCPU optimizer(w, h, false);
    renderCPU renderer(w, h);
    keyFrameCPU kframe;
    std::vector<frameCPU> frames;

    for (unsigned int i = 0; i < image_files.size(); i ++)
    {
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        cv::Mat gtDepth = cv::imread(depth_files[i], cv::IMREAD_GRAYSCALE);
        SE3f gtPose = poses[i].inverse();

        if (std::is_same<imageType, uchar>::value)
            image.convertTo(image, CV_8UC1);
        else if (std::is_same<imageType, int>::value)
            image.convertTo(image, CV_32SC1);
        else if (std::is_same<imageType, float>::value)
            image.convertTo(image, CV_32FC1);

        gtDepth.convertTo(gtDepth, CV_32FC1);
        gtDepth /= dataset.getDepthFactor();
        gtDepth *= 100.0;

        dataCPU<imageType> imageData(w, h, 0);
        imageData.set((imageType *)image.data);

        dataCPU<float> gtDepthData(w, h, 0);
        gtDepthData.set((float *)gtDepth.data);

        if(i == 0)
        {
            kframe = keyFrameCPU(imageData, vec2f(0.0, 0.0), gtPose, 1.0);
            kframe.initGeometryVerticallySmooth(cam);
            continue;
        }

        frameCPU frame(imageData, i);
        frame.setGlobalPose(gtPose);
        frame.setLocalPose(kframe.globalPoseToLocal(gtPose));

        float minViewAngle = M_PI;
        for(int i = 0; i < frames.size(); i++)
        {
            float viewAngle = kframe.meanViewAngle(frame.getLocalPose(), frames[i].getLocalPose());
            if(viewAngle < minViewAngle)
                minViewAngle = viewAngle;
        }

        if(minViewAngle < mesh_vo::last_min_angle)
            continue;

        frames.push_back(frame);
        if (frames.size() > mesh_vo::num_frames)
        {
            frames.erase(frames.begin());
        }
        else
        {
            continue;
        }

        int kframeIndex = frames.size() / 2;
        std::vector<frameCPU> oframes = frames;
        oframes.erase(oframes.begin() + kframeIndex);

        kframe = keyFrameCPU(frames[kframeIndex].getRawImage(0), vec2f(0.0, 0.0), frames[kframeIndex].getGlobalPose(), 1.0);
        kframe.initGeometryVerticallySmooth(cam);

        for (int i = 0; i < oframes.size(); i++)
        {
            oframes[i].setLocalPose(kframe.globalPoseToLocal(oframes[i].getGlobalPose()));
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
        {
            optimizer.init(oframes, kframe, cam, lvl);
            while (!optimizer.converged())
            {
                optimizer.step(oframes, kframe, cam, lvl);
            }
        }
        auto endTime = std::chrono::high_resolution_clock::now();

        /*
        dataMipMapCPU<float> error_buffer(w, h, -1.0);
        renderer.renderResidualParallel(kframe, oframes[oframes.size() - 1], error_buffer, cam, 1);
        show(error_buffer.get(1), "Error");

        dataMipMapCPU<float> depth_buffer(w, h, -1.0);
        renderer.renderDepthParallel(kframe, oframes[oframes.size() - 1].getLocalPose(), depth_buffer, cam, 1);
        dataCPU d = depth_buffer.get(1);
        d.invert();
        show(d, "Depth");
        */

        dataMipMapCPU<float> estMipMapDepthData(w, h, -1);
        renderer.renderDepthParallel(kframe, oframes[oframes.size() - 1].getLocalPose(), estMipMapDepthData, cam, 0);
        float error = computeImageError(estMipMapDepthData.get(0), gtDepthData);

        accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        accError += error;
        framesProcessedCounter++;
    }

    auto meanDuration = accProcessingTime.count() / framesProcessedCounter;
    float meanError = accError / framesProcessedCounter;
    std::cout << "Mean processing time " << meanDuration << " ms" << std::endl;
    std::cout << "Mean error " << meanError << " ms" << std::endl;

    // The test passes if the error is below the threshold
    EXPECT_LT(meanError, errorThreshold)
        << "mean error (" << meanError
        << ") exceeds the acceptable threshold (" << errorThreshold << ").";

    // EXPECT_LE(durationMs, acceptableTimeMs)
    //     << "Pose estimation took " << durationMs << "ms, which exceeds the acceptable threshold of "
    //     << acceptableTimeMs << "ms.";
}

TEST_F(RendererTestBase, ComputeMap)
{
    const int in_lvl = 0, out_lvl = 0;

    const long long acceptableTimeMs = 30;
    const float translationErrorThreshold = 1.5; // best = 0.0160271;
    const float rotationErrorThreshold = 0.0011;   // best = 0.00105154;

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accTranslationError = 0;
    float accRotationError = 0;
    int framesProcessedCounter = 0;

    std::vector<float> pos_buff_, tex_buff_, wei_buff_;
    CreateScreenQuad(pos_buff_, tex_buff_, wei_buff_);
    MeshCPU screen_mesh(pos_buff_, tex_buff_, wei_buff_);

    TextureCPU<float> kimage_cpu(w_, h_, 0.0f);
    UploadMatToTexture(kimage_cpu, 0, image_src_cv_);

    TextureCPU<float> kdepth_cpu(w_, h_, 0.0f);
    UploadMatToTexture(kdepth_cpu, 0, depth_src_cv_);

    MeshCPU mesh = CreateMesh(kdepth_cpu, cam_, 32);

    TextureCPU<Vec3> kdidxy_cpu(w_, h_, Vec3(0.0, 0.0, 0.0));

    DepthRendererCPU depth_renderer;
    ImageRendererCPU image_renderer;
    DIDxyRendererCPU didxy_renderer;
    ResidualRendererCPU residual_renderer;
    L2RendererCPU l2_renderer;

    NodataReducerCPU nodata_reducer;

    for (int lvl = 0; lvl < kdidxy_cpu.levels(); lvl++)
        didxy_renderer.Render(screen_mesh, lvl, lvl, kimage_cpu, kdidxy_cpu);

    KeyFrame kframe(Frame(kimage_cpu, kdidxy_cpu, 0, SE3(), pose_src_), mesh);

    PoseOptimizer optimizer(w_, h_, false);

    SE3 tracked_global_pose = kframe.frame().global_pose();

    TextureCPU<float> image_cpu(w_, h_, 0);
    TextureCPU<float> depth_cpu(w_, h_, 0);
    TextureCPU<Vec3> didxy_cpu(w_, h_, Vec3(0.0, 0.0, 0.0));
    TextureCPU<float> l2_texture(w_, h_, -1);

    for (unsigned int i = 1; i < image_files_.size(); i++)
    {
        std::cout << "Frame " << i << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[i], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[i], cv::IMREAD_GRAYSCALE);
        SE3 gt_pose = poses_[i].inverse();

        image_cv.convertTo(image_cv, CV_32FC1);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        //  gt_depth_cv /= dataset.getDepthFactor();
        //  gt_depth_cv *= 100.0;

        UploadMatToTexture(image_cpu, 0, image_cv);
        UploadMatToTexture(depth_cpu, 0, gt_depth_cv);

        for (int lvl = 0; lvl < kdidxy_cpu.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image_cpu, didxy_cpu);

        SE3 init_local_pose = kframe.globalPoseToLocal(tracked_global_pose);

        std::cout << "init_local_pose " << std::endl;
        std::cout << init_local_pose.translation() << std::endl;

        std::cout << "init_global_pose " << std::endl;
        std::cout << tracked_global_pose.translation() << std::endl;

        Frame frame(image_cpu, didxy_cpu, i, init_local_pose, tracked_global_pose);

        auto startTime = std::chrono::high_resolution_clock::now();
        for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
        {
            optimizer.init(frame, kframe, cam_, lvl);
            while (!optimizer.converged())
            {
                optimizer.step(frame, kframe, cam_, lvl);
            }
        }

        frame.global_pose() = kframe.localPoseToGlobal(frame.local_pose());

        tracked_global_pose = frame.global_pose();
        auto endTime = std::chrono::high_resolution_clock::now();

        std::array<double, 2> error = ComputeSE3Error(frame.global_pose(), gt_pose);

        accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        accTranslationError += error[0];
        accRotationError += error[1];
        framesProcessedCounter++;

        // change keyframe logic
        // float keyframeViewAngle = kframe.meanViewAngle(SE3(), frame.local_pose());

        image_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), image_cpu);
        Error nodata = nodata_reducer.reduce(1, image_cpu);
        float pnodata = nodata.getError() / image_cpu.size(1);
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent < mesh_vo::min_view_perc) // || keyframeViewAngle > mesh_vo::key_max_angle)
        {
            MeshCPU new_mesh = CreateMesh(depth_cpu, cam_, 32);
            kframe = KeyFrame(frame, new_mesh);

            frame.local_pose() = kframe.globalPoseToLocal(frame.global_pose());

            depth_renderer.Render(kframe.mesh(), SE3(), cam_, 1, depth_cpu);
            cv::Mat depth_mat = DownloadTexture(depth_cpu, 1, CV_32FC1);
            SaveDebugImageColor(depth_mat, "Depth keyframe_" + std::to_string(i) + ".png");

            image_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), image_cpu);
            cv::Mat image_mat = DownloadTexture(image_cpu, 1, CV_32FC1);
            SaveDebugImageColor(image_mat, "Frame keyframe_" + std::to_string(i) + ".png");

            Error nodata = nodata_reducer.reduce(1, image_cpu);
            float pnodata = nodata.getError() / image_cpu.size(1);
            float viewPercent = 1.0 - pnodata;

            std::cout << "new view percent " << viewPercent << std::endl;
        }

        residual_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), frame.image(), l2_texture);
        cv::Mat l2_mat = DownloadTexture(l2_texture, 1, CV_32FC1);
        SaveDebugImageColor(l2_mat, "l2_" + std::to_string(i) + ".png");
    }

    auto meanDuration = accProcessingTime.count() / framesProcessedCounter;
    float meanTranslationError = accTranslationError / framesProcessedCounter;
    float meanRotationError = accRotationError / framesProcessedCounter;
    std::cout << "Mean processing time " << meanDuration << " ms" << std::endl;
    std::cout << "Mean translation error " << meanTranslationError << " ms" << std::endl;
    std::cout << "Mean rotation error " << meanRotationError << " ms" << std::endl;

    // The test passes if the error is below the threshold
    EXPECT_LT(meanTranslationError, translationErrorThreshold)
        << "mean translation estimation error (" << meanTranslationError
        << ") exceeds the acceptable threshold (" << translationErrorThreshold << ").";

    EXPECT_LT(meanRotationError, rotationErrorThreshold)
        << "mean rotation estimation error (" << meanRotationError
        << ") exceeds the acceptable threshold (" << rotationErrorThreshold << ").";

    // EXPECT_LE(durationMs, acceptableTimeMs)
    //     << "Pose estimation took " << durationMs << "ms, which exceeds the acceptable threshold of "
    //     << acceptableTimeMs << "ms.";
}