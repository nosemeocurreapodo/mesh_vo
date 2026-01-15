#pragma once

#include <iostream>
#include <fstream>
#include <condition_variable>

//#include <pangolin/pangolin.h>

#include "core/types.h"
#include "core/mesh_helpers.h"

#include "backends/cpu/texturecpu.h"
#include "backends/cpu/meshcpu.h"
#include "backends/cpu/renderercpu.h"

#ifdef COMPILE_GL
#include "backends/gl/devicegl_glad.h"
#include "backends/gl/texturegl.h"
#include "backends/gl/meshgl.h"
#include "backends/gl/renderergl.h"
#endif

#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"

#include "optimizers/poseOptimizer.h"
#include "optimizers/poseExpOptimizer.h"
// #include "optimizers/poseVelOptimizerCPU.h"
#include "optimizers/mapOptimizer.h"
#include "optimizers/poseMapOptimizer.h"
#include "optimizers/poseExpMapOptimizer.h"
// #include "optimizers/intrinsicPoseMapOptimizerCPU.h"

// #include "visualizer/trayectoryPlotter.h"
// #include "visualizer/imagePlotter.h"
#include "utils/tictoc.h"

#include "tests/common/test_helpers.h"

template <typename T>
class ThreadSafeQueue
{
public:
    // Push an element into the queue
    void push(const T &value)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(value);
        }
        // cv_.notify_one();
        cv_.notify_all();
    }

    // Peek at the front element without removing it (blocks if the queue is empty)
    T peek()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]
                 { return !queue_.empty(); });
        return queue_.front();
    }

    // Try to peek an element; returns false if queue is empty
    bool try_peek(T &value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;
        value = queue_.front();
        return true;
    }

    // Pop an element from the queue (blocks if the queue is empty)
    T pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]
                 { return !queue_.empty(); });
        T value = queue_.front();
        queue_.pop();
        return value;
    }

    // Try to pop an element; returns false if queue is empty
    bool try_pop(T &value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;
        value = queue_.front();
        queue_.pop();
        return true;
    }

    // Check if the queue is empty
    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cv_;
};

class VisualOdometryThreaded
{
public:
    VisualOdometryThreaded(float fx, float fy, float cx, float cy, int width, int height, bool doMapping = true, bool doVisualization = true)
    {
#ifdef COMPILE_GL
        InitEGL();
#endif

        frameId_ = 0;

        width_ = width;
        height_ = height;

        cam_ = Camera(fy, fy, cx, cy, width, height);

        doMapping_ = doMapping;
        doVisualization_ = doVisualization;

        tLocalization_ = std::thread(&VisualOdometryThreaded::voThread, this);
        // tLocalization_ = std::thread(&VisualOdometryThreaded::localizationThread, this);
        // tMapping_ = std::thread(&VisualOdometryThreaded::mappingThread, this);

        // if (doVisualization)
        //     tVisualization_ = std::thread(&VisualOdometryThreaded::visualizationThread, this);
    }

    /*
    visualOdometryThreaded(dataCPU<imageType> &image, SE3f globalPose, cameraType _cam) : cam(_cam)
    {
        init(image, globalPose);
    }
    */

    ~VisualOdometryThreaded()
    {
        tLocalization_.join();
        tMapping_.join();
    }

    void flatInit(const ImageType *image_data)
    {
        std::vector<float> s_ver_buff;
        std::vector<int> s_idx_buff;
        CreateScreenQuad(s_ver_buff, s_idx_buff);
        Mesh screen_mesh(s_ver_buff, s_idx_buff, true, true, false);

        DIDxyRenderer didxy_renderer;

        Texture<ImageType> image_texture(width_, height_, -1, image_data);
        image_texture.generate_mipmaps(0);

        Texture<Vec3f> didxy_texture(width_, height_, Vec3f(0.0, 0.0, 0.0));
        for (int lvl = 0; lvl < didxy_texture.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image_texture, didxy_texture);

        std::vector<float> ver_buff;
        std::vector<int> idx_buff;

        CreateFlatMesh(0.5, 1.5, cam_, mesh_vo::mesh_width, ver_buff, idx_buff, true, true, false);
        Mesh mesh(ver_buff, idx_buff, true, true, false);

        KeyFrame kframe(image_texture, didxy_texture, SE3f(), mesh, 1.0, frameId_);

        kfQueue_.push(kframe);

        frameId_++;
    }

    void locAndMap(ImageType *image_data)
    {
        Texture<ImageType> image_texture(width_, height_, -1, image_data);
        image_texture.generate_mipmaps(0);
        while (iQueue_.empty() == false)
            ; // wait until the queue is empty
        iQueue_.push(image_texture);
    }

    bool isIdle()
    {
        return false;
    }

    KeyFrame getKeyframe()
    {
        return kfQueue_.peek();
    }

private:
    void localizationThread()
    {
        std::vector<float> s_ver_buff;
        std::vector<int> s_idx_buff;
        CreateScreenQuad(s_ver_buff, s_idx_buff);
        Mesh screen_mesh(s_ver_buff, s_idx_buff, true, true, false);

        DIDxyRenderer didxy_renderer;
        ResidualRenderer residual_renderer;
        ImageRenderer image_renderer;
        DepthRenderer depth_renderer;

        Texture<ImageType> image_texture(width_, height_, -1);
        Texture<Vec3f> didxy_texture(width_, height_, Vec3f(0.0, 0.0, 0.0));
        Texture<float> l2_texture(width_, height_, 0);
        Texture<float> depth_texture(width_, height_, 0);

        PoseExpOptimizer optimizer(width_, height_, true);
        // KeyFrame kframe(image_texture, didxy_texture, SE3f(), screen_mesh, 1.0, -1);

        KeyFrame kframe = kfQueue_.peek();

        SE3f lastLocalPose;
        SE3f lastLocalMovement;
        Vec2f lastLocalExposure(0.0, 0.0);

        tic_toc tt;

        while (true)
        {
            // check if there is a new kframe
            if (kfQueue_.try_peek(kframe))
            {
                lastLocalMovement = SE3f();
                lastLocalPose = SE3f();
            }

            Texture<ImageType> image_texture = iQueue_.pop();

            for (int lvl = 0; lvl < didxy_texture.levels(); lvl++)
                didxy_renderer.Render(screen_mesh, lvl, lvl, image_texture, didxy_texture);

            Frame frame(image_texture, didxy_texture, frameId_, kframe.id());

            // initialize the global and local pose
            frame.local_pose() = lastLocalMovement * lastLocalPose;
            frame.local_exposure() = lastLocalExposure;

            // this will update the local pose
            tt.tic();
            for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
            {
                optimizer.init(frame, kframe, cam_, lvl, lvl);
                // if (plotDebug)
                //{
                //     std::vector<dataCPU<float>> debugData = optimizer.getDebugData(frame, kframe, cam, 1);
                //     debugLocalizationQueue.push(debugData);
                // }
                while (true)
                {
                    optimizer.step(frame, kframe, cam_, lvl, lvl);
                    if (optimizer.converged())
                    {
                        // if (plotDebug)
                        //{
                        //     std::vector<dataCPU<float>> debugData = optimizer.getDebugData(frame, kframe, cam, 1);
                        //     debugLocalizationQueue.push(debugData);
                        // }
                        break;
                    }
                }
            }
            std::cout << "localization time " << tt.toc() << std::endl;

            lastLocalMovement = frame.local_pose() * lastLocalPose.inverse();
            lastLocalPose = frame.local_pose();
            lastLocalExposure = frame.local_exposure();

            fQueue_.push(frame);

            frameId_++;

            // Debug rendering
            depth_renderer.Render(kframe.mesh(),
                                  frame.local_pose(), cam_, 1, depth_texture);

            image_renderer.Render(kframe.mesh(),
                                  frame.local_pose(),
                                  frame.local_exposure(),
                                  cam_,
                                  1, 1,
                                  kframe.image(),
                                  image_texture);

            residual_renderer.Render(kframe.mesh(),
                                     frame.local_pose(),
                                     frame.local_exposure(),
                                     cam_,
                                     1, 1,
                                     kframe.image(), frame.image(), l2_texture);

            cv::Mat depth_mat = DownloadTextureToMat(depth_texture, 1);
            SaveDebugImage(depth_mat, "loc_depth_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

            cv::Mat image_mat = DownloadTextureToMat(image_texture, 1);
            SaveDebugImage(image_mat, "loc_image_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

            cv::Mat frame_mat = DownloadTextureToMat(frame.image(), 1);
            SaveDebugImage(frame_mat, "loc_frame_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

            cv::Mat l2_mat = DownloadTextureToMat(l2_texture, 1);
            SaveDebugImage(l2_mat, "loc_l2_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");
        }
    }

    void mappingThread()
    {
        std::vector<float> ver_buff;
        std::vector<int> idx_buff;

        Texture<ImageType> image_texture(width_, height_, -1);
        Texture<float> l2_texture(width_, height_, 0);
        Texture<float> depth_texture(width_, height_, -1);

        ImageRenderer image_renderer;
        ResidualRenderer residual_renderer;
        DepthRenderer depth_renderer;

        NodataReducerCPU nodata_reducer;

        PoseExpMapOptimizer optimizer(width_, height_, true);

        std::vector<Frame> frameStack;

        KeyFrame kframe = kfQueue_.peek();

        bool initial_kframe = true;

        tic_toc tt;

        while (true)
        {
            Frame frame = fQueue_.pop();

            assert(frame.keyframe_id() == kframe.id());

            float lastMinViewAngle = M_PI;
            for (Frame f : frameStack)
            {
                float lastViewAngle = kframe.meanViewAngle(f.local_pose(), frame.local_pose());
                if (lastViewAngle < lastMinViewAngle)
                    lastMinViewAngle = lastViewAngle;
            }

            if (lastMinViewAngle > mesh_vo::last_min_angle || initial_kframe)
            {
                frameStack.push_back(frame);
                if (frameStack.size() > mesh_vo::num_frames)
                    frameStack.erase(frameStack.begin());
            }

            if (frameStack.size() < mesh_vo::num_frames)
                continue;

            float keyframeViewAngle = kframe.meanViewAngle(SE3f(), frame.local_pose());

            depth_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, depth_texture);

            Error nodata;
            nodata_reducer.reduce(1, depth_texture, nodata);
            float pnodata = nodata.getError() / (depth_texture.width(1) * depth_texture.height(1));
            float viewPercent = 1.0 - pnodata;

            if (viewPercent > mesh_vo::min_view_perc && keyframeViewAngle < mesh_vo::key_max_angle && !initial_kframe)
                continue;

            // select new keyframe
            // int newKeyframeIndex = 0;
            int newKeyframeIndex = int(frameStack.size() / 2);
            // int newKeyframeIndex = int(goodFrames.size() - 1);
            Frame newKeyframe = frameStack[newKeyframeIndex];

            // CreateMesh(gt_depth_texture, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, false);
            CreateFlatMesh(0.5, 1.5, cam_, mesh_vo::mesh_width, ver_buff, idx_buff, true, true, false);

            Mesh mesh(ver_buff, idx_buff, true, true, false);

            SE3f reference_pose = newKeyframe.local_pose().inverse();
            SE3f global_pose = kframe.localPoseToGlobal(newKeyframe.local_pose());
            float global_scale = kframe.getGlobalScale();

            kframe = KeyFrame(newKeyframe.image(), newKeyframe.didxy(), global_pose, mesh, global_scale, newKeyframe.id());

            for (int i = 0; i < frameStack.size(); i++)
            {
                frameStack[i].local_pose() = frameStack[i].local_pose() * reference_pose;
            }

            std::vector<Frame> oframes = frameStack;
            // oframes.erase(oframes.begin() + newKeyframeIndex);

            tt.tic();
            for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
            {
                optimizer.init(oframes, kframe, cam_, lvl, lvl);
                // if (plotDebug)
                //{
                //     std::vector<dataCPU<float>> debugData = optimizer.getDebugData(keyframes, kframe, cam, 1);
                //     debugMappingQueue.push(debugData);
                // }
                while (true)
                {
                    optimizer.step(oframes, kframe, cam_, lvl, lvl);
                    if (optimizer.converged())
                    {
                        // if (plotDebug)
                        //{
                        //     std::vector<dataCPU<float>> debugData = optimizer.getDebugData(keyframes, kframe, cam, 1);
                        //     debugMappingQueue.push(debugData);
                        // }
                        break;
                    }
                }
            }
            std::cout << "mapping time " << tt.toc() << std::endl;

            // update the global poses
            for (size_t i = 0; i < oframes.size(); i++)
            {
                for (size_t j = 0; j < frameStack.size(); j++)
                {
                    if (oframes[i].id() == frameStack[j].id())
                    {
                        frameStack[j].local_pose() = oframes[i].local_pose();
                        frameStack[j].local_exposure() = oframes[i].local_exposure();
                    }
                }
            }

            kfQueue_.push(kframe);

            // Debug rendering
            depth_renderer.Render(kframe.mesh(),
                                  frame.local_pose(), cam_, 1, depth_texture);
            cv::Mat depth_mat = DownloadTextureToMat(depth_texture, 1);
            SaveDebugImage(depth_mat, "map_depth_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

            image_renderer.Render(kframe.mesh(),
                                  frame.local_pose(),
                                  frame.local_exposure(),
                                  cam_,
                                  1, 1,
                                  kframe.image(),
                                  image_texture);
            cv::Mat image_mat = DownloadTextureToMat(image_texture, 1);
            SaveDebugImage(image_mat, "map_image_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

            cv::Mat frame_mat = DownloadTextureToMat(frame.image(), 1);
            SaveDebugImage(frame_mat, "map_frame_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

            for (Frame f : frameStack)
            {
                residual_renderer.Render(kframe.mesh(),
                                         f.local_pose(),
                                         f.local_exposure(),
                                         cam_,
                                         1, 1,
                                         kframe.image(), f.image(), l2_texture);

                cv::Mat l2_mat = DownloadTextureToMat(l2_texture, 1);
                SaveDebugImage(l2_mat, "map_l2_" + std::to_string(kframe.id()) + "_" + std::to_string(f.id()) + ".png");
            }
        }
    }

    void voThread()
    {
        std::vector<float> s_ver_buff;
        std::vector<int> s_idx_buff;
        CreateScreenQuad(s_ver_buff, s_idx_buff);
        Mesh screen_mesh(s_ver_buff, s_idx_buff, true, true, false);

        DIDxyRenderer didxy_renderer;
        DepthRenderer depth_renderer;
        ImageRenderer image_renderer;
        ResidualRenderer residual_renderer;

        NodataReducerCPU nodata_reducer;

        PoseExpOptimizer poseOptimizer(width_, height_, true);
        PoseExpMapOptimizer poseMapOptimizer(width_, height_, true);

        Texture<ImageType> image_texture(width_, height_, -1);
        Texture<Vec3f> didxy_texture(width_, height_, Vec3f(0.0, 0.0, 0.0));
        Texture<float> depth_texture(width_, height_, -1);
        Texture<float> l2_texture(width_, height_, 0);

        std::vector<Frame> frameStack;
        std::vector<Frame> oframes;

        KeyFrame kframe = kfQueue_.peek();

        tic_toc tt;

        SE3f lastLocalPose;
        SE3f lastLocalMovement;
        Vec2f lastLocalExposure(0.0, 0.0);

        while (true)
        {
            // if there is a new image, compute its pose
            if (iQueue_.try_pop(image_texture))
            {
                for (int lvl = 0; lvl < didxy_texture.levels(); lvl++)
                    didxy_renderer.Render(screen_mesh, lvl, lvl, image_texture, didxy_texture);

                Frame frame(image_texture, didxy_texture, frameId_, kframe.id());

                // initialize the global and local pose
                frame.local_pose() = lastLocalMovement * lastLocalPose;
                frame.local_exposure() = lastLocalExposure;

                // this will update the local pose
                tt.tic();
                for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
                {
                    poseOptimizer.init(frame, kframe, cam_, lvl, lvl);
                    // if (plotDebug)
                    //{
                    //      std::vector<dataCPU<float>> debugData = poseOptimizer.getDebugData(frame, kframe, cam, 1);
                    //      debugLocalizationQueue.push(debugData);
                    //  }
                    while (true)
                    {
                        poseOptimizer.step(frame, kframe, cam_, lvl, lvl);
                        if (poseOptimizer.converged())
                        {
                            // if (doVisualization)
                            //{
                            //     std::vector<dataCPU<float>> debugData = poseOptimizer.getDebugData(frame, kframe, cam, 1);
                            //     debugLocalizationQueue.push(debugData);
                            // }
                            break;
                        }
                    }
                }
                // if (plotDebug)
                //{
                //     std::vector<dataCPU<float>> debugData = poseOptimizer.getDebugData(frame, kframe, cam, 1);
                //     debugLocalizationQueue.push(debugData);
                // }

                std::cout << "localization time " << tt.toc() << std::endl;

                lastLocalMovement = frame.local_pose() * lastLocalPose.inverse();
                lastLocalPose = frame.local_pose();
                lastLocalExposure = frame.local_exposure();

                frameId_++;

                // Debug rendering
                // depth_renderer.Render(kframe.mesh(),
                //                      frame.local_pose(), cam_, 1, depth_texture);

                // image_renderer.Render(kframe.mesh(),
                //                       frame.local_pose(),
                //                       frame.local_exposure(),
                //                       cam_,
                //                       1, 1,
                //                       kframe.image(),
                //                       image_texture);

                residual_renderer.Render(kframe.mesh(),
                                         frame.local_pose(),
                                         frame.local_exposure(),
                                         cam_,
                                         1, 1,
                                         kframe.image(), frame.image(), l2_texture);

                // cv::Mat depth_mat = DownloadTextureToMat(depth_texture, 1);
                // SaveDebugImage(depth_mat, "loc_depth_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

                // cv::Mat image_mat = DownloadTextureToMat(image_texture, 1);
                // SaveDebugImage(image_mat, "loc_image_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

                // cv::Mat frame_mat = DownloadTextureToMat(frame.image(), 1);
                // SaveDebugImage(frame_mat, "loc_frame_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

                cv::Mat l2_mat = DownloadTextureToMat(l2_texture, 1);
                SaveDebugImage(l2_mat, "loc_l2_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

                // Choose wether to save the frame or not
                float lastMinViewAngle = M_PI;
                for (Frame f : frameStack)
                {
                    float lastViewAngle = kframe.meanViewAngle(f.local_pose(), frame.local_pose());
                    if (lastViewAngle < lastMinViewAngle)
                        lastMinViewAngle = lastViewAngle;
                }

                if (lastMinViewAngle > mesh_vo::last_min_angle || kframe.id() == 0)
                {
                    frameStack.push_back(frame);
                    if (frameStack.size() > mesh_vo::num_frames)
                        frameStack.erase(frameStack.begin());
                }
                else
                    continue;

                // Choose wether to get a new keyframe
                if (frameStack.size() < mesh_vo::num_frames)
                    continue;

                float keyframeViewAngle = kframe.meanViewAngle(SE3f(), frame.local_pose());

                depth_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, depth_texture);

                Error nodata;
                nodata_reducer.reduce(1, depth_texture, nodata);
                float pnodata = nodata.getError() / (depth_texture.width(1) * depth_texture.height(1));
                float viewPercent = 1.0 - pnodata;

                if (kframe.id() != 0 && viewPercent > mesh_vo::min_view_perc && keyframeViewAngle < mesh_vo::key_max_angle)
                    continue;

                // save last keyframe, we wont be updating it anymore
                kfQueue_.push(kframe);

                // select new keyframe
                // int newKeyframeIndex = 0;
                int newKeyframeIndex = int(frameStack.size() / 2);
                // int newKeyframeIndex = int(frameStack.size() - 2);
                Frame newKeyframe = frameStack[newKeyframeIndex];

                std::vector<float> ver_buff;
                std::vector<int> idx_buff;
                // CreateMesh(gt_depth_texture, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, false);
                CreateFlatMesh(0.5, 1.5, cam_, mesh_vo::mesh_width, ver_buff, idx_buff, true, true, false);

                Mesh mesh(ver_buff, idx_buff, true, true, false);

                SE3f reference_pose = newKeyframe.local_pose().inverse();
                SE3f global_pose = kframe.localPoseToGlobal(newKeyframe.local_pose());
                float global_scale = kframe.getGlobalScale();

                kframe = KeyFrame(newKeyframe.image(), newKeyframe.didxy(), global_pose, mesh, global_scale, newKeyframe.id());

                // initialize the local poses
                for (size_t i = 0; i < frameStack.size(); i++)
                {
                    frameStack[i].local_pose() = frameStack[i].local_pose() * reference_pose;
                }

                oframes = frameStack;
                oframes.erase(oframes.begin() + newKeyframeIndex);

                // init the posemapoptimizer
                poseMapOptimizer.init(oframes, kframe, cam_, mesh_vo::mapping_fin_lvl, mesh_vo::mapping_fin_lvl);
                while (poseMapOptimizer.converged() == false)
                {
                    poseMapOptimizer.step(oframes, kframe, cam_, mesh_vo::mapping_fin_lvl, mesh_vo::mapping_fin_lvl);
                }

                for (size_t i = 0; i < oframes.size(); i++)
                {
                    for (size_t j = 0; j < frameStack.size(); j++)
                    {
                        if (oframes[i].id() == frameStack[j].id())
                        {
                            frameStack[j].local_pose() = oframes[i].local_pose();
                            frameStack[j].local_exposure() = oframes[i].local_exposure();
                        }
                    }
                }

                lastLocalPose = SE3f();
                lastLocalMovement = SE3f();
                lastLocalExposure = Vec2f(0.0, 0.0);

                // Debug rendering
                depth_renderer.Render(kframe.mesh(),
                                      SE3f(), cam_, 1, depth_texture);
                cv::Mat map_depth_mat = DownloadTextureToMat(depth_texture, 1);
                SaveDebugImage(map_depth_mat, "map_depth_" + std::to_string(kframe.id()) + ".png");

                // image_renderer.Render(kframe.mesh(),
                //                       frame.local_pose(),
                //                       frame.local_exposure(),
                //                       cam_,
                //                       1, 1,
                //                       kframe.image(),
                //                       image_texture);
                // cv::Mat map_image_mat = DownloadTextureToMat(image_texture, 1);
                // SaveDebugImage(map_image_mat, "map_image_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

                // cv::Mat map_frame_mat = DownloadTextureToMat(frame.image(), 1);
                // SaveDebugImage(frame_mat, "map_frame_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

                for (Frame f : frameStack)
                {
                    residual_renderer.Render(kframe.mesh(),
                                             f.local_pose(),
                                             f.local_exposure(),
                                             cam_,
                                             1, 1,
                                             kframe.image(), f.image(), l2_texture);

                    cv::Mat l2_mat = DownloadTextureToMat(l2_texture, 1);
                    SaveDebugImage(l2_mat, "map_l2_" + std::to_string(kframe.id()) + "_" + std::to_string(f.id()) + ".png");
                }
            }
            /*
            if (oframes.size() < 1)
                continue;

            // this will update the local pose and the local map
            tt.tic();
            poseMapOptimizer.step(oframes, kframe, cam_, mesh_vo::mapping_fin_lvl, mesh_vo::mapping_fin_lvl);
            std::cout << "mapping time " << tt.toc() << std::endl;

            // update the global poses
            for (size_t i = 0; i < oframes.size(); i++)
            {
                for (size_t j = 0; j < frameStack.size(); j++)
                {
                    if (oframes[i].id() == frameStack[j].id())
                    {
                        frameStack[j].local_pose() = oframes[i].local_pose();
                        frameStack[j].local_exposure() = oframes[i].local_exposure();
                    }
                }
            }
            */
        }
    }

    /*
    // Visualization thread using Pangolin
    int visualizationThread()
    {
        pangolin::CreateWindowAndBind("Main", 640, 480);
        glEnable(GL_DEPTH_TEST);

        // Define Projection and initial ModelView matrix
        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 100),
            pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));

        // Create Interactive View in window
        pangolin::Handler3D handler(s_cam);
        pangolin::View &d_cam = pangolin::CreateDisplay()
                                    .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                    .SetHandler(&handler);

        std::vector<SE3f> poses;

        ImageRendererGL image_renderer;
        trayectoryPlotter trayPlotter;
        std::vector<imagePlotter> imgPosePlotter;
        imgPosePlotter.push_back(imagePlotter(0, 0)); // keyframe
        imgPosePlotter.push_back(imagePlotter(0, 1)); // depth
        imgPosePlotter.push_back(imagePlotter(0, 2)); // frame
        imgPosePlotter.push_back(imagePlotter(0, 3)); // errpr

        std::vector<imagePlotter> imgMapPlotter;
        imgMapPlotter.push_back(imagePlotter(1, 0)); // keyframe
        imgMapPlotter.push_back(imagePlotter(1, 1)); // depth
        imgMapPlotter.push_back(imagePlotter(2, 0)); // frame
        imgMapPlotter.push_back(imagePlotter(2, 1)); // error
        imgMapPlotter.push_back(imagePlotter(3, 0)); // frame
        imgMapPlotter.push_back(imagePlotter(3, 1)); // error

        geomPlotter.compileShaders();
        trayPlotter.compileShaders();
        for (imagePlotter &imgPlotter : imgPosePlotter)
            imgPlotter.compileShaders();
        for (imagePlotter &imgPlotter : imgMapPlotter)
            imgPlotter.compileShaders();

        Keyframe kframe = kframe_;
        Frame frame;

        while (!pangolin::ShouldQuit())
        {
            if (kfQueue_.try_pop(kframe))
            {
                kframe.scaleVerticesAndWeights(1.0 / kframe.getGlobalScale());
                kframe.getGeometry().transform(kframe.getGlobalPose().inverse());
                poses.push_back(kframe.getGlobalPose());
                trayPlotter.setBuffers(poses, cam_);
            }

            if (!debugLocalizationQueue_.empty())
            {
                std::vector<dataCPU<float>> debugLoc = debugLocalizationQueue.pop();

                for (int i = 0; i < debugLoc.size(); i++)
                {
                    if (i >= imgPosePlotter.size())
                        break;
                    vec2f minMax = debugLoc[i].getMinMax();
                    debugLoc[i].normalize(minMax(0), minMax(1));
                    imgPosePlotter[i].setBuffers(debugLoc[i]);
                }
            }

            if (!debugMappingQueue_.empty())
            {
                std::vector<dataCPU<float>> debugLoc = debugMappingQueue.pop();

                for (int i = 0; i < debugLoc.size(); i++)
                {
                    if (i >= imgMapPlotter.size())
                        break;
                    vec2f minMax = debugLoc[i].getMinMax();
                    debugLoc[i].normalize(minMax(0), minMax(1));
                    imgMapPlotter[i].setBuffers(debugLoc[i]);
                }
            }


            //if (fQueue.try_peek(frame))
           // {
            //    poses.push_back(frame.getGlobalPose());
            //    plotter.setBuffers(poses, cam);
            //}


            // Clear screen and activate view to render into
            glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);
            pangolin::OpenGlMatrix mvp = s_cam.GetProjectionModelViewMatrix();

            geomPlotter.draw(mvp);
            trayPlotter.draw(mvp);
            for (imagePlotter &iPlotter : imgPosePlotter)
                iPlotter.draw();
            for (imagePlotter &iPlotter : imgMapPlotter)
                iPlotter.draw();

            // Swap frames and Process Events
            pangolin::FinishFrame();
        }

        return 0;
    }
    */

    std::thread tLocalization_;
    std::thread tMapping_;
    std::thread tVisualization_;

    ThreadSafeQueue<Texture<ImageType>> iQueue_;
    ThreadSafeQueue<Frame> fQueue_;
    ThreadSafeQueue<KeyFrame> kfQueue_;

    Camera cam_;
    int width_;
    int height_;
    int frameId_;

    bool doMapping_;
    bool doVisualization_;
};