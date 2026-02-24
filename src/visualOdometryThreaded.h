#pragma once

#include <iostream>
#include <fstream>
#include <condition_variable>

// #include <pangolin/pangolin.h>

#include "mpdr/common/types.h"
#include "mpdr/common/mesh_helpers.h"
#include "mpdr/common/helpers.h"

#include "mpdr/backends/cpu/texturecpu.h"
#include "mpdr/backends/cpu/meshcpu.h"
#include "mpdr/backends/cpu/renderercpu.h"

#ifdef COMPILE_GL
#include "mpdr/backends/gl/devicegl_glad.h"
#include "mpdr/backends/gl/texturegl.h"
#include "mpdr/backends/gl/meshgl.h"
#include "mpdr/backends/gl/renderergl.h"
#endif

#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"

#include "poseEstimator.h"
#include "poseDepthEstimator.h"

// #include "visualizer/trayectoryPlotter.h"
// #include "visualizer/imagePlotter.h"
#include "utils/tictoc.h"

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
        Mesh screen_mesh;
        CreateScreenQuad(screen_mesh);

        DIDxyRenderer didxy_renderer;

        Texture<ImageType> image_texture(width_, height_, -1, image_data);
        image_texture.generate_mipmaps(0);

        Texture<Vec3f> didxy_texture(width_, height_, Vec3f(0.0, 0.0, 0.0));
        for (int lvl = 0; lvl < didxy_texture.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image_texture, didxy_texture);

        Mesh mesh;
        CreateFlatMesh(0.5, 1.5, cam_, mesh_vo::mesh_width, mesh);

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
        Mesh screen_mesh;
        CreateScreenQuad(screen_mesh);

        DIDxyRenderer didxy_renderer;
        ImageRenderer image_renderer;
        DepthRenderer depth_renderer;

        Texture<ImageType> image_texture(width_, height_, -1);
        Texture<Vec3f> didxy_texture(width_, height_, Vec3f(0.0, 0.0, 0.0));
        Texture<float> depth_texture(width_, height_, 0);

        PoseEstimator estimator(width_, height_, true);
        // KeyFrame kframe(image_texture, didxy_texture, SE3f(), screen_mesh, 1.0, -1);

        KeyFrame kframe = kfQueue_.peek();

        tic_toc tt;

        while (true)
        {
            // check if there is a new kframe
            if (kfQueue_.try_peek(kframe))
            {
                // estimator.changeKeyframe()
            }

            Texture<ImageType> image_texture = iQueue_.pop();

            for (int lvl = 0; lvl < didxy_texture.levels(); lvl++)
                didxy_renderer.Render(screen_mesh, lvl, lvl, image_texture, didxy_texture);

            Frame frame(image_texture, didxy_texture, frameId_, kframe.id());

            tt.tic();
            estimator.guess(frame);
            estimator.estimate(frame, kframe, cam_);
            std::cout << "localization time " << tt.toc() << std::endl;

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

            cv::Mat depth_mat = DownloadTextureToMat(depth_texture, 1);
            SaveDebugImage(depth_mat, "loc_depth_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

            cv::Mat image_mat = DownloadTextureToMat(image_texture, 1);
            SaveDebugImage(image_mat, "loc_image_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");

            cv::Mat frame_mat = DownloadTextureToMat(frame.image(), 1);
            SaveDebugImage(frame_mat, "loc_frame_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + ".png");
        }
    }

    void mappingThread()
    {
        Texture<ImageType> image_texture(width_, height_, -1);
        Texture<float> depth_texture(width_, height_, -1);

        ImageRenderer image_renderer;
        DepthRenderer depth_renderer;

        NodataReducerCPU nodata_reducer;

        PoseDepthEstimator estimator(width_, height_, true);

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
                float lastViewAngle = kframe.meanViewAngle(f.local_pose(), frame.local_pose(), cam_);
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

            float keyframeViewAngle = kframe.meanViewAngle(SE3f(), frame.local_pose(), cam_);

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

            estimator.changeKeyframe(newKeyframe, frameStack, kframe, cam_);

            std::vector<Frame> oframes = frameStack;
            // oframes.erase(oframes.begin() + newKeyframeIndex);

            tt.tic();
            estimator.estimate(frameStack, kframe, cam_);
            std::cout << "mapping time " << tt.toc() << std::endl;

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
                image_renderer.Render(kframe.mesh(),
                                      f.local_pose(),
                                      f.local_exposure(),
                                      cam_,
                                      1, 1,
                                      kframe.image(), image_texture);

                cv::Mat image_mat = DownloadTextureToMat(image_texture, 1);
                cv::Mat ref_mat = DownloadTextureToMat(f.image(), 1);
                cv::Mat l2_mat = ref_mat - image_mat;
                SaveDebugImage(l2_mat, "map_l2_" + std::to_string(kframe.id()) + "_" + std::to_string(f.id()) + ".png");
            }
        }
    }

    void voThread()
    {
        Mesh screen_mesh;
        CreateScreenQuad(screen_mesh);

        DIDxyRenderer didxy_renderer;
        DepthRenderer depth_renderer;
        ImageRenderer image_renderer;

        NodataReducerCPU nodata_reducer;

        PoseOptimizer poseOptimizer(width_, height_, false);
        PoseDepthOptimizer poseMapOptimizer(width_, height_, true);

        Texture<ImageType> image_texture(width_, height_, -1);
        Texture<Vec3f> didxy_texture(width_, height_, Vec3f(0.0, 0.0, 0.0));
        Texture<float> depth_texture(width_, height_, -1);

        std::vector<Frame> frameStack;
        std::vector<Frame> oframes;

        KeyFrame kframe = kfQueue_.peek();

        tic_toc tt;

        SE3f lastLocalPose;
        SE3f lastLocalMovement;
        Vec2f lastLocalExposure(0.0, 0.0);

        cv::Mat image_mat;
        cv::Mat depth_mat;
        cv::Mat ref_mat;
        cv::Mat l2_mat;

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

                ////////////////////// Debug ///////////////////
                image_renderer.Render(kframe.mesh(),
                                      frame.local_pose(),
                                      frame.local_exposure(),
                                      cam_,
                                      1, 1,
                                      kframe.image(), image_texture);

                image_mat = DownloadTextureToMat(image_texture, 1);
                ref_mat = DownloadTextureToMat(frame.image(), 1);
                l2_mat = ref_mat - image_mat;
                SaveDebugImage(l2_mat, "loc_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + "_init_l2.png");
                /////////////

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

                /////////// Debug /////////////
                image_renderer.Render(kframe.mesh(),
                                      frame.local_pose(),
                                      frame.local_exposure(),
                                      cam_,
                                      1, 1,
                                      kframe.image(), image_texture);

                image_mat = DownloadTextureToMat(image_texture, 1);
                l2_mat = ref_mat - image_mat;
                SaveDebugImage(l2_mat, "loc_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + "_opt_l2.png");

                // Choose wether to save the frame or not
                float lastMinViewAngle = M_PI;
                for (Frame f : frameStack)
                {
                    float lastViewAngle = kframe.meanViewAngle(f.local_pose(), frame.local_pose(), cam_);
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

                float keyframeViewAngle = kframe.meanViewAngle(SE3f(), frame.local_pose(), cam_);

                Error nodata;
                nodata_reducer.reduce(1, image_texture, nodata);
                float pnodata = nodata.getError() / (image_texture.width(1) * image_texture.height(1));
                float viewPercent = 1.0 - pnodata;

                if (kframe.id() != 0 && viewPercent > mesh_vo::min_view_perc && keyframeViewAngle < mesh_vo::key_max_angle)
                    continue;

                std::cout << "Creating new keyframe because viewpercent: " << viewPercent << " keyframeViewAngle: " << keyframeViewAngle << std::endl;

                // save last keyframe, we wont be updating it anymore
                kfQueue_.push(kframe);

                // select new keyframe
                // int newKeyframeIndex = 0;
                int newKeyframeIndex = int(frameStack.size() / 2);
                // int newKeyframeIndex = int(frameStack.size() - 2);
                Frame newKeyframe = frameStack[newKeyframeIndex];

                depth_renderer.Render(kframe.mesh(),
                                      newKeyframe.local_pose(),
                                      cam_,
                                      0,
                                      depth_texture);

                Mesh mesh;
                CreateMesh(depth_texture.MapRead(0).data(),
                           cam_,
                           depth_texture.width(0),
                           depth_texture.height(0),
                           mesh_vo::mesh_width,
                           mesh);
                // CreateFlatMesh(0.5, 1.5, cam_, mesh_vo::mesh_width, ver_buff, idx_buff, true, false, false);

                SE3f reference_pose = newKeyframe.local_pose().inverse();
                SE3f global_pose = kframe.localPoseToGlobal(newKeyframe.local_pose());
                float global_scale = kframe.getGlobalScale();

                kframe = KeyFrame(newKeyframe.image(), newKeyframe.didxy(), global_pose, mesh, global_scale, newKeyframe.id());

                lastLocalPose = lastLocalPose * reference_pose;
                lastLocalMovement = SE3f(); // tracked_local_movement * reference_pose;
                lastLocalExposure = Vec2f(0.0, 0.0);

                frame.local_pose() = frame.local_pose() * reference_pose;

                // initialize the local poses
                for (size_t i = 0; i < frameStack.size(); i++)
                {
                    frameStack[i].local_pose() = frameStack[i].local_pose() * reference_pose;
                }

                /////////////////// Debug ///////////////////
                depth_renderer.Render(kframe.mesh(),
                                      SE3f(), cam_, 1, depth_texture);
                depth_mat = DownloadTextureToMat(depth_texture, 1);
                SaveDebugImage(depth_mat, "map_" + std::to_string(kframe.id()) + "_init_depth.png");

                for (Frame f : frameStack)
                {
                    image_renderer.Render(kframe.mesh(),
                                          f.local_pose(),
                                          f.local_exposure(),
                                          cam_,
                                          1, 1,
                                          kframe.image(), image_texture);

                    image_mat = DownloadTextureToMat(image_texture, 1);
                    ref_mat = DownloadTextureToMat(f.image(), 1);
                    l2_mat = ref_mat - image_mat;
                    SaveDebugImage(l2_mat, "map_" + std::to_string(kframe.id()) + "_" + std::to_string(f.id()) + "_ini_l2.png");
                }
                ///////////////////////////

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

                frameStack[newKeyframeIndex].local_pose() = SE3f();

                float mean_depth = kframe.meanDepth();
                std::cout << "Mean depth: " << mean_depth << std::endl;
                kframe.scaleMesh(mean_depth / mesh_vo::mapping_mean_depth);

                for (size_t i = 0; i < frameStack.size(); i++)
                {
                    frameStack[i].scalePose(mean_depth / mesh_vo::mapping_mean_depth);
                }

                lastLocalPose.translation() /= (mean_depth / mesh_vo::mapping_mean_depth);
                lastLocalMovement.translation() /= (mean_depth / mesh_vo::mapping_mean_depth);

                /////////////////// Debug ///////////////////
                depth_renderer.Render(kframe.mesh(),
                                      SE3f(), cam_, 1, depth_texture);
                depth_mat = DownloadTextureToMat(depth_texture, 1);
                SaveDebugImage(depth_mat, "map_" + std::to_string(kframe.id()) + "_opt_depth.png");

                for (Frame f : frameStack)
                {
                    image_renderer.Render(kframe.mesh(),
                                          f.local_pose(),
                                          f.local_exposure(),
                                          cam_,
                                          1, 1,
                                          kframe.image(), image_texture);

                    image_mat = DownloadTextureToMat(image_texture, 1);
                    ref_mat = DownloadTextureToMat(f.image(), 1);
                    l2_mat = ref_mat - image_mat;
                    SaveDebugImage(l2_mat, "map_" + std::to_string(kframe.id()) + "_" + std::to_string(f.id()) + "_opt_l2.png");
                }
                ///////////////////////////
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