#pragma once

#include <iostream>
#include <fstream>

#include <pangolin/pangolin.h>

#include "cpu/frameCPU.h"
#include "cpu/keyFrameCPU.h"

#include "optimizers/poseOptimizerCPU.h"
#include "optimizers/poseVelOptimizerCPU.h"
#include "optimizers/mapOptimizerCPU.h"
#include "optimizers/poseMapOptimizerCPU.h"
#include "optimizers/intrinsicPoseMapOptimizerCPU.h"

#include "visualizer/geometryPlotter.h"
#include "visualizer/trayectoryPlotter.h"
#include "visualizer/imagePlotter.h"
#include "cpu/OpenCVDebug.h"
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
        cv_.notify_one();
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

class visualOdometryThreaded
{
public:
    visualOdometryThreaded(cameraType _cam) : cam(_cam)
    {
    }

    visualOdometryThreaded(dataCPU<imageType> &image, SE3f globalPose, cameraType _cam) : cam(_cam)
    {
        init(image, globalPose);
    }

    ~visualOdometryThreaded()
    {
        tLocalization.join();
        tMapping.join();
    }

    void init(dataCPU<imageType> &image, SE3f globalPose)
    {
        initialKeyframe = keyFrameCPU(image, vec2f(0.0, 0.0), globalPose, 1.0);
        initialKeyframe.initGeometryVerticallySmooth(cam);

        width = image.width;
        height = image.height;
        frameId = 0;

        tLocalization = std::thread(&visualOdometryThreaded::voThread, this);
        // tLocalization = std::thread(&visualOdometryThreaded::localizationThread, this);
        // tMapping = std::thread(&visualOdometryThreaded::mappingThread, this);
        tVisualization = std::thread(&visualOdometryThreaded::visualizationThread, this);

        plotDebug = true;
    }

    void locAndMap(dataCPU<imageType> &image)
    {
        iQueue.push(image);
    }

    bool isIdle()
    {
        return false;
    }

    keyFrameCPU getKeyframe()
    {
        return kfQueue.peek();
    }

private:
    void localizationThread()
    {
        poseOptimizerCPU optimizer(width, height);
        keyFrameCPU kframe = initialKeyframe;

        SE3f lastGlobalPose;
        SE3f lastGlobalMovement;

        tic_toc tt;

        while (true)
        {
            // keep only the last keyframe
            while (true)
            {
                if (!kfQueue.try_pop(kframe))
                    break;
            }

            dataCPU<imageType> image = iQueue.pop();

            // keep on reading and keep the last image
            while (true)
            {
                if (!iQueue.try_pop(image))
                    break;
            }

            frameCPU frame(image, frameId);

            // initialize the global and local pose
            frame.setGlobalPose(lastGlobalMovement * lastGlobalPose);
            frame.setLocalPose(kframe.globalPoseToLocal(frame.getGlobalPose()));

            // this will update the local pose
            tt.tic();
            for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
            {
                optimizer.init(frame, kframe, cam, lvl);
                // if (plotDebug)
                //{
                //     std::vector<dataCPU<float>> debugData = optimizer.getDebugData(frame, kframe, cam, 1);
                //     debugLocalizationQueue.push(debugData);
                // }
                while (true)
                {
                    optimizer.step(frame, kframe, cam, lvl);
                    if (optimizer.converged())
                    {
                        if (plotDebug)
                        {
                            std::vector<dataCPU<float>> debugData = optimizer.getDebugData(frame, kframe, cam, 1);
                            debugLocalizationQueue.push(debugData);
                        }
                        break;
                    }
                }
            }
            std::cout << "localization time " << tt.toc() << std::endl;

            // update the global pose
            SE3f newGlobalPose = kframe.localPoseToGlobal(frame.getLocalPose());
            frame.setGlobalPose(newGlobalPose);

            lastGlobalMovement = newGlobalPose * lastGlobalPose.inverse();
            lastGlobalPose = newGlobalPose;

            fQueue.push(frame);

            frameId++;
        }
    }

    void mappingThread()
    {
        poseMapOptimizerCPU optimizer(width, height);
        renderCPU renderer(width, height);
        dataMipMapCPU<imageType> image_buffer(width, height, -1);
        dataMipMapCPU<float> depth_buffer(width, height, -1);
        dataMipMapCPU<float> weight_buffer(width, height, -1);

        std::vector<frameCPU> frameStack;
        keyFrameCPU kframe = initialKeyframe;

        tic_toc tt;

        while (true)
        {
            frameCPU frame = fQueue.pop();

            while (true)
            {
                float lastMinViewAngle = M_PI;
                for (frameCPU f : frameStack)
                {
                    float lastViewAngle = meanViewAngle(kframe, kframe.globalPoseToLocal(f.getGlobalPose()), kframe.globalPoseToLocal(frame.getGlobalPose()));
                    if (lastViewAngle < lastMinViewAngle)
                        lastMinViewAngle = lastViewAngle;
                }

                if (lastMinViewAngle > mesh_vo::last_min_angle)
                {
                    frameStack.push_back(frame);
                    if (frameStack.size() > mesh_vo::num_frames)
                        frameStack.erase(frameStack.begin());
                }

                if (!fQueue.try_pop(frame))
                    break;
            }

            if (frameStack.size() < mesh_vo::num_frames)
                continue;

            float keyframeViewAngle = meanViewAngle(kframe, SE3f(), kframe.globalPoseToLocal(frame.getGlobalPose()));

            image_buffer.setToNoData(1);
            renderer.renderImageParallel(kframe, kframe.globalPoseToLocal(frame.getGlobalPose()), image_buffer, cam, 1);
            float pnodata = image_buffer.get(1).getPercentNoData();
            float viewPercent = 1.0 - pnodata;

            if (viewPercent > mesh_vo::min_view_perc && keyframeViewAngle < mesh_vo::key_max_angle)
                continue;

            // select new keyframe
            // int newKeyframeIndex = 0;
            int newKeyframeIndex = int(frameStack.size() / 2);
            // int newKeyframeIndex = int(goodFrames.size() - 1);
            frameCPU newKeyframe = frameStack[newKeyframeIndex];

            depth_buffer.setToNoData(0);
            weight_buffer.setToNoData(0);
            renderer.renderDepthParallel(kframe, kframe.globalPoseToLocal(newKeyframe.getGlobalPose()), depth_buffer, cam, 0);
            renderer.renderWeightParallel(kframe, kframe.globalPoseToLocal(newKeyframe.getGlobalPose()), weight_buffer, cam, 0);
            renderer.renderInterpolate(depth_buffer.get(0));

            kframe = keyFrameCPU(newKeyframe.getRawImage(0), vec2f(0.0, 0.0), newKeyframe.getGlobalPose(), kframe.getGlobalScale());
            kframe.initGeometryFromDepth(depth_buffer.get(0), weight_buffer.get(0), cam);

            vec2f meanStd = kframe.getGeometry().meanStdDepth();
            // vec2f minMax = kframe.getGeometry().minMaxDepthParams();
            // vec2f minMax = kframe.getGeometry().minMaxDepthVertices();
            float scale = mesh_vo::mapping_mean_depth / meanStd(0);

            kframe.scaleVerticesAndWeights(scale);

            std::vector<frameCPU> keyframes = frameStack;
            keyframes.erase(keyframes.begin() + newKeyframeIndex);

            // initialize the local poses
            for (size_t i = 0; i < keyframes.size(); i++)
            {
                keyframes[i].setLocalPose(kframe.globalPoseToLocal(keyframes[i].getGlobalPose()));
            }

            // this will update the local pose and the local map
            // optimizer.optimize(frameStack, kframe, cam);

            tt.tic();
            for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
            {
                optimizer.init(keyframes, kframe, cam, lvl);
                if (plotDebug)
                {
                    std::vector<dataCPU<float>> debugData = optimizer.getDebugData(keyframes, kframe, cam, 1);
                    debugMappingQueue.push(debugData);
                }
                while (true)
                {
                    optimizer.step(keyframes, kframe, cam, lvl);
                    if (optimizer.converged())
                    {
                        if (plotDebug)
                        {
                            std::vector<dataCPU<float>> debugData = optimizer.getDebugData(keyframes, kframe, cam, 1);
                            debugMappingQueue.push(debugData);
                        }
                        break;
                    }
                }
            }
            std::cout << "mapping time " << tt.toc() << std::endl;

            // update the global poses
            for (size_t i = 0; i < keyframes.size(); i++)
            {
                keyframes[i].setGlobalPose(kframe.localPoseToGlobal(keyframes[i].getLocalPose()));
                for (size_t j = 0; j < frameStack.size(); j++)
                {
                    if (keyframes[i].getId() == frameStack[j].getId())
                    {
                        frameStack[j].setLocalPose(keyframes[i].getLocalPose());
                        frameStack[j].setGlobalPose(keyframes[i].getGlobalPose());
                    }
                }
            }

            kfQueue.push(kframe);
        }
    }

    void voThread()
    {
        poseOptimizerCPU poseOptimizer(width, height);
        //poseVelOptimizerCPU poseOptimizer(width, height);
        poseMapOptimizerCPU poseMapOptimizer(width, height);

        renderCPU renderer(width, height);
        dataMipMapCPU<imageType> image_buffer(width, height, -1);
        dataMipMapCPU<float> depth_buffer(width, height, -1);
        dataMipMapCPU<float> weight_buffer(width, height, -1);

        dataCPU<imageType> image;
        std::vector<frameCPU> frameStack;
        std::vector<frameCPU> keyframes;

        keyFrameCPU kframe = initialKeyframe;

        tic_toc tt;

        SE3f lastGlobalPose;
        SE3f lastGlobalMovement;

        while (true)
        {
            // if there is a new image, compute its pose
            if (iQueue.try_pop(image))
            {
                // keep on reading and keep the last image
                while (true)
                {
                    if (!iQueue.try_pop(image))
                        break;
                }

                frameCPU frame(image, frameId);

                // initialize the global and local pose
                frame.setGlobalPose(lastGlobalMovement * lastGlobalPose);
                frame.setLocalPose(kframe.globalPoseToLocal(frame.getGlobalPose()));

                // this will update the local pose
                tt.tic();
                for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
                {
                    poseOptimizer.init(frame, kframe, cam, lvl);
                    // if (plotDebug)
                    //{
                    //      std::vector<dataCPU<float>> debugData = poseOptimizer.getDebugData(frame, kframe, cam, 1);
                    //      debugLocalizationQueue.push(debugData);
                    //  }
                    while (true)
                    {
                        poseOptimizer.step(frame, kframe, cam, lvl);
                        if (poseOptimizer.converged())
                        {
                            //     if (plotDebug)
                            //     {
                            //         std::vector<dataCPU<float>> debugData = poseOptimizer.getDebugData(frame, kframe, cam, 1);
                            //         debugLocalizationQueue.push(debugData);
                            //     }
                            break;
                        }
                    }
                }
                if (plotDebug)
                {
                    std::vector<dataCPU<float>> debugData = poseOptimizer.getDebugData(frame, kframe, cam, 1);
                    debugLocalizationQueue.push(debugData);
                }

                std::cout << "localization time " << tt.toc() << std::endl;

                // update the global pose
                frame.setGlobalPose(kframe.localPoseToGlobal(frame.getLocalPose()));

                lastGlobalMovement = frame.getGlobalPose() * lastGlobalPose.inverse();
                lastGlobalPose = frame.getGlobalPose();

                frameId++;

                // Choose wether to save the frame or not
                float lastMinViewAngle = M_PI;
                for (frameCPU f : frameStack)
                {
                    float lastViewAngle = meanViewAngle(kframe, f.getLocalPose(), frame.getLocalPose());
                    if (lastViewAngle < lastMinViewAngle)
                        lastMinViewAngle = lastViewAngle;
                }

                if (lastMinViewAngle > mesh_vo::last_min_angle)
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

                float keyframeViewAngle = meanViewAngle(kframe, SE3f(), frame.getLocalPose());

                image_buffer.setToNoData(1);
                renderer.renderImageParallel(kframe, frame.getLocalPose(), image_buffer, cam, 1);
                float pnodata = image_buffer.get(1).getPercentNoData();
                float viewPercent = 1.0 - pnodata;

                if (viewPercent > mesh_vo::min_view_perc && keyframeViewAngle < mesh_vo::key_max_angle)
                    continue;

                // save last keyframe, we wont be updating it anymore
                kfQueue.push(kframe);

                // select new keyframe
                // int newKeyframeIndex = 0;
                int newKeyframeIndex = int(frameStack.size() / 2);
                // int newKeyframeIndex = int(frameStack.size() - 2);
                frameCPU newKeyframe = frameStack[newKeyframeIndex];

                depth_buffer.setToNoData(1);
                weight_buffer.setToNoData(1);
                renderer.renderDepthParallel(kframe, newKeyframe.getLocalPose(), depth_buffer, cam, 1);
                renderer.renderWeightParallel(kframe, newKeyframe.getLocalPose(), weight_buffer, cam, 1);
                //dataCPU<float> depth = depth_buffer.get(1);
                renderer.renderInterpolate(depth_buffer.get(1));

                kframe = keyFrameCPU(newKeyframe.getRawImage(0), vec2f(0.0, 0.0), newKeyframe.getGlobalPose(), kframe.getGlobalScale());
                kframe.initGeometryFromDepth(depth_buffer.get(1), weight_buffer.get(1), cam);
                // kframe.initGeometryVerticallySmooth(cam);

                vec2f meanStd = kframe.getGeometry().meanStdDepth();
                // vec2f minMax = kframe.getGeometry().minMaxDepthParams();
                // vec2f minMax = kframe.getGeometry().minMaxDepthVertices();
                float scale = mesh_vo::mapping_mean_depth / meanStd(0);

                kframe.scaleVerticesAndWeights(scale);

                // initialize the local poses
                for (size_t i = 0; i < frameStack.size(); i++)
                {
                    frameStack[i].setLocalPose(kframe.globalPoseToLocal(frameStack[i].getGlobalPose()));
                }

                keyframes = frameStack;
                keyframes.erase(keyframes.begin() + newKeyframeIndex);

                // renderer.renderIdepthLineSearch(kframe, frameStack[0], cam, 1);

                // init the posemapoptimizer
                poseMapOptimizer.init(keyframes, kframe, cam, mesh_vo::mapping_fin_lvl);
            }

            if (keyframes.size() < 1)
                continue;

            // this will update the local pose and the local map
            tt.tic();
            poseMapOptimizer.step(keyframes, kframe, cam, mesh_vo::mapping_fin_lvl);
            std::cout << "mapping time " << tt.toc() << std::endl;
            if (plotDebug)
            {
                std::vector<dataCPU<float>> debugData = poseMapOptimizer.getDebugData(keyframes, kframe, cam, 1);
                debugMappingQueue.push(debugData);
            }

            // update the global poses
            for (size_t i = 0; i < keyframes.size(); i++)
            {
                keyframes[i].setGlobalPose(kframe.localPoseToGlobal(keyframes[i].getLocalPose()));
                for (size_t j = 0; j < frameStack.size(); j++)
                {
                    if (keyframes[i].getId() == frameStack[j].getId())
                    {
                        frameStack[j].setLocalPose(keyframes[i].getLocalPose());
                        frameStack[j].setGlobalPose(keyframes[i].getGlobalPose());
                    }
                }
            }
        }
    }

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

        geometryPlotter geomPlotter;
        trayectoryPlotter trayPlotter;
        std::vector<imagePlotter> imgPosePlotter;
        imgPosePlotter.push_back(imagePlotter(0, 0));//keyframe
        imgPosePlotter.push_back(imagePlotter(0, 1));//depth
        imgPosePlotter.push_back(imagePlotter(0, 2));//frame
        imgPosePlotter.push_back(imagePlotter(0, 3));//errpr

        std::vector<imagePlotter> imgMapPlotter;
        imgMapPlotter.push_back(imagePlotter(1, 0));//keyframe
        imgMapPlotter.push_back(imagePlotter(1, 1));//depth
        imgMapPlotter.push_back(imagePlotter(2, 0));//frame
        imgMapPlotter.push_back(imagePlotter(2, 1));//error
        imgMapPlotter.push_back(imagePlotter(3, 0));//frame
        imgMapPlotter.push_back(imagePlotter(3, 1));//error

        geomPlotter.compileShaders();
        trayPlotter.compileShaders();
        for (imagePlotter &imgPlotter : imgPosePlotter)
            imgPlotter.compileShaders();
        for (imagePlotter &imgPlotter : imgMapPlotter)
            imgPlotter.compileShaders();

        keyFrameCPU kframe = initialKeyframe;
        frameCPU frame;

        dataCPU<float> imageFloat = kframe.getRawImage(0).convert<float>();
        geomPlotter.setBuffers(imageFloat, kframe.getGeometry());

        while (!pangolin::ShouldQuit())
        {
            if (kfQueue.try_pop(kframe))
            {
                kframe.scaleVerticesAndWeights(1.0 / kframe.getGlobalScale());
                kframe.getGeometry().transform(kframe.getGlobalPose().inverse());
                poses.push_back(kframe.getGlobalPose());
                trayPlotter.setBuffers(poses, cam);
                dataCPU<float> imageFloat2 = kframe.getRawImage(0).convert<float>();
                geomPlotter.setBuffers(imageFloat2, kframe.getGeometry());
            }

            if (!debugLocalizationQueue.empty())
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

            if (!debugMappingQueue.empty())
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

            /*
            if (fQueue.try_peek(frame))
            {
                poses.push_back(frame.getGlobalPose());
                plotter.setBuffers(poses, cam);
            }
            */

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

    float meanViewAngle(keyFrameCPU &kframe, SE3f pose1, SE3f pose2)
    {
        int lvl = 1;

        geometryType scene1 = kframe.getGeometry();
        scene1.transform(pose1);
        scene1.project(cam);

        geometryType scene2 = kframe.getGeometry();
        scene2.transform(pose2);
        scene2.project(cam);

        SE3f relativePose = pose1 * pose2.inverse();

        SE3f frame1PoseInv = relativePose.inverse();
        SE3f frame2PoseInv = SE3f();

        vec3f frame1Translation = frame1PoseInv.translation();
        vec3f frame2Translation = frame2PoseInv.translation();

        std::vector<int> vIds = scene2.getVerticesIds();

        float accAngle = 0;
        int count = 0;
        for (int vId : vIds)
        {
            vertex vert = scene2.getVertex(vId);

            vec3f diff1 = vert.ver - frame1Translation;
            vec3f diff2 = vert.ver - frame2Translation;

            assert(diff1.norm() > 0 && diff2.norm() > 0);

            vec3f diff1Normalized = diff1 / diff1.norm();
            vec3f diff2Normalized = diff2 / diff2.norm();

            float cos_angle = diff1Normalized.dot(diff2Normalized);
            cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);
            float angle = std::acos(cos_angle);

            assert(!std::isnan(angle));

            accAngle += angle;
            count += 1;
        }

        return accAngle / count;
    }

    std::thread tLocalization;
    std::thread tMapping;
    std::thread tVisualization;

    ThreadSafeQueue<dataCPU<imageType>> iQueue;
    ThreadSafeQueue<frameCPU> fQueue;
    ThreadSafeQueue<keyFrameCPU> kfQueue;

    ThreadSafeQueue<std::vector<dataCPU<float>>> debugLocalizationQueue;
    ThreadSafeQueue<std::vector<dataCPU<float>>> debugMappingQueue;

    cameraType cam;
    int width;
    int height;
    int frameId;

    keyFrameCPU initialKeyframe;

    bool plotDebug;
};