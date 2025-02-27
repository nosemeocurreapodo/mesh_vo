#pragma once

#include <iostream>
#include <fstream>

#include "cpu/frameCPU.h"
#include "cpu/keyFrameCPU.h"

#include "optimizers/poseOptimizerCPU.h"
#include "optimizers/mapOptimizerCPU.h"
#include "optimizers/poseMapOptimizerCPU.h"
#include "optimizers/intrinsicPoseMapOptimizerCPU.h"

#include "cpu/OpenCVDebug.h"

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

    visualOdometryThreaded(dataCPU<float> &image, SE3f globalPose, cameraType _cam) : cam(_cam)
    {
        initialKeyframe = keyFrameCPU(image, vec2f(0.0, 0.0), globalPose, 1.0);
        initialKeyframe.initGeometryVerticallySmooth(cam);

        width = image.width;
        height = image.height;
        frameId = 0;

        tLocalization = std::thread(&visualOdometryThreaded::localizationThread, this);
        tMapping = std::thread(&visualOdometryThreaded::mappingThread, this);

        plotDebug = true;
    }

    ~visualOdometryThreaded()
    {
        tLocalization.join();
        tMapping.join();
    }

    void init(dataCPU<float> &image, SE3f globalPose)
    {
        initialKeyframe = keyFrameCPU(image, vec2f(0.0, 0.0), globalPose, 1.0);
        initialKeyframe.initGeometryVerticallySmooth(cam);

        width = image.width;
        height = image.height;
        frameId = 0;

        tLocalization = std::thread(&visualOdometryThreaded::localizationThread, this);
        tMapping = std::thread(&visualOdometryThreaded::mappingThread, this);

        plotDebug = true;
    }

    void locAndMap(dataCPU<float> &image)
    {
        // frameCPU frame(image, 0);
        // fQueue.push(frame);
        // if(fQueue.size() > 5)
        //     fQueue.pop();

        iQueue.push(image);

        if (plotDebug)
        {
            while (!debugLocalizationQueue.empty())
            {
                std::vector<dataCPU<float>> debugLoc = debugLocalizationQueue.pop();
                show(debugLoc, "localization Debug");
            }
            while (!debugMappingQueue.empty())
            {
                std::vector<dataCPU<float>> debugLoc = debugMappingQueue.pop();
                show(debugLoc, "mapping Debug");
            }
        }
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
        dataMipMapCPU<float> depth_buffer(width, height, -1);
        dataMipMapCPU<float> weight_buffer(width, height, -1);

        std::vector<frameCPU> frameStack;

        keyFrameCPU kframe = initialKeyframe;

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

            //int lvl = 1;
            //renderer.renderImageParallel(kframe, kframe.globalPoseToLocal(frame.getGlobalPose()), depth_buffer, cam, lvl);
            //float pnodata = depth_buffer.get(lvl).getPercentNoData();
            //float viewPercent = 1.0 - pnodata;

            //if (viewPercent > mesh_vo::min_view_perc)
            //    continue;

            // select new keyframe
            // int newKeyframeIndex = 0;
            int newKeyframeIndex = int(frameStack.size() / 2);
            // int newKeyframeIndex = int(goodFrames.size() - 1);
            frameCPU newKeyframe = frameStack[newKeyframeIndex];

            // render its idepth
            int lvl = 0;

            renderer.renderDepthParallel(kframe, kframe.globalPoseToLocal(newKeyframe.getGlobalPose()), depth_buffer, cam, lvl);
            renderer.renderWeightParallel(kframe, kframe.globalPoseToLocal(newKeyframe.getGlobalPose()), weight_buffer, cam, lvl);
            renderer.renderInterpolate(depth_buffer.get(lvl));

            kframe = keyFrameCPU(newKeyframe.getRawImage(0), vec2f(0.0, 0.0), newKeyframe.getGlobalPose(), kframe.getGlobalScale());
            kframe.initGeometryFromDepth(depth_buffer.get(lvl), weight_buffer.get(lvl), cam);

            vec2f meanStd = kframe.getGeometry().meanStdDepth();
            // vec2f minMax = kframe.getGeometry().minMaxDepthParams();
            // vec2f minMax = kframe.getGeometry().minMaxDepthVertices();
            float scale = 1.0 / meanStd(0);

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

            for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
            {
                optimizer.init(keyframes, kframe, cam, lvl);
                // if (plotDebug)
                //{
                //     std::vector<dataCPU<float>> debugData = optimizer.getDebugData(keyframes, kframe, cam, 1);
                //     debugMappingQueue.push(debugData);
                // }
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