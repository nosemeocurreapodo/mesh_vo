#pragma once

#include <iostream>
#include <fstream>

#include "cpu/frameCPU.h"
#include "cpu/keyFrameCPU.h"

#include "optimizers/poseOptimizerCPU.h"
#include "optimizers/mapOptimizerCPU.h"
#include "optimizers/poseMapOptimizerCPU.h"
#include "optimizers/intrinsicPoseMapOptimizerCPU.h"

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

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cv_;
};

class visualOdometryThreaded
{
public:
    visualOdometryThreaded(dataCPU<float> &image, SE3f globalPose, cameraType _cam, int _width, int _height):
    frameId(0), cam(_cam), width(_width), height(_height),
    tLocalization(&visualOdometryThreaded::localizationThread, this),
    tMapping(&visualOdometryThreaded::mappingThread, this)
    {
        keyFrameCPU initialkeyframe(width, height);
        initialkeyframe.init(image, vec2f(0.0, 0.0), globalPose, 1.0);
    }

    ~visualOdometryThreaded()
    {
        tLocalization.join();
        tMapping.join();
    }

    void locAndMap(dataCPU<float> &image)
    {
        iQueue.push(image);
    }

    keyFrameCPU getKeyframe()
    {
        return kfQueue.peek();
    }

private:
    void localizationThread()
    {
        poseOptimizerCPU optimizer(width, height);

        keyFrameCPU kframe(width, height);

        while (true)
        {
            if (!kfQueue.empty())
            {
                kframe = kfQueue.peek();
            }

            dataCPU<imageType> image = iQueue.pop();
            frameCPU frame(image, image.computeFrameDerivative(), frameId);
            // frame.setLocalPose(lastLocalMovement * lastLocalPose);
            optimizer.optimize(frame, kframe, cam);
            fQueue.push(frame);
            frameId++;
        }
    }

    void mappingThread()
    {
        poseMapOptimizerCPU optimizer(width, height);
        std::vector<frameCPU> frameStack;

        keyFrameCPU kframe(width, height);

        while (true)
        {
            if (!fQueue.empty())
            {
                frameStack.push_back(fQueue.pop());
            }
            optimizer.optimize(frameStack, kframe, cam);
            kfQueue.push(kframe);
        }
    }

    std::thread tLocalization;
    std::thread tMapping;

    ThreadSafeQueue<dataCPU<imageType>> iQueue;
    ThreadSafeQueue<frameCPU> fQueue;
    ThreadSafeQueue<keyFrameCPU> kfQueue;
    cameraType cam;
    int width;
    int height;
    int frameId;
};