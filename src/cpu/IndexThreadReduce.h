/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>

#include <stdio.h>
#include <iostream>

#include "frame.h"
#include "HJPose.h"

#define NUM_THREADS 4

class IndexThreadReduce
{

public:
    inline IndexThreadReduce()
    {
        minIndex = 0;
        maxIndex = 0;
        stepSize = 1;
        lvl = 0;
        callPerIndex = std::bind(&IndexThreadReduce::callPerIndexDefault, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);

        running = true;
        for(int i=0;i<NUM_THREADS;i++)
        {
            isDone[i] = false;
            workerThreads[i] = std::thread(&IndexThreadReduce::workerLoop, this, i);
        }
        //printf("created ThreadReduce\n");
    }
    inline ~IndexThreadReduce()
    {
        running = false;

        exMutex.lock();
        todo_signal.notify_all();
        exMutex.unlock();

        for(int i=0;i<NUM_THREADS;i++)
            workerThreads[i].join();

        //printf("destroyed ThreadReduce\n");
    }

    inline HJPose reduce(std::function<HJPose(frame* ,int,int,int)> _callPerIndex, frame* __frame, int _lvl, int ymin, int ymax)
    {
        //std::cout << "reduce called " << std::endl;

        std::unique_lock<std::mutex> lock(exMutex);

        stepSize = (ymax - ymin)/NUM_THREADS;
        // save
        callPerIndex = _callPerIndex;
        minIndex = ymin;
        maxIndex = ymax;
        lvl = _lvl;
        _frame = __frame;

        // go worker threads!
        for(int i=0;i<NUM_THREADS;i++)
            isDone[i] = false;

        // let them start!
        todo_signal.notify_all();

        //std::cout << "reduce waiting for threads to finish " << std::endl;

        //printf("reduce waiting for threads to finish\n");
        // wait for all worker threads to signal they are done.
        while(true)
        {
            // wait for at least one to finish
            done_signal.wait(lock);
            //printf("thread finished!\n");

            //std::cout << "reduce threads finished " << std::endl;

            // check if actually all are finished.
            bool allDone = true;
            for(int i=0;i<NUM_THREADS;i++)
                allDone = allDone && isDone[i];

            // all are finished! exit.
            if(allDone)
                break;
        }

        //std::cout << "reduce threads finished " << std::endl;

        minIndex = 0;
        maxIndex = 0;
        this->callPerIndex = std::bind(&IndexThreadReduce::callPerIndexDefault, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);

        HJPose result;
        for(int i = 0; i < NUM_THREADS; i++)
            result += results[i];

        return result;

        //printf("reduce done (all threads finished)\n");
    }

private:
    std::thread workerThreads[NUM_THREADS];
    bool isDone[NUM_THREADS];
    HJPose results[NUM_THREADS];
    frame *_frame;

    std::mutex exMutex;
    std::condition_variable todo_signal;
    std::condition_variable done_signal;

    int minIndex;
    int maxIndex;
    int stepSize;

    int lvl;

    bool running;

    std::function<HJPose(frame*,int,int,int)> callPerIndex;

    HJPose callPerIndexDefault(frame *_frame, int _lvl, int _ymin, int _ymax)
    {
        printf("ERROR: should never be called....\n");
        HJPose p;
        return p;
    }

    void workerLoop(int idx)
    {
        //std::cout << "worker thead " << idx << std::endl;

        std::unique_lock<std::mutex> lock(exMutex);

        while(running)
        {
            todo_signal.wait(lock);
            lock.unlock();

            //std::cout << "worker thread " << idx << " has something to do!" << std::endl;



            //assert(callPerIndex != 0);

            int startIndex = minIndex + stepSize*idx;
            int endIndex = startIndex + stepSize;

            //std::cout << "worker thread " << idx << " start " << startIndex << " end " << endIndex << std::endl;

            results[idx] = callPerIndex(_frame, lvl, startIndex, endIndex);

            //std::cout << "worker thread " << idx << " done" << std::endl;
            lock.lock();
            isDone[idx] = true;
            done_signal.notify_all();
            //lock.unlock();

            //std::cout << "worker thread " << idx << " waiting" << std::endl;
        }
    }
};

