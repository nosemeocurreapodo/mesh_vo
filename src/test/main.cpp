#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/convertAhandaPovRayToStandard.h"
#include "visualOdometry.h"
#include "common/camera.h"
#include "cpu/frameCPU.h"
#include "cpu/dataCPU.h"

#include <Eigen/Core>
#include "sophus/se3.hpp"

inline bool fileExist(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char * argv[])
{
    if(argc != 2)
    {
        std::cout << "usage: ./testapp /path/to/dataset" << std::endl;
        return 0;
    }

    std::string dataset_path = argv[1];

    int frameNumber = 0;
    int frameCounterDirection = 1;

    int framesToTrack = 20;//rand() % 10 + 50;
    int framesTracked = 0;

    int width = 640;
    int height = 480;
    float fx, fy, cx, cy;
    fx = 481.20; fy = 480.0; cx = 319.5; cy = 239.5;

    camera cam(fx, fy, cx, cy, width, height);
    cam.resize(512, 512);

    cv::Mat imageMat = cv::imread(dataset_path + "images/scene_000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat idepthMat = cv::imread(dataset_path + "depths/scene_000.png", cv::IMREAD_GRAYSCALE);
    Sophus::SE3f initPose = readPose(dataset_path + "poses/scene_000.txt");

    imageMat.convertTo(imageMat, CV_32FC1);
    cv::resize(imageMat, imageMat, cv::Size(cam.width, cam.height), cv::INTER_AREA);

    dataCPU<float> image(cam.width, cam.height, -1.0);
    image.set((float*)imageMat.data);
    
    idepthMat.convertTo(idepthMat, CV_32FC1);
    cv::resize(idepthMat, idepthMat, cv::Size(cam.width, cam.height), cv::INTER_AREA);

    //normalize idepth
    cv::Scalar mean, stddev;
    cv::meanStdDev(idepthMat, mean, stddev);
    //std::cout << "mean: " << mean << " std: " << stddev << std::endl;
    idepthMat = (idepthMat - mean[0])/stddev[0];
    //set new mean/std
    idepthMat = idepthMat*0.3f + 1.0f;

    dataCPU<float> idepth(cam.width, cam.height, -1.0);
    idepth.set((float*)idepthMat.data);

    dataCPU<float> ivar(cam.width, cam.height, -1.0);
    ivar.set(1.0, 0);
    ivar.generateMipmaps();

    visualOdometry odometry(cam);

    odometry.initScene(image, idepth, ivar);
    //odometry.initScene(image);

    while(1){
        framesTracked++;
        frameNumber += frameCounterDirection;
        if(frameNumber > 598)
            frameCounterDirection = -1;
        if(frameNumber < 2)
            frameCounterDirection = 1;
        //if(frameNumber > 50)
        //    return 1;

        char image_filename[500];
        char RT_filename[500];

        //file name
        sprintf(image_filename, "images/scene_%03d.png", frameNumber);
        sprintf(RT_filename, "poses/scene_%03d.txt", frameNumber);

        std::string image_path = dataset_path + image_filename;
        std::string pose_path = dataset_path + RT_filename;

        cv::Mat imageMat = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        Sophus::SE3f realPose = readPose(pose_path)*initPose.inverse();

        imageMat.convertTo(imageMat, CV_32FC1);
        cv::resize(imageMat, imageMat, cv::Size(cam.width, cam.height), cv::INTER_AREA);

        image.set((float*)imageMat.data);

        //odometry.localization(image);
        odometry.mapping(image, realPose);
        //odometry.locAndMap(image);
        //Sophus::SE3f estPose = visual_odometry.calcPose(frameFloat);
        //visual_odometry.addFrameToStack(frameFloat, realPose);
        //visual_odometry.updateMap();

        std::cout << "real pose " << std::endl;
        std::cout << realPose.matrix() << std::endl;

        //std::cout << "est pose " << std::endl;
        //std::cout << visual_odometry.trackedPose.matrix() << std::endl;

        //cv::imshow("image", frame);
        //cv::waitKey(30);
    }

    return 1;
}
