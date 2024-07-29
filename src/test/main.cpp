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

int main(void)
{
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

    cv::Mat imageMat = cv::imread("../../desktop_dataset/scene_000.png", cv::IMREAD_GRAYSCALE);
    Sophus::SE3f initPose = readPose("../../desktop_dataset/scene_000.txt");
    
    imageMat.convertTo(imageMat, CV_32FC1);
    cv::resize(imageMat, imageMat, cv::Size(cam.width, cam.height), cv::INTER_AREA);

    dataCPU<float> image(cam.width, cam.height, -1.0);
    image.set((float*)imageMat.data);
    
    cv::Mat idepthMat;
    cv::FileStorage fs("../../desktop_dataset/scene_depth_000.yml", cv::FileStorage::READ );
    fs["idepth"] >> idepthMat;

    idepthMat.convertTo(idepthMat, CV_32FC1);
    cv::resize(idepthMat, idepthMat, cv::Size(cam.width, cam.height), cv::INTER_AREA);
    //cv::blur(idepthMat, idepthMat, cv::Size(5, 5));

    dataCPU<float> idepth(cam.width, cam.height, -1.0);
    idepth.set((float*)idepthMat.data);

    visualOdometry odometry(cam);

    //odometry.initScene(image, idepth);
    odometry.initScene(image);

    while(1){
        framesTracked++;
        frameNumber += frameCounterDirection;
        if(frameNumber > 598)
            frameCounterDirection = -1;
        if(frameNumber < 2)
            frameCounterDirection = 1;
        if(frameNumber > 50)
            return 1;

        char image_filename[500];
        char RT_filename[500];

        //file name
        sprintf(image_filename,"../../desktop_dataset/scene_%03d.png", frameNumber);
        sprintf(RT_filename,"../../desktop_dataset/scene_%03d.txt", frameNumber);

        cv::Mat imageMat = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);
        Sophus::SE3f realPose = readPose(RT_filename)*initPose.inverse();

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
