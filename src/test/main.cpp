#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/convertAhandaPovRayToStandard.h"
#include "mesh_vo.h"

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

    cv::Mat initFrame = cv::imread("../../desktop_dataset/scene_000.png", cv::IMREAD_GRAYSCALE);
    Sophus::SE3f initPose = readPose("../../desktop_dataset/scene_000.txt");

    cv::Mat initIdepth;
    cv::FileStorage fs("../../desktop_dataset/scene_depth_000.yml", cv::FileStorage::READ );
    fs["idepth"] >> initIdepth;


    mesh_vo visual_odometry(fx,fy,cx,cy,width,height);

    visual_odometry.initWithRandomIdepth(initFrame, initPose*initPose.inverse());
    //visual_odometry.initWithIdepth(initFrame, initIdepth, initPose*initPose.inverse());

    while(1){
        framesTracked++;
        frameNumber += frameCounterDirection;
        if(frameNumber > 598)
            frameCounterDirection = -1;
        if(frameNumber < 2)
            frameCounterDirection = 1;

        char image_filename[500];
        char RT_filename[500];

        //file name
        sprintf(image_filename,"../../desktop_dataset/scene_%03d.png", frameNumber);
        sprintf(RT_filename,"../../desktop_dataset/scene_%03d.txt", frameNumber);

        cv::Mat frame = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);
        Sophus::SE3f realPose = readPose(RT_filename)*initPose.inverse();

        visual_odometry.localization(frame);
        //visual_odometry.mapping(frame, realPose);
        //visual_odometry.visual_odometry(frame);
        //Sophus::SE3f estPose = visual_odometry.calcPose(frameFloat);
        //visual_odometry.addFrameToStack(frameFloat, realPose);
        //visual_odometry.updateMap();

        std::cout << "real pose " << std::endl;
        std::cout << realPose.matrix() << std::endl;


        //std::cout << "est pose " << std::endl;
        //std::cout << visual_odometry.trackedPose.matrix() << std::endl;

        cv::imshow("image", frame);
        cv::waitKey(30);
    }

    return 1;
}
