#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

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

    cv::Mat keyFrame = cv::imread("../../Trajectory_30_seconds/scene_000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat keyframeFLoat;
    keyFrame.convertTo(keyframeFLoat, CV_32FC1, 1.0/255.0);
    cv::flip(keyframeFLoat, keyframeFLoat,0);
    Sophus::SE3f keyframePose = readPose("../../Trajectory_30_seconds/scene_000.txt");

    cv::Mat iDepth;
    cv::FileStorage fs("../../Trajectory_30_seconds/scene_depth_000.yml", cv::FileStorage::READ );
    fs["idepth"] >> iDepth;

    mesh_vo visual_odometry(fx,fy,cx,cy,width,height);

    visual_odometry.setKeyframeWithIdepth(keyframeFLoat.clone(), iDepth.clone());


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
        sprintf(image_filename,"../../Trajectory_30_seconds/scene_%03d.png", frameNumber);
        sprintf(RT_filename,"../../Trajectory_30_seconds/scene_%03d.txt", frameNumber);

        Sophus::SE3f pose = readPose(RT_filename);
        cv::Mat frame = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);
        cv::Mat frameFloat;
        frame.convertTo(frameFloat, CV_32FC1, 1.0/255.0);
        cv::flip(frameFloat, frameFloat,0);

        Sophus::SE3f realPose = pose*keyframePose.inverse();

        visual_odometry.updatePose(frameFloat);

        //std::cout << "real pose " << std::endl;
        //std::cout << realPose.matrix() << std::endl;
        //std::cout << "est pose " << std::endl;
        //std::cout << visual_odometry.framePose.matrix() << std::endl;

        //cv::imshow("image", frame);
        //cv::imshow("keyframe", keyFrame);
        //cv::imshow("idepth", iDepth);
        //cv::waitKey(30);

        if(framesTracked >= framesToTrack)
        {

            char depth_filename[500];
            sprintf(depth_filename,"../../../Trajectory_30_seconds/scene_depth_%03d.yml", frameNumber);

            if(!fileExist(depth_filename))
                continue;

            cv::FileStorage fs(depth_filename, cv::FileStorage::READ );
            fs["idepth"] >> iDepth;

            printf("frames tracked %d\n", framesTracked);
            framesTracked = 0;
            framesToTrack =  rand() % 5 + 25;//25;

            visual_odometry.setKeyframeWithIdepth(frame.clone(), iDepth.clone());
            //visual_odometry.reset();

            keyFrame = frame.clone();
            keyframePose = pose;


            //save depth
            //char depth_filename[500];
            //sprintf(depth_filename,"../../../Trajectory_30_seconds/scene_depth_%03d.yml", keyframeNumber);

            //cv::FileStorage fs(depth_filename, cv::FileStorage::WRITE );
            //fs << "idepth" << iDepth;  //choose any key here, just be consistant with the one below

            //Mat fm;
            //FileStorage fs("my.yml", FileStorage::READ );
            //fs["idepth"] >> fm;

        }
    }

    return 1;
}
