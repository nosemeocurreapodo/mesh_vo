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

    int in_width = 640;
    int in_height = 480;
    float in_fx, in_fy, in_cx, in_cy;
    in_fx = 481.20; in_fy = 480.0; in_cx = 319.5; in_cy = 239.5;

    int out_width = 512;
    int out_height = 512;

    float xp = float(out_width)/in_width;
    float yp = float(out_height)/in_height;

    float out_fx, out_fy, out_cx, out_cy;
    out_fx = in_fx*xp; out_fy = in_fy*yp; out_cx = in_cx*xp; out_cy = in_cy*yp;

    cv::Mat keyFrame = cv::imread("../../desktop_dataset/scene_000.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat keyframeFLoat;
    //keyFrame.convertTo(keyframeFLoat, CV_32FC1, 1.0/1.0);
    cv::flip(keyFrame, keyFrame,0);
    cv::Mat keyframeResized;//dst image
    cv::resize(keyFrame,keyframeResized,cv::Size(out_width, out_height));//resize image
    Sophus::SE3f keyframePose = readPose("../../desktop_dataset/scene_000.txt");

    cv::Mat iDepth;
    cv::FileStorage fs("../../desktop_dataset/scene_depth_000.yml", cv::FileStorage::READ );
    fs["idepth"] >> iDepth;
    cv::Mat idepthResized;
    cv::resize(iDepth,idepthResized,cv::Size(out_width, out_height));//resize image


    mesh_vo visual_odometry(out_fx,out_fy,out_cx,out_cy,out_width,out_height);

    //visual_odometry.setKeyframeRandomIdepth(keyframeFLoat.clone());
    visual_odometry.setKeyframeWithIdepth(keyframeResized.clone(), idepthResized.clone());

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

        Sophus::SE3f pose = readPose(RT_filename);
        cv::Mat frame = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);
        //cv::Mat frameFloat;
        //frame.convertTo(frameFloat, CV_32FC1, 1.0/1.0);
        cv::flip(frame, frame, 0);
        cv::Mat frameResized;//dst image
        cv::resize(frame,frameResized,cv::Size(out_width, out_height));//resize image

        Sophus::SE3f realPose = pose*keyframePose.inverse();


        visual_odometry.visual_odometry(frameResized);
        //Sophus::SE3f estPose = visual_odometry.calcPose(frameFloat);
        //visual_odometry.addFrameToStack(frameFloat, realPose);
        //visual_odometry.updateMap();

        std::cout << "real pose " << std::endl;
        std::cout << realPose.matrix() << std::endl;
        std::cout << "est pose " << std::endl;
        std::cout << visual_odometry.trackedPose.matrix() << std::endl;

        //cv::imshow("image", frame);
        //cv::imshow("keyframe", keyFrame);
        //cv::imshow("idepth", iDepth);
        //cv::waitKey(30);

        //if(framesTracked >= framesToTrack)
        if(false)
        {

            char depth_filename[500];
            sprintf(depth_filename,"../../desktop_dataset/scene_depth_%03d.yml", frameNumber);

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
