#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "utils/convertAhandaPovRayToStandard.h"

#include "mesh_vo.h"

//#include "DepthEstimator/Common/se3.h"
#include <Eigen/Core>
#include "sophus/se3.hpp"

int main(void)
{
    int frameId = 0;
    int frameNumber = 0;
    int frameCounterDirection = 1;
    float frameTimeStamp = 0.0;

    int framesToTrack = 50;//rand() % 10 + 50;
    int framesTracked = 0;

    int width = 640;
    int height = 480;

    //cv::Mat frame, frameGray, frameGrayResized, frameGrad, cameraMatrix, R, T;

    float fx, fy, cx, cy;
    fx = 481.20; fy = 480.0; cx = 319.5; cy = 239.5;

    cv::Mat iniImageUchar = cv::imread("../../../Trajectory_30_seconds/scene_000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat iniImageFloat;
    iniImageUchar.convertTo(iniImageFloat, CV_32FC1);
    Sophus::SE3f iniPose = readPose("../../../Trajectory_30_seconds/scene_000.txt");

    mesh_vo visual_odometry(fx, fy, cx, cy, width, height);

    cv::Mat frameFloat;
    cv::Mat keyFrame = iniImageFloat;
    Sophus::SE3f keyframePose = iniPose;

    visual_odometry.setKeyFrame(keyFrame);

    int num_past_poses = 5;
    Sophus::SE3f post_poses[num_past_poses];


    Eigen::MatrixXf covariance = Eigen::MatrixXf::Zero(6,6);
    int covariance_num = 0;


    while(1){
        frameId++;
        framesTracked++;
        frameTimeStamp += 0.03;
        frameNumber += frameCounterDirection;
        if(frameNumber > 598)
            frameCounterDirection = -1;
        if(frameNumber < 2)
            frameCounterDirection = 1;

        char image_filename[500];
        char RT_filename[500];

        //file name
        sprintf(image_filename,"../../../Trajectory_30_seconds/scene_%03d.png", frameNumber);
        sprintf(RT_filename,"../../../Trajectory_30_seconds/scene_%03d.txt", frameNumber);

                //read image and pose
        Sophus::SE3f pose = readPose(RT_filename);
        cv::Mat frameUchar = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);
        frameUchar.convertTo(frameFloat, CV_32FC1);

        Sophus::SE3f realPose = pose*keyframePose.inverse();
        for(int i=0; i < num_past_poses-1; i++)
            post_poses[i] = post_poses[i+1];
        post_poses[num_past_poses-1] = realPose;

        /*
        Sophus::SE3f poseDiff = post_poses[num_past_poses-1]*post_poses[num_past_poses-2].inverse();
        Eigen::Matrix<float, 6, 1> inc = Sophus::SE3f::log(poseDiff);

        covariance_num++;
        for(int j = 0; j < 6; j++)
            for(int k = 0; k < 6; k++)
                covariance(j,k) += inc(j)*inc(k);

        std::cout << "covariance num " << covariance_num << std::endl;
        std::cout << "covariance " << covariance/covariance_num << std::endl;
        */


        depthEstimator.update(frameFloat, realPose);//<---- importante!!

        //poseEstimator.framePose = post_poses[num_past_poses-2];
        visual_odometry.updatePose(frameFloat);

        std::cout << "real pose " << std::endl;
        std::cout << realPose.matrix() << std::endl;
        std::cout << "est pose " << std::endl;
        std::cout << poseEstimator.framePose.matrix() << std::endl;

        cv::Mat iDepth = depthEstimator.calculateIdepth();
        poseEstimator.setIdepth(iDepth);

        cv::imshow("image", frame);
        cv::imshow("keyframe", keyFrame);
        cv::imshow("idepth", iDepth);
        cv::waitKey(30);


        //if( cv::norm(pose.getTranslation() - keyframePose.getTranslation()) < 0.1 && framesTracked > framesToTrack)
        if(framesTracked == framesToTrack)
        //if(false)
        {
            printf("frames tracked %d\n", framesTracked);
            framesTracked = 0;
            framesToTrack =  rand() % 5 + 25;//25;

            depthEstimator.setKeyFrame(frame);
            poseEstimator.setKeyFrame(frame);
            //depthEstimator.initializeRandomly(frame.get());
            keyFrame = frame;
            keyframePose = pose;

            depthEstimator.reset();
        }
    }

    return 1;
}
