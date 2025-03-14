#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/convertAhandaPovRayToStandard.h"
#include "common/camera.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "visualOdometry.h"
#include "visualOdometryThreaded.h"

inline bool fileExist(const std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "usage: " << argv[0] << " /path/to/dataset" << std::endl;
        return 0;
    }

    std::string dataset_path = argv[1];

    int frameNumber = 0;
    int frameCounterDirection = 1;

    int framesTracked = 0;

    cv::Mat imageMat = cv::imread(dataset_path + "images/scene_000.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat idepthMat = cv::imread(dataset_path + "depths/scene_000.png", cv::IMREAD_GRAYSCALE);
    Sophus::SE3f initPose = readPose(dataset_path + "poses/scene_000.txt");

    if (std::is_same<imageType, uchar>::value)
        imageMat.convertTo(imageMat, CV_8UC1);
    else if (std::is_same<imageType, int>::value)
        imageMat.convertTo(imageMat, CV_32SC1);
    else if (std::is_same<imageType, float>::value)
        imageMat.convertTo(imageMat, CV_32FC1);

    int width = imageMat.cols;
    int height = imageMat.rows;

    float fx = height;     // 481.20;
    float fy = height;     // 480.0;
    float cx = width / 2;  // 319.5;
    float cy = height / 2; // 239.5;

    cameraType cam(fx, fy, cx, cy, width, height);

    // to avoid idepth = 0 in the data
    // idepthMat = idepthMat + 1.0;
    // idepthMat.convertTo(idepthMat, CV_32FC1);
    // cv::resize(idepthMat, idepthMat, cv::Size(cam.width, cam.height), cv::INTER_AREA);

    dataCPU<imageType> image(width, height, 0);
    // dataCPU<float> idepth(IMAGE_WIDTH, IMAGE_HEIGHT, -1.0);

    image.set((imageType *)imageMat.data);
    // idepth.set((float*)idepthMat.data);

    /*
    //get corner from image
    int maxCorners = MESH_WIDTH*MESH_HEIGHT;
    std::vector<cv::Point2f> corners;
    double qualityLevel = 0.000000001;
    double minDistance = 20;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    cv::goodFeaturesToTrack( idepthMat,
                         corners,
                         maxCorners,
                         qualityLevel,
                         minDistance,
                         cv::Mat(),
                         blockSize,
                         gradientSize,
                         useHarrisDetector,
                         k );

    std::vector<vec2<float>> pixels;
    std::vector<float> idepths;

    for(auto corner : corners)
    {
        vec2<float> pixel(corner.x, corner.y);
        float id = idepth.get(pixel(1), pixel(0), 0);

        pixels.push_back(pixel);
        idepths.push_back(id);
    }
    */

    visualOdometryThreaded odometry(width, height);
    odometry.init(image, SE3f(), cam);

    // odometry.initScene(image, pixels, idepths, Sophus::SE3f());
    // odometry.init(image, idepth, Sophus::SE3f());
    // odometry.init(image, SE3f());

    while (1)
    {
        framesTracked++;
        frameNumber += frameCounterDirection;
        if (frameNumber > 598)
            frameCounterDirection = -1;
        if (frameNumber < 2)
            frameCounterDirection = 1;
        // if(frameNumber > 100)
        //     return 1;

        char image_filename[500];
        char RT_filename[500];

        // file name
        sprintf(image_filename, "images/scene_%03d.png", frameNumber);
        sprintf(RT_filename, "poses/scene_%03d.txt", frameNumber);

        std::string image_path = dataset_path + image_filename;
        std::string pose_path = dataset_path + RT_filename;

        cv::Mat imageMat = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        SE3f realPose = readPose(pose_path) * initPose.inverse();
        // scale the translation, so that the initial map has a mean depth of 1.0
        realPose.translation() = realPose.translation() * 0.25;

        if (std::is_same<imageType, uchar>::value)
            imageMat.convertTo(imageMat, CV_8UC1);
        else if (std::is_same<imageType, int>::value)
            imageMat.convertTo(imageMat, CV_32SC1);
        else if (std::is_same<imageType, float>::value)
            imageMat.convertTo(imageMat, CV_32FC1);
        // cv::resize(imageMat, imageMat, cv::Size(mesh_vo::image_width, mesh_vo::image_height), cv::INTER_AREA);

        image.set((imageType *)imageMat.data);
        odometry.locAndMap(image);

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return 1;
}
