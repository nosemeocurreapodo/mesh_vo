#pragma once

#include <opencv2/opencv.hpp>

#include "cpu/dataCPU.h"

inline void show(dataCPU<float> &data, std::string window_name, int lvl)
{
    std::array<int, 2> datasize = data.getSize(lvl);

    cv::Mat toShow(datasize[1], datasize[0], CV_32FC1); // (uchar*)data.get(lvl))
    
    for(int y = 0; y < datasize[1]; y++)
    {
        for(int x = 0; x < datasize[0]; x++)
        {
            toShow.at<float>(y, x) = data.get(y, x, lvl);
        }
    }

    cv::Mat mask, maskInv;
    cv::inRange(toShow, data.nodata, data.nodata, mask);
    cv::bitwise_not(mask, maskInv);

    // cv::Mat zeros = cv::Mat(texture[lvl].rows, texture[lvl].cols, CV_32FC1, cv::Scalar(0));

    cv::normalize(toShow, toShow, 1.0, 0.0, cv::NORM_MINMAX, CV_32F, maskInv);

    cv::Mat nodataImage;
    toShow.copyTo(nodataImage);
    nodataImage.setTo(1.0, mask);

    if (toShow.channels() == 1)
    {
        std::vector<cv::Mat> tomerge;
        tomerge.push_back(toShow);
        tomerge.push_back(toShow);
        tomerge.push_back(nodataImage);
        cv::merge(tomerge, toShow);
    }

    std::array<int, 2> datasize0 = data.getSize(0);

    cv::resize(toShow, toShow, cv::Size(datasize0[0], datasize0[1]));
    cv::imshow(window_name, toShow);
    cv::waitKey(30);
}
