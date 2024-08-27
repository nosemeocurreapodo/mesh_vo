#pragma once

#include <opencv2/opencv.hpp>
#include "cpu/dataCPU.h"
#include "common/HGEigenSparse.h"

inline void saveH(HGEigenSparse &data, std::string file_name)
{
    std::map<int, int> obsParamIds = data.getParamIds();
    Eigen::SparseMatrix<float> H = data.getH(obsParamIds);

    cv::Mat toShow(H.outerSize(), H.innerSize(), CV_32FC1, 0.0);

    for (int k = 0; k < H.outerSize(); ++k)
        for (Eigen::SparseMatrix<float>::InnerIterator it(H, k); it; ++it)
        {
            // it.value();
            // it.row();   // row index
            // it.col();   // col index (here it is equal to k)
            // it.index(); // inner index, here it is equal to it.row()

            int y = it.row();
            int x = it.col();
            float val1 = it.value();
            float val = std::log10(std::fabs(val1) + 1.0);

            toShow.at<float>(y, x) = val;
        }

    // toShow = cv::abs(toShow);
    // double min, max;
    // cv::minMaxLoc(toShow, &min, &max);
    // toShow = toShow / min;
    cv::normalize(toShow, toShow, 255.0, 0.0, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(file_name, toShow);

    // cv::imshow(file_name, toShowLog);
    // cv::waitKey(30);
}

inline void show(dataCPU<float> &data, std::string window_name, bool colorize, int lvl)
{
    std::array<int, 2> datasize = data.getSize(lvl);

    cv::Mat toShow(datasize[1], datasize[0], CV_32FC1); // (uchar*)data.get(lvl))

    for (int y = 0; y < datasize[1]; y++)
    {
        for (int x = 0; x < datasize[0]; x++)
        {
            toShow.at<float>(y, x) = data.get(y, x, lvl);
        }
    }

    cv::Mat mask, maskInv;
    cv::inRange(toShow, data.nodata, data.nodata, mask);
    cv::bitwise_not(mask, maskInv);

    // float min, max;
    // cv::minMaxIdx(toShow, min, max, maskInv);
    // cv::threshold(toShow, toShow);

    // cv::Mat zeros = cv::Mat(texture[lvl].rows, texture[lvl].cols, CV_32FC1, cv::Scalar(0));

    cv::normalize(toShow, toShow, 255.0, 0.0, cv::NORM_MINMAX, CV_8UC1, maskInv);
    toShow.setTo(0, mask);

    cv::Mat toShow2;
    if (colorize)
    {
        cv::applyColorMap(toShow, toShow2, cv::COLORMAP_JET);
    }
    else
    {
        toShow2 = toShow;
    }

    std::array<int, 2> datasize0 = data.getSize(0);

    cv::resize(toShow2, toShow2, cv::Size(datasize0[0], datasize0[1]));

    cv::imshow(window_name, toShow2);
    cv::waitKey(30);
    /*
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
    */
}
