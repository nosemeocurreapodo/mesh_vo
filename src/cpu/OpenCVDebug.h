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

template <int width, int height, typename type>
inline void show(dataCPU<width, height, type> &data, std::string window_name, bool colorize, bool upsample, int channel, int lvl)
{
    int lvlWidth = data.lvlWidth[lvl];
    int lvlHeight = data.lvlHeight[lvl];

    cv::Mat toShow(lvlHeight, lvlWidth, CV_32FC1); // (uchar*)data.get(lvl))

    type nodata = data.nodata;

    for (int y = 0; y < lvlHeight; y++)
    {
        for (int x = 0; x < lvlWidth; x++)
        {
            type in_val = data.get(y, x, lvl);
            // cv::Vec3f out_val(in_val(0), in_val(1), in_val(2));
            // toShow.at<cv::Vec3f>(y, x) = out_val;
            toShow.at<float>(y, x) = in_val(channel);
        }
    }

    cv::Mat mask, maskInv;
    cv::inRange(toShow, nodata(channel), nodata(channel), mask);
    cv::bitwise_not(mask, maskInv);

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

    if (upsample)
    {
        cv::resize(toShow2, toShow2, cv::Size(width, height));
    }

    cv::imshow(window_name, toShow2);
    cv::waitKey(30);
}
