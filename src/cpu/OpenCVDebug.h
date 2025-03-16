#pragma once

#include <opencv2/opencv.hpp>
#include "cpu/dataCPU.h"
#include "common/DenseLinearProblem.h"

static void saveH(DenseLinearProblem &data, std::string file_name)
{
    /*
    std::map<int, int> obsParamIds = data.getObservedParamIds();
    Eigen::SparseMatrix<float> H = data.getHSparse(obsParamIds);

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
    */

    // cv::imshow(file_name, toShowLog);
    // cv::waitKey(30);
}

static cv::Mat prepareToShow(dataCPU<float> &data, bool colorize)
{
    cv::Mat toShow(data.height, data.width, CV_32FC1);

    std::memcpy(toShow.data, data.get(), sizeof(float) * data.width * data.height);

    cv::Mat mask, maskInv;
    cv::inRange(toShow, data.nodata, data.nodata, mask);
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

    return toShow2;
}

static void show(dataCPU<float> &data, std::string window_name)
{
    cv::Mat toShow = prepareToShow(data, false);
    cv::imshow(window_name, toShow);
    cv::waitKey(30);
}

static void show(std::vector<dataCPU<float>> &data, std::string window_name)
{
    for (int i = 0; i < data.size(); i++)
    {
        assert(data[i].width == data[0].width && data[i].height == data[0].height);
    }

    int nImagesWidth = 4;
    int nImagesHeight = int(data.size() / nImagesWidth);
    if (data.size() % nImagesWidth != 0)
        nImagesHeight += 1;

    cv::Mat toShow(data[0].height * nImagesHeight, data[0].width * nImagesWidth, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < data.size(); i++)
    {
        cv::Mat toShow2 = prepareToShow(data[i], false);
        toShow2.copyTo(toShow(cv::Rect((i % nImagesWidth) * toShow2.cols, int(i / nImagesWidth) * toShow2.rows, toShow2.cols, toShow2.rows)));
        // toShow(cv::Rect((i % nImagesWidth) * data[0].width, int(i / nImagesWidth) * data[0].height, data[0].width, data[0].height)) = toShow2;
    }

    // cv::Mat toShow = prepareToShow(data[0], false);

    cv::imshow(window_name, toShow);
    cv::waitKey(30);
}
