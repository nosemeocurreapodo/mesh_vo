#pragma once

#include <opencv2/opencv.hpp>

#include "params.h"

template <typename Type>
class data_cpu
{
public:
    data_cpu(Type nodata_value)
    {
        nodata = nodata_value;

        int dtype = CV_8UC1;

        if (typeid(Type) == typeid(uchar))
            dtype = CV_8UC1;
        if (typeid(Type) == typeid(char))
            dtype = CV_8SC1;
        if (typeid(Type) == typeid(int))
            dtype = CV_32SC1;
        if (typeid(Type) == typeid(float))
            dtype = CV_32FC1;
        if (typeid(Type) == typeid(double))
            dtype = CV_64FC1;

        for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            float scale = std::pow(2.0f, float(lvl));

            int width_s = int(MAX_WIDTH / scale);
            int height_s = int(MAX_HEIGHT / scale);

            texture[lvl] = cv::Mat(height_s, width_s, dtype, cv::Scalar(nodata));
        }
    }

    void set(Type value, int y, int x, int lvl)
    {
        texture[lvl].at<Type>(y, x) = value;
    }

    void set(Type value, float y, float x, int lvl)
    {
        texture[lvl].at<Type>(int(y), int(x)) = value;
    }

    void set(Type value, int lvl)
    {
        texture[lvl].setTo(value);
    }

    Type get(int y, int x, int lvl)
    {
        return texture[lvl].at<Type>(y, x);
    }

    Type get(float y, float x, int lvl)
    {
        return texture[lvl].at<Type>(int(y), int(x));
    }

    cv::Mat& get(int lvl)
    {
        return texture[lvl];
    }

    void generateMipmaps(int baselvl)
    {
        for (int lvl = baselvl + 1; lvl < MAX_LEVELS; lvl++)
        {
            float scale = std::pow(2.0f, float(lvl));

            int width_s = int(MAX_WIDTH / scale);
            int height_s = int(MAX_HEIGHT / scale);

            cv::resize(texture[baselvl], texture[lvl], cv::Size(width_s, height_s), cv::INTER_LANCZOS4);
        }
    }

    void copyTo(data_cpu &data)
    {
        data.nodata = nodata;
        for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            texture[lvl].copyTo(data.texture[lvl]);
        }
    }

    void show(std::string window_name, int lvl)
    {
        cv::Mat toShow;
        cv::resize(texture[lvl], toShow, texture[0].size());
        cv::normalize(toShow, toShow, 1.0, 0.0, cv::NORM_MINMAX, CV_32F);
        if (toShow.channels() == 2)
        {
            cv::Mat zeros = cv::Mat(texture[0].rows, texture[0].cols, texture[0].type(), cv::Scalar(0));
            std::vector<cv::Mat> tomerge;
            tomerge.push_back(toShow);
            tomerge.push_back(zeros);
            cv::merge(tomerge, toShow);
        }
        cv::imshow(window_name, toShow);
        cv::waitKey(30);
    }

    Type nodata;

private:
    cv::Mat texture[MAX_LEVELS];
};
