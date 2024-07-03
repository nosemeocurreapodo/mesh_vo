#pragma once

#include <opencv2/opencv.hpp>

#include "params.h"

template <typename Type>
class dataCPU
{
public:
    dataCPU(int width, int height, Type nodata_value)
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

            int width_s = int(width / scale);
            int height_s = int(height / scale);

            texture[lvl] = cv::Mat(cv::Size(width_s, height_s), dtype, cv::Scalar(nodata));
        }
    }

    std::array<int, 2> getSize(int lvl)
    {
        return {texture[lvl].cols, texture[lvl].rows};
    }

    void set(Type value, int y, int x, int lvl)
    {
        texture[lvl].at<Type>(y, x) = value;
    }

    void set(Type value, float y, float x, int lvl)
    {
        texture[lvl].at<Type>(int(y), int(x)) = value;
    }

    void setNormalized(Type value, float norm_y, float norm_x, int lvl)
    {
        float y = norm_y*texture[lvl].rows;
        float x = norm_x*texture[lvl].cols;
        texture[lvl].at<Type>(int(y), int(x)) = value;
    }

    void set(Type value, int lvl)
    {
        texture[lvl].setTo(value);
    }

    void set(cv::Mat frame, int lvl)
    {
        cv::Mat frame_newtype;
        frame.convertTo(frame_newtype, texture[lvl].type());
        cv::resize(frame_newtype, texture[lvl], texture[lvl].size(), cv::INTER_AREA);
    }

    void set(cv::Mat frame)
    {
        set(frame, 0);
        generateMipmaps();
    }

    Type get(int y, int x, int lvl)
    {
        return texture[lvl].at<Type>(y, x);
    }

    Type get(float y, float x, int lvl)
    {
        // bilinear interpolation (-2 because the read the next pixel)
        int _x = std::min(std::max(int(x), 0), texture[lvl].cols-2);
        int _y = std::min(std::max(int(y), 0), texture[lvl].rows-2);
        float dx = x - _x;
        float dy = y - _y;

        float weight_tl = (1.0 - dx) * (1.0 - dy);
        float weight_tr = (dx) * (1.0 - dy);
        float weight_bl = (1.0 - dx) * (dy);
        float weight_br = (dx) * (dy);

        Type pix = weight_tl * texture[lvl].at<Type>(_y, _x) +
                   weight_tr * texture[lvl].at<Type>(_y, _x + 1) +
                   weight_bl * texture[lvl].at<Type>(_y + 1, _x) +
                   weight_br * texture[lvl].at<Type>(_y + 1, _x + 1);

        return pix;
        // return texture[lvl].at<Type>(int(y), int(x));
    }

    Type getNormalized(float norm_y, float norm_x, int lvl)
    {
        float y = norm_y*texture[lvl].rows;
        float x = norm_x*texture[lvl].cols;
        return get(y, x, lvl);
    }

    cv::Mat &get(int lvl)
    {
        return texture[lvl];
    }

    void reset(int lvl)
    {
        texture[lvl].setTo(nodata);
    }

    void reset()
    {
        for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
            reset(lvl);
    }

    void generateMipmaps(int baselvl = 0)
    {
        for (int lvl = baselvl + 1; lvl < MAX_LEVELS; lvl++)
        {
            cv::resize(texture[baselvl], texture[lvl], texture[lvl].size(), cv::INTER_AREA);
        }
    }

    void copyTo(dataCPU &data)
    {
        data.nodata = nodata;
        for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            texture[lvl].copyTo(data.texture[lvl]);
        }
    }

    cv::Mat getNoDataMask(int lvl)
    {
        cv::Mat matnodata;
        cv::inRange(texture[lvl], nodata, nodata, matnodata);
        return matnodata;
    }

    float getPercentNoData(int lvl)
    {
        cv::Mat matnodata = getNoDataMask(lvl);
        return float(cv::countNonZero(matnodata)) / (texture[lvl].cols * texture[lvl].rows);
    }


    void show(std::string window_name, int lvl)
    {
        cv::Mat toShow;
        texture[lvl].copyTo(toShow);
        
        cv::Mat mask, maskInv;
        cv::inRange(toShow, nodata, nodata, mask);
        cv::bitwise_not(mask, maskInv);

        //cv::Mat zeros = cv::Mat(texture[lvl].rows, texture[lvl].cols, CV_32FC1, cv::Scalar(0));

        cv::normalize(toShow, toShow, 1.0, 0.0, cv::NORM_MINMAX, CV_32F, maskInv);

        cv::Mat nodataImage; 
        toShow.copyTo(nodataImage);
        nodataImage.setTo(1.0, mask);

        if(toShow.channels() == 1)
        {
            std::vector<cv::Mat> tomerge;
            tomerge.push_back(toShow);
            tomerge.push_back(toShow);
            tomerge.push_back(nodataImage);
            cv::merge(tomerge, toShow);
        }
        
        cv::resize(toShow, toShow, texture[0].size());
        cv::imshow(window_name, toShow);
        cv::waitKey(30);
    }

    Type nodata;

private:
    cv::Mat texture[MAX_LEVELS];
};
