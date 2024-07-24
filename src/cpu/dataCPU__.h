#pragma once

#include <opencv2/opencv.hpp>

template <typename Type>
class dataCPU
{
public:
    dataCPU(int width, int height, Type nodata_value)
    {
        nodata = nodata_value;
        int dtype;

        if (typeid(Type) == typeid(uchar))
        {
            dtype = CV_8UC1;
        }
        if (typeid(Type) == typeid(char))
        {
            dtype = CV_8SC1;
        }
        if (typeid(Type) == typeid(int))
        {
            dtype = CV_32SC1;
        }
        if (typeid(Type) == typeid(float))
        {
            dtype = CV_32FC1;
        }
        if (typeid(Type) == typeid(double))
        {
            dtype = CV_64FC1;
        }
        if (typeid(Type) == typeid(std::array<float, 3>))
        {
            dtype = CV_32FC3;
        }
        if( typeid(Type) == typeid(std::array<int, 3>))
        {
            dtype = CV_32SC3;
        }

        int lvl = 0;
        while (true)
        {
            float scale = std::pow(2.0f, float(lvl));

            int width_s = int(width / scale);
            int height_s = int(height / scale);

            if (width_s <= 0 || height_s <= 0)
                break;

            texture[lvl] = cv::Mat(cv::Size(width_s, height_s), dtype);
            set(nodata, lvl);
            lvl++;
        }

        max_lvl = lvl;
    }

    dataCPU(const dataCPU &other)
    {
        nodata = other.nodata;
        max_lvl = other.max_lvl;
        // std::fill(std::begin(texture), std::end(texture), nullptr);
        for (int lvl = 0; lvl < max_lvl; lvl++)
        {
            other.texture[lvl].copyTo(texture[lvl]);
        }
    }

    dataCPU &operator=(const dataCPU &other)
    {
        if (this != &other)
        {
            nodata = other.nodata;
            max_lvl = other.max_lvl;

            for (int lvl = 0; lvl < max_lvl; lvl++)
            {
                other.texture[lvl].copyTo(texture[lvl]);
            }
        }
        return *this;
    }

    ~dataCPU()
    {
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
        float y = norm_y * texture[lvl].rows;
        float x = norm_x * texture[lvl].cols;
        texture[lvl].at<Type>(int(y), int(x)) = value;
    }

    void set(Type value, int lvl)
    {
        texture[lvl].setTo(value);
    }

    void set(Type *data, int lvl)
    {
        std::memcpy(texture[lvl].data, data, sizeof(Type) * getSize(lvl)[0] * getSize(lvl)[1]);
    }

    void set(Type *data)
    {
        set(data, 0);
        generateMipmaps();
    }

    void setSmooth(int lvl, float start = 0.5, float end = 1.0)
    {
        for (int y = 0; y < getSize(lvl)[1]; y++)
        {
            for (int x = 0; x < getSize(lvl)[0]; x++)
            {
                float val = start + (end - start) * float(y) / (getSize(lvl)[1] - 1.0);
                set(val, y, x, lvl);
            }
        }
    }

    void setRandom(int lvl, float min = 0.5, float max = 1.0)
    {
        for (int y = 0; y < getSize(lvl)[1]; y++)
        {
            for (int x = 0; x < getSize(lvl)[0]; x++)
            {
                float val = (max - min) * float(rand() % 1000) / 1000.0 + min;
                set(val, y, x, lvl);
            }
        }
    }

    Type get(int y, int x, int lvl)
    {
        return texture[lvl].at<Type>(y, x);
    }

    Type get(float y, float x, int lvl)
    {
        // bilinear interpolation (-2 because the read the next pixel)
        // int _x = std::min(std::max(int(x), 0), texture[lvl].cols-2);
        // int _y = std::min(std::max(int(y), 0), texture[lvl].rows-2);
        if (y > texture[lvl].rows - 2 || x > texture[lvl].cols - 2)
            return texture[lvl].at<Type>(int(y), int(x));

        int _x = int(x);
        int _y = int(y);
        float dx = x - _x;
        float dy = y - _y;

        float weight_tl = (1.0 - dx) * (1.0 - dy);
        float weight_tr = (dx) * (1.0 - dy);
        float weight_bl = (1.0 - dx) * (dy);
        float weight_br = (dx) * (dy);

        Type pix = weight_tl * get(_y, _x, lvl) +
                   weight_tr * get(_y, _x + 1, lvl) +
                   weight_bl * get(_y + 1, _x, lvl) +
                   weight_br * get(_y + 1, _x + 1, lvl);

        return pix;
        // return texture[lvl].at<Type>(int(y), int(x));
    }

    Type getNormalized(float norm_y, float norm_x, int lvl)
    {
        float y = norm_y * texture[lvl].rows;
        float x = norm_x * texture[lvl].cols;
        return get(y, x, lvl);
    }

    Type *get(int lvl)
    {
        return texture[lvl].data;
    }

    void generateMipmaps(int baselvl = 0)
    {
        for (int lvl = baselvl + 1; lvl < MAX_LEVELS; lvl++)
        {
            cv::resize(texture[baselvl], texture[lvl], texture[lvl].size(), cv::INTER_AREA);
        }
    }

    float getPercentNoData(int lvl)
    {
        cv::Mat matnodata = getNoDataMask(lvl);
        return float(cv::countNonZero(matnodata)) / (texture[lvl].cols * texture[lvl].rows);
    }

    dataCPU sub(dataCPU &other, int lvl)
    {
        dataCPU<Type> result(getSize(0)[1], getSize(0)[0], nodata);
        for (int y = 0; y < getSize(lvl)[1]; y++)
            for (int x = 0; x < getSize(lvl)[0]; x++)
            {
                Type p1 = get(y, x, lvl);
                Type p2 = other.get(y, x, lvl);
                if(p1 == nodata || p2 == other.nodata)
                    continue;
                Type res = p1 - p2;
                result.set(res, y, x, lvl);
            }
        return result;
    }

    Type nodata;

private:
    cv::Mat getNoDataMask(int lvl)
    {
        cv::Mat matnodata;
        cv::inRange(texture[lvl], nodata, nodata, matnodata);
        return matnodata;
    }

    cv::Mat texture[MAX_LEVELS];
    int max_lvl;
};
