#include "data.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

data_cpu::data_cpu()
{

}

data_cpu::data_cpu(int height, int width, int channels, int dtype)
{
    cvtype = dtype;

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        float scale = std::pow(2.0f, float(lvl));

        int width_s = int(MAX_WIDTH/scale);
        int height_s = int(MAX_HEIGHT/scale);

        texture[lvl] = cv::Mat(height_s, width_s, cvtype, cv::Scalar(0));
    }
}

void data_cpu::generateMipmaps(int baselvl)
{
    for(int lvl = baselvl+1; lvl < MAX_LEVELS; lvl++)
    {
        float scale = std::pow(2.0f, float(lvl));

        int width_s = int(MAX_WIDTH/scale);
        int height_s = int(MAX_HEIGHT/scale);

        cv::resize(texture[baselvl], texture[lvl], cv::Size(width_s, height_s), cv::INTER_LANCZOS4);
    }
}
