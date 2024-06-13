#pragma once

#include <opencv2/core.hpp>
#include <glad/glad.h>

#include "params.h"

class data_cpu
{
public:

    data_cpu();
    data_cpu(int height, int width, int channels, int dtype);

    void generateMipmaps(int baselvl);

    cv::Mat texture[MAX_LEVELS];

    int cvtype;

private:

};
