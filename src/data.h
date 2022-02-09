#pragma once

#include <opencv2/core.hpp>
#include <glad/glad.h>

#include "params.h"

class data
{
public:

    data();
    data(int height, int width, int channels, GLenum datatype, GLint filtertype, GLint wraptype);

    void cpu_to_gpu(int lvl);
    void gpu_to_cpu(int lvl);

    void generateMipmapsCPU(int baselvl);
    void generateMipmapsGPU(int baselvl);

    unsigned int gpuTexture;
    cv::Mat cpuTexture[MAX_LEVELS];

    GLint glinternalFormat;
    GLenum glformat;
    GLenum gltype;

    int cvtype;

private:

};
