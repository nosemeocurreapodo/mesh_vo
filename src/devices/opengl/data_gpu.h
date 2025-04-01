#pragma once

#include <opencv2/core.hpp>
#include <glad/glad.h>

#include "params.h"

class data_gpu
{
public:

    data_gpu();
    data_gpu(int height, int width, int channels, GLenum datatype, GLint filtertype, GLint wraptype);

    //void cpu_to_gpu(int lvl);
    //void gpu_to_cpu(int lvl);

    void generateMipmaps(int baselvl);

    unsigned int gpuTexture;

    GLint glinternalFormat;
    GLenum glformat;
    GLenum gltype;

private:

};
