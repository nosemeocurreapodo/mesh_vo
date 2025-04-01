#include "data_gpu.h"

#include <iostream>

data_gpu::data_gpu()
{

}

data_gpu::data_gpu(int height, int width, int channels, GLenum datatype, GLint filtertype, GLint wraptype)
{
    gltype = datatype;

    if(channels == 1)
    {
        glformat = GL_RED;
    }
    if(channels == 2)
    {
        glformat = GL_RG;
    }
    if(channels == 3)
    {
        glformat = GL_RGB;
    }
    if(channels == 4)
    {
        glformat = GL_RGBA;
    }

    if(gltype == GL_FLOAT)
    {
        if(channels == 1)
        {
            glinternalFormat = GL_R32F;
        }
        if(channels == 2)
        {
            glinternalFormat = GL_RG32F;
        }
        if(channels == 3)
        {
            glinternalFormat = GL_RGB32F;
        }
        if(channels == 4)
        {
            glinternalFormat = GL_RGBA32F;
        }
    }

    if(gltype == GL_UNSIGNED_BYTE)
    {
        if(channels == 1)
        {
            glinternalFormat = GL_R8;
        }
        if(channels == 2)
        {
            glinternalFormat = GL_RG8;
        }
        if(channels == 3)
        {
            glinternalFormat = GL_RGB8;
        }
        if(channels == 4)
        {
            glinternalFormat = GL_RGBA8;
        }
    }

    if(gltype == GL_INT)
    {
        if(channels == 1)
        {
            glinternalFormat = GL_R32I;
        }
        if(channels == 2)
        {
            glinternalFormat = GL_RG32I;
        }
        if(channels == 3)
        {
            glinternalFormat = GL_RGB32I;
        }
        if(channels == 4)
        {
            glinternalFormat = GL_RGBA32I;
        }
    }

    glGenTextures(1, &gpuTexture);
    glBindTexture(GL_TEXTURE_2D, gpuTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filtertype);//GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filtertype);//GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wraptype);// GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wraptype);//GL_MIRRORED_REPEAT);//border los de afuera son erroneos
    float color[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);
    glTexImage2D(GL_TEXTURE_2D, 0, glinternalFormat, width, height, 0, glformat, gltype, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

}

/*
void data::cpu_to_gpu(int lvl)
{
    cv::Mat flipped;
    cv::flip(cpuTexture[lvl], flipped, 0);
    glBindTexture(GL_TEXTURE_2D, gpuTexture);
    glTexSubImage2D(GL_TEXTURE_2D, lvl, 0, 0, cpuTexture[lvl].cols, cpuTexture[lvl].rows, glformat, gltype, (float *)flipped.data);
}

void data::gpu_to_cpu(int lvl)
{
    //glBindTexture(GL_TEXTURE_2D, gpuTexture);
    //glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, (float *)cpuTexture[lvl].data);
    //cv::flip(cpuTexture[lvl],cpuTexture[lvl],0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, gpuTexture, lvl);
    glReadPixels( 0, 0, cpuTexture[lvl].cols, cpuTexture[lvl].rows, glformat, gltype, (float *)cpuTexture[lvl].data);
    cv::flip(cpuTexture[lvl],cpuTexture[lvl],0);
}
*/

void data::generateMipmaps(int baselvl)
{
    glBindTexture(GL_TEXTURE_2D, gpuTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, baselvl);
    glGenerateMipmap(GL_TEXTURE_2D);
}




