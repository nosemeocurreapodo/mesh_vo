#include "data.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

data::data()
{

}

data::data(int height, int width, int channels, GLenum datatype, GLint filtertype, GLint wraptype)
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
            cvtype = CV_32FC1;
        }
        if(channels == 2)
        {
            glinternalFormat = GL_RG32F;
            cvtype = CV_32FC2;
        }
        if(channels == 3)
        {
            glinternalFormat = GL_RGB32F;
            cvtype = CV_32FC3;
        }
        if(channels == 4)
        {
            glinternalFormat = GL_RGBA32F;
            cvtype = CV_32FC4;
        }
    }

    if(gltype == GL_UNSIGNED_BYTE)
    {
        if(channels == 1)
        {
            glinternalFormat = GL_R8;
            cvtype = CV_8UC1;
        }
        if(channels == 2)
        {
            glinternalFormat = GL_RG8;
            cvtype = CV_8UC2;
        }
        if(channels == 3)
        {
            glinternalFormat = GL_RGB8;
            cvtype = CV_8UC3;
        }
        if(channels == 4)
        {
            glinternalFormat = GL_RGBA8;
            cvtype = CV_8UC4;
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

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        float scale = std::pow(2.0f, float(lvl));

        int width_s = int(MAX_WIDTH/scale);
        int height_s = int(MAX_HEIGHT/scale);

        cpuTexture[lvl] = cv::Mat(height_s, width_s, cvtype, cv::Scalar(0));
    }
}

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

void data::generateMipmapsCPU(int baselvl)
{
    for(int lvl = baselvl+1; lvl < MAX_LEVELS; lvl++)
    {
        float scale = std::pow(2.0f, float(lvl));

        int width_s = int(MAX_WIDTH/scale);
        int height_s = int(MAX_HEIGHT/scale);

        cv::resize(cpuTexture[baselvl],cpuTexture[lvl], cv::Size(width_s, height_s), cv::INTER_AREA);
    }
}

void data::generateMipmapsGPU(int baselvl)
{
    glBindTexture(GL_TEXTURE_2D, gpuTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, baselvl);
    glGenerateMipmap(GL_TEXTURE_2D);
}




