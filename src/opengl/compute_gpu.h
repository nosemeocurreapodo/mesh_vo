#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

//#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/compute.h>
//#include <learnopengl/feedback_shader.h>
//#include <learnopengl/camera.h>

class compute_gpu
{
public:
    compute_gpu()
    {
            vertexViewCountShader.init("isVertexInView.vs","isVertexInView.gs","isVertexInView.fs");
    vertexViewCountShader.use();

    errorShader.init("error.vs", "error.fs");
    errorShader.use();
    errorShader.setInt("keyframe", 0);
    errorShader.setInt("frame", 1);

    errorVertexShader.init("errorVertex.vs", "errorVertex.gs", "errorVertex.fs");
    errorVertexShader.use();
    errorVertexShader.setInt("keyframe", 0);
    errorVertexShader.setInt("frame", 1);

    jacobianPoseShader.init("JPose.vs", "JPose.fs");
    jacobianPoseShader.use();
    jacobianPoseShader.setInt("keyframe", 0);
    jacobianPoseShader.setInt("frame", 1);
    jacobianPoseShader.setInt("frameDer", 2);

    jacobianPoseShader_v2.init("JPose.vs", "JPose_v2.fs");
    jacobianPoseShader_v2.use();
    jacobianPoseShader_v2.setInt("keyframe", 0);
    jacobianPoseShader_v2.setInt("frame", 1);
    jacobianPoseShader_v2.setInt("frameDer", 2);

    jacobianMapShader.init("JMap.vs", "JMap.gs", "JMap.fs");
    jacobianMapShader.use();
    jacobianMapShader.setInt("keyframe", 0);
    jacobianMapShader.setInt("frame", 1);
    jacobianMapShader.setInt("frameDer", 2);

    jacobianPoseMapShader.init("JPoseMap.vs", "JPoseMap.gs", "JPoseMap.fs");
    jacobianPoseMapShader.use();
    jacobianPoseMapShader.setInt("keyframe", 0);
    jacobianPoseMapShader.setInt("frame", 1);
    jacobianPoseMapShader.setInt("frameDer", 2);

    idepthShader.init("idepth.vs", "idepth.fs");
    occupancyShader.init("occupancy.vs", "occupancy.fs");

    frameDerivativeShader.init("frameDerivative.vs", "frameDerivative.fs");
    frameDerivativeShader.use();
    frameDerivativeShader.setInt("frame", 0);

    showTextureShader.init("showTexture.vs", "showTexture.fs");
    showTextureShader.use();
    showTextureShader.setInt("frame", 0);

    copyShader.init("copy.vs", "copy.fs");
    copyShader.use();
    copyShader.setInt("inTexture", 0);

    view3DShader.init("view3D.vs", "view3D.fs");
    view3DShader.use();
    view3DShader.setInt("keyframe", 0);

    computeHJPoseAndReduceCShader.init("JPose.cs");
    computeErrorAndReduceCShader.init("error.cs");
    reduceErrorShader.init("reduceError.cs");
    reduceHJPoseShader.init("reduceHJPose.cs");
    reduceRShader.init("reduceRTexture.cs");
    reduceRGBAShader.init("reduceRGBATexture.cs");
    }

private:

    //opengl data
    GLFWwindow* frameWindow;



    unsigned int scene_VBO, scene_VAO, scene_EBO;
    unsigned int frame_VBO, frame_VAO;

    unsigned int framebuffer;
    unsigned int rbo;
    unsigned int feedbackrbo;

    //shaders
    Shader idepthShader;
    Shader occupancyShader;
    Shader vertexViewCountShader;

    Shader errorShader;
    Shader errorVertexShader;
    Shader frameDerivativeShader;
    Shader jacobianPoseShader;
    Shader jacobianPoseShader_v2;
    Shader jacobianPoseMapShader;
    Shader jacobianMapShader;

    Shader copyShader;
    Shader showTextureShader;
    Shader view3DShader;

    Compute computeErrorAndReduceCShader;
    Compute computeHJPoseAndReduceCShader;
    Compute reduceErrorShader;
    Compute reduceHJPoseShader;
    Compute reduceRGBAShader;
    Compute reduceRShader;
};


/*
float mesh_vo::calcOccupancyGPU(Sophus::SE3f framePose, int lvl)
{
    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, occupancyTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcIdepth" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // activate shader
    occupancyShader.use();
    occupancyShader.setMat4("framePose", eigen2glm_mat4(framePose.matrix()));
    occupancyShader.setMat4("opencv2opengl", opencv2opengl);
    occupancyShader.setMat4("projection", projMat[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glBindTexture(GL_TEXTURE_2D, idepthTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, idepth_cpu_data);

    int count = 0;

    for(int index = 0; index < width[lvl]*height[lvl]; index++)
    {
        float idepth = idepth_cpu_data[index];

        if(idepth > 0.0)
            count++;
    }

    return float(count)/(width[lvl]*height[lvl]);
}
*/
/*
void mesh_vo::showDebug(unsigned int frame, Sophus::SE3f framePose, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[0],height[0]);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(-1.0f, -1.0f, -1.0f, -1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, frame);


    // activate shader
    debugShader.use();

    debugShader.setMat4("cameraPose", eigen2glm_mat4(framePose.matrix()));
    debugShader.setMat4("opencv2opengl", opencv2opengl);
    debugShader.setMat4("projection", projMat[lvl]);
    debugShader.setFloat("fx", fx[lvl]);
    debugShader.setFloat("fy", fy[lvl]);
    debugShader.setFloat("cx", cx[lvl]);
    debugShader.setFloat("cy", cy[lvl]);
    debugShader.setFloat("fxinv", fxinv[lvl]);
    debugShader.setFloat("fyinv", fyinv[lvl]);
    debugShader.setFloat("cxinv", cxinv[lvl]);
    debugShader.setFloat("cyinv", cyinv[lvl]);
    debugShader.setFloat("width", width[lvl]);
    debugShader.setFloat("height", height[lvl]);
    debugShader.setInt("lvl", lvl);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(frameWindow);
}
*/


void mesh_vo::jacobianMapTextureGPU(frame &_frame, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vertexIdData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, _frame.error.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, _frame.jp0.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, _frame.jp1.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, _frame.jp2.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, debugData.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0,
                                GL_COLOR_ATTACHMENT1,
                                GL_COLOR_ATTACHMENT2,
                                GL_COLOR_ATTACHMENT3,
                                GL_COLOR_ATTACHMENT4,
                                GL_COLOR_ATTACHMENT5};

    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcHJMap" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(-1.0, -1.0f, -1.0f, -1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeData.image.gpuTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frame.image.gpuTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _frame.der.gpuTexture);

    // activate shader
    jacobianMapShader.use();

    jacobianMapShader.setMat4("framePose", eigen2glm_mat4((_frame.pose*keyframeData.pose.inverse()).matrix()));
    jacobianMapShader.setMat4("opencv2opengl", opencv2opengl);
    jacobianMapShader.setMat4("projection", projMat[lvl]);
    jacobianMapShader.setFloat("fx", fx[lvl]);
    jacobianMapShader.setFloat("fy", fy[lvl]);
    jacobianMapShader.setFloat("cx", cx[lvl]);
    jacobianMapShader.setFloat("cy", cy[lvl]);
    jacobianMapShader.setFloat("fxinv", fxinv[lvl]);
    jacobianMapShader.setFloat("fyinv", fyinv[lvl]);
    jacobianMapShader.setFloat("cxinv", cxinv[lvl]);
    jacobianMapShader.setFloat("cyinv", cyinv[lvl]);
    jacobianMapShader.setFloat("dx", dx[lvl]);
    jacobianMapShader.setFloat("dy", dy[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, 0, 0);

}

void mesh_vo::reduceHJMapGPU(frame &_frame, int lvl)
{
    /*
    vertexIdData.generateMipmapsGPU(lvl);
    primitiveIdData.generateMipmapsGPU(lvl);
    residualData.generateMipmapsGPU(lvl);
    d_I_d_p0Data.generateMipmapsGPU(lvl);
    d_I_d_p1Data.generateMipmapsGPU(lvl);
    d_I_d_p2Data.generateMipmapsGPU(lvl);
*/
    vertexIdData.gpu_to_cpu(lvl);
    _frame.error.gpu_to_cpu(lvl);
    _frame.jp0.gpu_to_cpu(lvl);
    _frame.jp1.gpu_to_cpu(lvl);
    _frame.jp2.gpu_to_cpu(lvl);

    int count = 0;
    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            //std::cout << "pixel index " << index << std::endl;
            //acumular para cada pixel
            float error = _frame.error.cpuTexture[lvl].at<float>(y,x);

            //std::cout << "x " << x << " y " << y << std::endl;
            //std::cout << "error " << error << std::endl;

            if(error != error)
            {
                std::cout << "nand in error " << error << std::endl;
                continue;
            }

            //cada pixel aporta 9 parametros
            //81 coeficientes en H! (pero es simetrico, osea son: 9 diagonales y 8+7+6+5+4+3+2+1(36) no diagonales)
            int vertexID[3];
            vertexID[0] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[0]);
            vertexID[1] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[1]);
            vertexID[2] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[2]);

            //std::cout << "vertexId " << vertexID[0] << " " << vertexID[1] << " " << vertexID[2] << std::endl;

            if(vertexID[0] < 0 || vertexID[1] < 0 || vertexID[2] < 0)
            {
                //std::cout << "something wrong with the vertexId" << std::endl;
                continue;
            }

            //std::cout << "updating vertex " << vertexID[0] << " " << vertexID[1] << " " << vertexID[2] << std::endl;
            //std::cout << "vertices " << vertexID[0] << " " << vertexID[1] << " " << vertexID[2] << std::endl;
            //std::cout << "error " << error << std::endl;

            //std::cout << "incides " << std::endl;
            //for(int i = 0; i < 9; i++)
            //    std::cout << indices[i] << std::endl;

            float J[3];
            J[0] = _frame.jp0.cpuTexture[lvl].at<float>(y,x);
            J[1] = _frame.jp1.cpuTexture[lvl].at<float>(y,x);
            J[2] = _frame.jp2.cpuTexture[lvl].at<float>(y,x);

            //std::cout << "J " << J[0] << " " << J[1] << " " << J[2] << std::endl;

            bool someNand = false;
            for(int i = 0; i < 3; i++)
            {
                if(J[i]!=J[i])
                {
                    std::cout << "J nand " << i << " " << x << " " << y << " " << vertexID[0] << " " << vertexID[1] << " " << vertexID[2] << std::endl;
                    someNand = true;
                }
            }
            if(someNand == true)
                continue;
            count++;
            //ahora si, actualizo las matrices usando los indices de cada vertice
            for(int i = 0; i < 3; i++)
            {
                J_depth(vertexID[i]) += J[i]*error;

                for(int j = i; j < 3; j++)
                {
                    //acc_H_depth(vertexID[i],vertexID[j]) += J[i]*J[j];
                    float jj = J[i]*J[j];
                    H_depth.coeffRef(vertexID[i],vertexID[j]) += jj;
                    H_depth.coeffRef(vertexID[j],vertexID[i]) += jj;
                }
            }
        }
    if(count > 0)
    {
        H_depth /= count;
        J_depth /= count;
    }
}


void mesh_vo::HJMapStackGPU(int lvl)
{
    H_depth.setZero();
    J_depth.setZero();
    int count = 0;
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == true)
        {
            jacobianMapTextureGPU(frameDataStack[i],lvl);
            reduceHJMapGPU(frameDataStack[i], lvl);
            count++;

            showGPU(debugData,lvl);
            cv::waitKey(1000.0);
        }
    }
    if(count > 0)
    {
        H_depth /= count;
        J_depth /= count;
    }
}

void mesh_vo::HJMapGPU(frame &_frame, int lvl)
{
    H_depth.setZero();
    J_depth.setZero();
    jacobianMapTextureGPU(_frame, lvl);
    reduceHJMapGPU(_frame, lvl);
}


