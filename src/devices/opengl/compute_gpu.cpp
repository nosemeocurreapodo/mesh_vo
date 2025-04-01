#pragma once

#include "gpu.h"

gpu::gpu(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
{
    float xp = float(MAX_WIDTH)/_width;
    float yp = float(MAX_HEIGHT)/_height;

    float out_fx, out_fy, out_cx, out_cy;
    out_fx = _fx*xp; out_fy = _fy*yp; out_cx = _cx*xp; out_cy = _cy*yp;

    max_idepth = 100.0;
    min_idepth = 0.001;

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        float scale = std::pow(2.0f, float(lvl));

        width[lvl] = int(MAX_WIDTH/scale);
        height[lvl] = int(MAX_HEIGHT/scale);

        dx[lvl] = 1.0/width[lvl];
        dy[lvl] = 1.0/height[lvl];

        Eigen::Matrix3f K = Eigen::Matrix3f::Zero();
        K(0,0) = out_fx/scale;
        K(1,1) = out_fy/scale;
        K(2,2) = 1.0f;
        K(0,2) = out_cx/scale;
        K(1,2) = out_cy/scale;

        fx[lvl] = K(0,0);
        fy[lvl] = K(1,1);
        cx[lvl] = K(0,2);
        cy[lvl] = K(1,2);

        Eigen::Matrix3f KInv = K.inverse();

        fxinv[lvl] = KInv(0,0);
        fyinv[lvl] = KInv(1,1);
        cxinv[lvl] = KInv(0,2);
        cyinv[lvl] = KInv(1,2);

        //std::cout << "K" << std::endl;
        //std::cout << K.matrix() << std::endl;
        //std::cout << "Kinv" << std::endl;
        //std::cout << KInv.matrix() << std::endl;

        projMat[lvl] = create_glm_prj_matrix(out_fx/scale, out_fy/scale, out_cx/scale, out_cy/scale, MAX_WIDTH/scale, MAX_HEIGHT/scale, 0.000001, 100000.0);
    }

    opencv2opengl = glm::mat4(1.0f);
    opencv2opengl[1][1] = -1.0;
    opencv2opengl[2][2] = -1.0;


    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_VISIBLE, false);

    // glfw window creation
    // --------------------
    frameWindow = NULL;
    frameWindow = glfwCreateWindow(width[0], height[0], "debug window", NULL, NULL);
    if (frameWindow == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        //return -1;
    }

    glfwMakeContextCurrent(frameWindow);

    glfwSwapInterval(0);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        //return -1;
    }

    //const GLubyte* vendor = glGetString​(GL_VENDOR); // Returns the vendor
    //const GLubyte* renderer = glGetString​(GL_RENDERER); // Returns a hint to the model

    glGenVertexArrays(1, &frame_VAO);
    glGenBuffers(1, &frame_VBO);

    //configure frame vertex
    glBindVertexArray(frame_VAO);

    std::vector<float> frame_vertices;
    frame_vertices.push_back(-1.0);
    frame_vertices.push_back(1.0);
    frame_vertices.push_back(0.0);
    frame_vertices.push_back(1.0);

    frame_vertices.push_back(-1.0);
    frame_vertices.push_back(-1.0);
    frame_vertices.push_back(0.0);
    frame_vertices.push_back(0.0);

    frame_vertices.push_back(1.0);
    frame_vertices.push_back(-1.0);
    frame_vertices.push_back(1.0);
    frame_vertices.push_back(0.0);

    frame_vertices.push_back(-1.0);
    frame_vertices.push_back(1.0);
    frame_vertices.push_back(0.0);
    frame_vertices.push_back(1.0);

    frame_vertices.push_back(1.0);
    frame_vertices.push_back(-1.0);
    frame_vertices.push_back(1.0);
    frame_vertices.push_back(0.0);

    frame_vertices.push_back(1.0);
    frame_vertices.push_back(1.0);
    frame_vertices.push_back(1.0);
    frame_vertices.push_back(1.0);

    glBindBuffer(GL_ARRAY_BUFFER, frame_VBO);
    glBufferData(GL_ARRAY_BUFFER, frame_vertices.size()*sizeof(float), frame_vertices.data(), GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);






    glGenVertexArrays(1, &scene_VAO);
    glGenBuffers(1, &scene_VBO);
    glGenBuffers(1, &scene_EBO);

    glBindVertexArray(scene_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
    glBufferData(GL_ARRAY_BUFFER, scene_vertices.size()*sizeof(float), scene_vertices.data(), GL_STREAM_DRAW);
    // 3. copy our index array in a element buffer for OpenGL to use
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, scene_indices.size()*sizeof(unsigned int), scene_indices.data(), GL_STREAM_DRAW);

    //        // pixel attribute
    //        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    //        glEnableVertexAttribArray(0);
    //        // idepth attribute
    //        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));
    //        glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    /*
    //feedback buffer
    // Create transform feedback buffer
    glGenBuffers(1, &feedbackrbo);
    glBindBuffer(GL_ARRAY_BUFFER, feedbackrbo);
    glBufferData(GL_ARRAY_BUFFER, scene_vertices.size() * sizeof(float), nullptr, GL_STATIC_READ);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, feedbackrbo);
    */

    // framebuffer configuration
    // -------------------------
    //unsigned int framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    // create a color attachment texture
    //unsigned int textureColorbuffer;
    //    glGenTextures(1, &textureColorbuffer);
    //    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    //    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    //    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
    // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, MAX_WIDTH, MAX_HEIGHT); // use a single renderbuffer object for both a depth AND stencil buffer.
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
    // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
    //    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    //        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;

    //    glGenFramebuffers(1, &idepthbuffer);
    //    glBindFramebuffer(GL_FRAMEBUFFER, idepthbuffer);
    //    glGenRenderbuffers(1, &idepthrbo);
    //    glBindRenderbuffer(GL_RENDERBUFFER, idepthrbo);
    //    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, vwidth, vheight); // use a single renderbuffer object for both a depth AND stencil buffer.
    //    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, idepthrbo); // now actually attach it


    keyframeData = frame(height[0], width[0]);
    lastframeData = frame(height[0], width[0]);
    for(int i = 0; i < MAX_FRAMES; i++)
        frameDataStack[i] = frame(height[0], width[0]);

    vertexIdData = data(height[0], width[0], 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    debugData = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    view3DData = data(height[0], width[0], 4, GL_UNSIGNED_BYTE, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

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

    //acc_H_depth = Eigen::MatrixXf::Zero(vwidth*vheight, vwidth*vheight);
    H_depth = Eigen::SparseMatrix<float>(VERTEX_HEIGH*VERTEX_WIDTH, VERTEX_HEIGH*VERTEX_WIDTH);
    J_depth = Eigen::VectorXf::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
    inc_depth = Eigen::VectorXf::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
    count_depth = Eigen::VectorXi::Zero(VERTEX_HEIGH*VERTEX_WIDTH);

    H_joint = Eigen::MatrixXf::Zero(MAX_FRAMES*6+VERTEX_HEIGH*VERTEX_WIDTH,MAX_FRAMES*6+VERTEX_HEIGH*VERTEX_WIDTH);
    J_joint = Eigen::VectorXf::Zero(MAX_FRAMES*6+VERTEX_HEIGH*VERTEX_WIDTH);
    inc_joint = Eigen::VectorXf::Zero(MAX_FRAMES*6+VERTEX_HEIGH*VERTEX_WIDTH);
    count_joint = Eigen::VectorXi::Zero(MAX_FRAMES*6+VERTEX_HEIGH*VERTEX_WIDTH);
}


void setVBO(std::vector<float> scene_vertices, std::vector<float> scene_indices)
{
    glBindVertexArray(scene_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
    //glBufferData(GL_ARRAY_BUFFER, scene_vertices.size()*sizeof(float), scene_vertices.data(), GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

    // 3. copy our index array in a element buffer for OpenGL to use
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene_EBO);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, scene_indices.size()*sizeof(unsigned int), scene_indices.data(), GL_STREAM_DRAW);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, scene_indices.size()*sizeof(unsigned int), scene_indices.data());
}

void setEBO(std::vector<float> scene_indices)
{
    //is the EBO assosiated with the VAO???
    glBindVertexArray(scene_VAO);

    // 3. copy our index array in a element buffer for OpenGL to use
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene_EBO);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, scene_indices.size()*sizeof(unsigned int), scene_indices.data(), GL_STREAM_DRAW);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, scene_indices.size()*sizeof(unsigned int), scene_indices.data());
}

void setFrame()
{
        //save frame in gpu memory
    keyframeData.pose = _pose;

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_keyFrame,keyframeData.image.cpuTexture[lvl],cv::Size(width[lvl], height[lvl]), cv::INTER_LANCZOS4);
        keyframeData.image.cpu_to_gpu(lvl);

        calcIdepthGPU(keyframeData, lvl);
        keyframeData.idepth.gpu_to_cpu(lvl);
    }
}

float mesh_vo::errorGPU_v2(frame *_frame, int lvl)
{
    int srclvl = lvl;
    int dstlvl = srclvl + 5;

    glActiveTexture(GL_TEXTURE0);
    glBindImageTexture( 0, keyframeData.image.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_R8UI);

    glActiveTexture(GL_TEXTURE1);
    glBindImageTexture( 1, _frame->image.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_R8UI);

    glActiveTexture(GL_TEXTURE2);
    glBindImageTexture( 2, keyframeData.idepth.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_R32F );

    glActiveTexture(GL_TEXTURE3);
    glBindImageTexture( 3, _frame->error.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );

    glActiveTexture(GL_TEXTURE4);
    glBindImageTexture( 4, _frame->count.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );

    computeErrorAndReduceCShader.use();

    computeErrorAndReduceCShader.setFloat("fx", fx[srclvl]);
    computeErrorAndReduceCShader.setFloat("fy", fy[srclvl]);
    computeErrorAndReduceCShader.setFloat("cx", cx[srclvl]);
    computeErrorAndReduceCShader.setFloat("cy", cy[srclvl]);
    computeErrorAndReduceCShader.setFloat("fxinv", fxinv[srclvl]);
    computeErrorAndReduceCShader.setFloat("fyinv", fyinv[srclvl]);
    computeErrorAndReduceCShader.setFloat("cxinv", cxinv[srclvl]);
    computeErrorAndReduceCShader.setFloat("cyinv", cyinv[srclvl]);
    computeErrorAndReduceCShader.setFloat("dx", dx[srclvl]);
    computeErrorAndReduceCShader.setFloat("dy", dy[srclvl]);
    computeErrorAndReduceCShader.setMat4("framePose", eigen2glm_mat4((_frame->pose*keyframeData.pose.inverse()).matrix()));

    glDispatchCompute(width[dstlvl], height[dstlvl], 1 );
    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    float error = reduceErrorGPU(_frame, dstlvl);
    //float error = reduceErrorComputeGPU(_frame, dstlvl);
    return error;
}

float mesh_vo::errorStackGPU(int lvl)
{
    float error = 0.0;
    int count = 0;
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == true)
        {
            error += errorGPU(&frameDataStack[i],lvl);
            count++;
        }
    }
    if(count > 0)
        error /= count;
    return error;
}

void mesh_vo::errorTextureGPU(frame *_frame, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    //return calcResidual_CPU(frame, framePose, lvl);

    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _frame->error.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcResidual" << std::endl;

    float error_value = -1.0;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(error_value, error_value, error_value, error_value);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeData.image.gpuTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frame->image.gpuTexture);

    // activate shader
    errorShader.use();

    errorShader.setMat4("framePose", eigen2glm_mat4((_frame->pose*keyframeData.pose.inverse()).matrix()));
    errorShader.setMat4("opencv2opengl", opencv2opengl);
    errorShader.setMat4("projection", projMat[lvl]);
    errorShader.setFloat("fx", fx[lvl]);
    errorShader.setFloat("fy", fy[lvl]);
    errorShader.setFloat("cx", cx[lvl]);
    errorShader.setFloat("cy", cy[lvl]);
    errorShader.setFloat("fxinv", fxinv[lvl]);
    errorShader.setFloat("fyinv", fyinv[lvl]);
    errorShader.setFloat("cxinv", cxinv[lvl]);
    errorShader.setFloat("cyinv", cyinv[lvl]);
    errorShader.setFloat("dx", dx[lvl]);
    errorShader.setFloat("dy", dy[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
}

float mesh_vo::reduceErrorGPU(frame* _frame, int lvl)
{
    _frame->error.gpu_to_cpu(lvl);
    _frame->count.gpu_to_cpu(lvl);

    float error = 0.0;
    int count = 0;
    for(int x = 0; x < width[lvl]; x++)
        for(int y = 0; y < height[lvl]; y++)
        {
            error += _frame->error.cpuTexture[lvl].at<float>(y,x);
            count += _frame->count.cpuTexture[lvl].at<float>(y,x);
        }

    if(count > 0)
    {
        error /= count;
    }
    else
    {
        std::cout << "some problem in calcErrorGPU, maybe images dont overlap" << std::endl;
        error = 1230000000000000000000000000.0f;
    }

    return error;
}

float mesh_vo::reduceErrorComputeGPU(frame* _frame, int lvl)
{
    float result = 0.0;
    int dstlvl = lvl + 3;
    if(dstlvl >= MAX_LEVELS)
    {
        result = reduceErrorGPU(_frame, lvl);
    }
    else
    {
        //tic_toc t;
        //t.tic();

        glActiveTexture(GL_TEXTURE0);
        glBindImageTexture( 0, _frame->error.gpuTexture, lvl, GL_FALSE, 0, GL_READ_ONLY, GL_R32F );

        glActiveTexture(GL_TEXTURE1);
        glBindImageTexture( 1, _frame->error.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );

        glActiveTexture(GL_TEXTURE2);
        glBindImageTexture( 2, _frame->count.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );

        // activate shader
        reduceErrorShader.use();

        glDispatchCompute(width[dstlvl], height[dstlvl], 1 );
        glMemoryBarrier( GL_ALL_BARRIER_BITS );

        //glFinish();
        //std::cout << "reduce compute time " << t.toc() << std::endl;

        result = reduceErrorGPU(_frame, dstlvl);
    }
    return result;
}

void mesh_vo::errorVertexGPU(frame &_frame, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    //return calcResidual_CPU(frame, framePose, lvl);

    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vertexIdData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, _frame.error.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcResidual" << std::endl;

    float error_value = -1.0;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(error_value, error_value, error_value, error_value);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeData.image.gpuTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frame.image.gpuTexture);


    // activate shader
    errorVertexShader.use();

    errorVertexShader.setMat4("framePose", eigen2glm_mat4((_frame.pose*keyframeData.pose.inverse()).matrix()));
    errorVertexShader.setMat4("opencv2opengl", opencv2opengl);
    errorVertexShader.setMat4("projection", projMat[lvl]);
    errorVertexShader.setFloat("fx", fx[lvl]);
    errorVertexShader.setFloat("fy", fy[lvl]);
    errorVertexShader.setFloat("cx", cx[lvl]);
    errorVertexShader.setFloat("cy", cy[lvl]);
    errorVertexShader.setFloat("fxinv", fxinv[lvl]);
    errorVertexShader.setFloat("fyinv", fyinv[lvl]);
    errorVertexShader.setFloat("cxinv", cxinv[lvl]);
    errorVertexShader.setFloat("cyinv", cyinv[lvl]);
    errorVertexShader.setFloat("dx", dx[lvl]);
    errorVertexShader.setFloat("dy", dy[lvl]);


    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);

}

float mesh_vo::reduceErrorVertexGPU(frame _frame, int lvl)
{
    //int new_lvl[MAX_LEVELS] = {5,6,7,8,9};
    //reduceFloat(errorTexture, lvl, new_lvl[lvl]);
    //int new_lvl[MAX_LEVELS] = {0,1,2,3,4};

    //errorData.generateMipmapsGPU(lvl);

    _frame.error.gpu_to_cpu(lvl);
    vertexIdData.gpu_to_cpu(lvl);

    float error = 0.0;
    int count = 0;
    for(int x = 0; x < width[lvl]; x++)
        for(int y = 0; y < height[lvl]; y++)
        {
            float res = _frame.error.cpuTexture[lvl].at<float>(y,x);

            int vertexID[3];
            vertexID[0] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[0]);
            vertexID[1] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[1]);
            vertexID[2] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[2]);

            if(vertexID[0] < 0 || vertexID[1] < 0 || vertexID[2] < 0)
            {
                continue;
            }

            if(res < 0.0)
            {
                continue;
            }

            count++;
            error += res;
            //error_vertex(vertexID[0]) += res;
            //error_vertex(vertexID[1]) += res;
            //error_vertex(vertexID[2]) += res;
            count_depth(vertexID[0])++;
            count_depth(vertexID[1])++;
            count_depth(vertexID[2])++;
        }

    /*
    for(int i = 0; i < vwidth*vheight; i++)
    {
        if(acc_count(i) > 0)
            error_depth(i) = error_depth(i)/acc_count(i);
    }
    */
    if(count > 0)//width[new_lvl]*height[new_lvl]*0.7)
        error /= count;
    else
    {
        std::cout << "some problem in calcErrorGPU, maybe images dont overlap" << std::endl;
        error = 1230000000000000000000000000.0f;
    }

    return error;
}

float mesh_vo::errorGPU(frame *_frame, int lvl)
{
    errorTextureGPU(_frame, lvl);
    //float error =  reduceErrorGPU(lvl);
    float error = reduceErrorComputeGPU(_frame, lvl);
    return error;
}

void mesh_vo::calcIdepthGPU(frame &_frame, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0, width[lvl], height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _frame.idepth.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcIdepth" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // activate shader
    idepthShader.use();
    idepthShader.setMat4("framePose", eigen2glm_mat4((_frame.pose*keyframeData.pose.inverse()).matrix()));
    idepthShader.setMat4("opencv2opengl", opencv2opengl);
    idepthShader.setMat4("projection", projMat[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);
}

void mesh_vo::jacobianPoseMapTextureGPU(frame &_frame, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vertexIdData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, _frame.error.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, _frame.jtra.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, _frame.jrot.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, _frame.jp0.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, _frame.jp1.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, _frame.jp2.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT7, debugData.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0,
                                GL_COLOR_ATTACHMENT1,
                                GL_COLOR_ATTACHMENT2,
                                GL_COLOR_ATTACHMENT3,
                                GL_COLOR_ATTACHMENT4,
                                GL_COLOR_ATTACHMENT5,
                                GL_COLOR_ATTACHMENT6,
                                GL_COLOR_ATTACHMENT7};

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
    jacobianPoseMapShader.use();

    jacobianPoseMapShader.setMat4("framePose", eigen2glm_mat4((_frame.pose*keyframeData.pose.inverse()).matrix()));
    jacobianPoseMapShader.setMat4("opencv2opengl", opencv2opengl);
    jacobianPoseMapShader.setMat4("projection", projMat[lvl]);
    jacobianPoseMapShader.setFloat("fx", fx[lvl]);
    jacobianPoseMapShader.setFloat("fy", fy[lvl]);
    jacobianPoseMapShader.setFloat("cx", cx[lvl]);
    jacobianPoseMapShader.setFloat("cy", cy[lvl]);
    jacobianPoseMapShader.setFloat("fxinv", fxinv[lvl]);
    jacobianPoseMapShader.setFloat("fyinv", fyinv[lvl]);
    jacobianPoseMapShader.setFloat("cxinv", cxinv[lvl]);
    jacobianPoseMapShader.setFloat("cyinv", cyinv[lvl]);
    jacobianPoseMapShader.setFloat("dx", dx[lvl]);
    jacobianPoseMapShader.setFloat("dy", dy[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, 0, 0);

}

void mesh_vo::HJPoseMapStackGPU(int lvl)
{
    H_joint.setZero();
    J_joint.setZero();
    int count = 0;
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == true)
        {
            jacobianPoseMapTextureGPU(frameDataStack[i],lvl);
            reduceHJPoseMapGPU(i,lvl);
            count++;

            showGPU(debugData,lvl);
            cv::waitKey(1000.0);
        }
    }
    if(count > 0)
    {
        H_joint /= count;
        J_joint /= count;
    }
}

void mesh_vo::HJPoseMapGPU(frame &_frame, int lvl)
{
    H_joint.setZero();
    J_joint.setZero();
    jacobianPoseMapTextureGPU(_frame, lvl);
    reduceHJPoseMapGPU(0,lvl);
}

void mesh_vo::reduceHJPoseMapGPU(int frameIndex, int lvl)
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
    frameDataStack[frameIndex].error.gpu_to_cpu(lvl);
    frameDataStack[frameIndex].jtra.gpu_to_cpu(lvl);
    frameDataStack[frameIndex].jrot.gpu_to_cpu(lvl);
    frameDataStack[frameIndex].jp0.gpu_to_cpu(lvl);
    frameDataStack[frameIndex].jp1.gpu_to_cpu(lvl);
    frameDataStack[frameIndex].jp2.gpu_to_cpu(lvl);

    int count = 0;
    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            cv::Vec3f d_I_d_tra = frameDataStack[frameIndex].jtra.cpuTexture[lvl].at<cv::Vec3f>(y,x);
            cv::Vec3f d_I_d_rot = frameDataStack[frameIndex].jrot.cpuTexture[lvl].at<cv::Vec3f>(y,x);
            float residual = frameDataStack[frameIndex].error.cpuTexture[lvl].at<float>(y,x);

            float J_depth[3];
            J_depth[0] = frameDataStack[frameIndex].jp0.cpuTexture[lvl].at<float>(y,x);
            J_depth[1] = frameDataStack[frameIndex].jp1.cpuTexture[lvl].at<float>(y,x);
            J_depth[2] = frameDataStack[frameIndex].jp2.cpuTexture[lvl].at<float>(y,x);

            int vertexID[3];
            vertexID[0] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[0]);
            vertexID[1] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[1]);
            vertexID[2] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[2]);

            if(vertexID[0] < 0 || vertexID[1] < 0 || vertexID[2] < 0)
            {
                continue;
            }

            int paramID[9];
            paramID[0] = frameIndex*6;
            paramID[1] = frameIndex*6+1;
            paramID[2] = frameIndex*6+2;
            paramID[3] = frameIndex*6+3;
            paramID[4] = frameIndex*6+4;
            paramID[5] = frameIndex*6+5;
            paramID[6] = MAX_FRAMES*6+vertexID[0];
            paramID[7] = MAX_FRAMES*6+vertexID[1];
            paramID[8] = MAX_FRAMES*6+vertexID[2];

            Eigen::Matrix<float, 9, 1> J;
            J << d_I_d_tra[0], d_I_d_tra[1], d_I_d_tra[2], d_I_d_rot[0], d_I_d_rot[1], d_I_d_rot[2], J_depth[0], J_depth[1], J_depth[2];

            if(residual!=residual)
            {
                std::cout << "nand in residual " << std::endl;
                continue;
            }
            bool valid = true;
            for(int i = 0; i < 9; i++)
            {
                if(J[i] != J[i])
                {
                    std::cout << "nand in J " << i << " " << x << " " << y << std::endl;
                    valid = false;;
                }
            }
            if(valid == false)
            {
                continue;
            }
            count++;
            for(int i = 0; i < 9; i++)
            {
                J_joint(paramID[i]) += J[i]*residual;
                count_joint(paramID[i]) ++;

                for(int j = i; j < 9; j++)
                {
                    float jj = J[i]*J[j];
                    H_joint(paramID[i],paramID[j]) += jj;
                    H_joint(paramID[j],paramID[i]) += jj;
                }
            }
        }
    if(count > 0)
    {
        H_joint /= count;
        J_joint /= count;
    }
}

void mesh_vo::copyGPU(data &_src, data &_dst, int lvl)
{
    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _dst.gpuTexture, lvl);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcResidual" << std::endl;

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _src.gpuTexture);

    copyShader.use();

    glBindVertexArray(frame_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}


void mesh_vo::showGPU(data &_data, int lvl)
{
    glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[0], height[0]);

    // visualize
    // now bind back to default framebuffer and draw a quad plane with the attached framebuffer color texture
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.
    glDisable(GL_CULL_FACE);
    // clear all relevant buffers
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessery actually, since we won't be able to see behind the quad anyways)
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _data.gpuTexture);

    showTextureShader.use();
    showTextureShader.setInt("lvl", lvl);

    glBindVertexArray(frame_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glfwSwapBuffers(frameWindow);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
}


void mesh_vo::vertexViewCountTextureGPU(frame &_frame, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0, width[lvl], height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vertexIdData.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcIdepth" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(-1.0f, -1.0f, -1.0f, -1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // activate shader
    vertexViewCountShader.use();
    vertexViewCountShader.setMat4("framePose", eigen2glm_mat4((_frame.pose*keyframeData.pose.inverse()).matrix()));
    vertexViewCountShader.setMat4("opencv2opengl", opencv2opengl);
    vertexViewCountShader.setMat4("projection", projMat[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);
}

void mesh_vo::calcDerivativeGPU(frame &_frame, int lvl)
{
    //calculate frame derivative
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0, width[lvl], height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameDerivative, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _frame.der.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! d_C_d_z" << std::endl;


    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! frameDerivative" << std::endl;
    //    auto fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    //    if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
    //        std::cout << "Framebuffer not complete: " << fboStatus << std::endl;


    //    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT)
    //        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! incomplete attachment" << std::endl;
    //    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT)
    //        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! missing attachment" << std::endl;
    //    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER)
    //        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! incomplete draw buffer" << std::endl;
    //    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER)
    //        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! incomplete read buffer" << std::endl;
    //    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_UNSUPPORTED)
    //        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! unsupported" << std::endl;

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    //glClearColor(-1.0f, -1.0f, -1.0f, -1.0f);
    //glClear(GL_COLOR_BUFFER_BIT);
    // bind textures on corresponding texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _frame.image.gpuTexture);
    // activate shader
    frameDerivativeShader.use();
    frameDerivativeShader.setFloat("dx", dx[lvl]);
    frameDerivativeShader.setFloat("dy", dy[lvl]);
    // render frame
    glBindVertexArray(frame_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}



void mesh_vo::view3DTexture(Sophus::SE3f pose, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0, width[lvl], height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view3DData.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcIdepth" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeData.image.gpuTexture);

    // activate shader
    view3DShader.use();
    view3DShader.setMat4("framePose", eigen2glm_mat4((pose*keyframeData.pose.inverse()).matrix()));
    view3DShader.setMat4("opencv2opengl", opencv2opengl);
    view3DShader.setMat4("projection", projMat[lvl]);
    view3DShader.setFloat("fx", fx[lvl]);
    view3DShader.setFloat("fy", fy[lvl]);
    view3DShader.setFloat("cx", cx[lvl]);
    view3DShader.setFloat("cy", cy[lvl]);
    view3DShader.setFloat("dx", dx[lvl]);
    view3DShader.setFloat("dy", dy[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
}

void mesh_vo::vertexViewCountGPU(frame &_frame, int lvl)
{
    /*
    _frame.vertexViewCount.setZero();
    vertexViewCountTextureGPU(_frame, lvl);
    reduceVertexViewCountGPU(_frame,lvl);
    */
}



void mesh_vo::reduceVertexViewCountGPU(frame &_frame,int lvl)
{
    /*
    vertexIdData.gpu_to_cpu(lvl);

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            int vertexID[3];
            vertexID[0] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[0]);
            vertexID[1] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[1]);
            vertexID[2] = int(vertexIdData.cpuTexture[lvl].at<cv::Vec3f>(y,x)[2]);

            if(vertexID[0] < 0 || vertexID[1] < 0 || vertexID[2] < 0)
            {
                //std::cout << "something wrong with the vertexId" << std::endl;
                continue;
            }

            for(int i = 0; i < 3; i++)
                _frame.vertexViewCount(vertexID[i])++;
        }

    for(int i = 0; i < VERTEX_HEIGH*VERTEX_WIDTH; i++)
    {
        if(_frame.vertexViewCount(i) > 0.7*(width[lvl]/VERTEX_WIDTH)*(height[lvl]/VERTEX_HEIGH))
            _frame.vertexViewCount(i) = 1;
        else
            _frame.vertexViewCount(i) = 0;
    }
    */
}

HJPose mesh_vo::HJPoseGPU(frame* _frame, int lvl)
{
    jacobianPoseTextureGPU(_frame, lvl);

    //_frame->error.gpu_to_cpu(lvl);
    //_frame->jtra.gpu_to_cpu(lvl);
    //_frame->jrot.gpu_to_cpu(lvl);
    //HJPose _hjpose = reduceHJPoseGPUPerIndex(_frame, lvl, 0, height[0]);
    //HJPose _hjpose = treadReducer.reduce(std::bind(&mesh_vo::reduceHJPoseGPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    HJPose _hjpose = reduceHJPoseGPU(_frame, lvl);
    //std::cout << "count is " << _hjpose.count << std::endl;
    //std::cout << "error is " << _hjpose.error << std::endl;
    _hjpose.H_pose /= _hjpose.count;
    _hjpose.J_pose /= _hjpose.count;
    _hjpose.error /= _hjpose.count;
    return _hjpose;
}

void mesh_vo::HJPoseGPU_v2(frame *_frame, int lvl)
{
    jacobianPoseTextureGPU_v2(_frame, lvl);
    reduceHJPoseGPU_v2(_frame, lvl);
}

HJPose mesh_vo::HJPoseGPU_v3(frame *_frame, int lvl)
{
    int srclvl = lvl;
    int dstlvl = srclvl + 3;

    glActiveTexture(GL_TEXTURE0);
    glBindImageTexture( 0, keyframeData.image.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_R8UI);
    glActiveTexture(GL_TEXTURE1);
    glBindImageTexture( 1, _frame->image.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_R8UI);
    glActiveTexture(GL_TEXTURE2);
    glBindImageTexture( 2, _frame->der.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glActiveTexture(GL_TEXTURE2);
    glBindImageTexture( 3, keyframeData.idepth.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_R32F );
    glActiveTexture(GL_TEXTURE3);
    glBindImageTexture( 4, _frame->gradient1.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE4);
    glBindImageTexture( 5, _frame->gradient2.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE5);
    glBindImageTexture( 6, _frame->hessian1.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE6);
    glBindImageTexture( 7, _frame->hessian2.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE7);
    glBindImageTexture( 8, _frame->hessian3.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE8);
    glBindImageTexture( 9, _frame->hessian4.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE9);
    glBindImageTexture( 10, _frame->hessian5.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE10);
    glBindImageTexture( 11, _frame->hessian6.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE11);
    glBindImageTexture( 12, _frame->error.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glActiveTexture(GL_TEXTURE12);
    glBindImageTexture( 13, _frame->count.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    computeHJPoseAndReduceCShader.use();

    computeHJPoseAndReduceCShader.setFloat("fx", fx[srclvl]);
    computeHJPoseAndReduceCShader.setFloat("fy", fy[srclvl]);
    computeHJPoseAndReduceCShader.setFloat("cx", cx[srclvl]);
    computeHJPoseAndReduceCShader.setFloat("cy", cy[srclvl]);
    computeHJPoseAndReduceCShader.setFloat("fxinv", fxinv[srclvl]);
    computeHJPoseAndReduceCShader.setFloat("fyinv", fyinv[srclvl]);
    computeHJPoseAndReduceCShader.setFloat("cxinv", cxinv[srclvl]);
    computeHJPoseAndReduceCShader.setFloat("cyinv", cyinv[srclvl]);
    computeHJPoseAndReduceCShader.setFloat("dx", dx[srclvl]);
    computeHJPoseAndReduceCShader.setFloat("dy", dy[srclvl]);
    computeHJPoseAndReduceCShader.setMat4("framePose", eigen2glm_mat4((_frame->pose*keyframeData.pose.inverse()).matrix()));

    glDispatchCompute(width[dstlvl], height[dstlvl], 1 );
    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    HJPose _hjpose = reduceHJPoseGPU_v3(_frame, dstlvl);
    _hjpose.H_pose /= _hjpose.count;
    _hjpose.J_pose /= _hjpose.count;
    _hjpose.error /= _hjpose.count;
    return _hjpose;
}

void mesh_vo::jacobianPoseTextureGPU(frame *_frame, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _frame->error.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, _frame->jrot.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, _frame->jtra.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0,
                                GL_COLOR_ATTACHMENT1,
                                GL_COLOR_ATTACHMENT2};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! JPose" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeData.image.gpuTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frame->image.gpuTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _frame->der.gpuTexture);

    // activate shader
    jacobianPoseShader.use();

    jacobianPoseShader.setMat4("framePose", eigen2glm_mat4((_frame->pose*keyframeData.pose.inverse()).matrix()));
    jacobianPoseShader.setMat4("opencv2opengl", opencv2opengl);
    jacobianPoseShader.setMat4("projection", projMat[lvl]);
    jacobianPoseShader.setFloat("fx", fx[lvl]);
    jacobianPoseShader.setFloat("fy", fy[lvl]);
    jacobianPoseShader.setFloat("cx", cx[lvl]);
    jacobianPoseShader.setFloat("cy", cy[lvl]);
    jacobianPoseShader.setFloat("fxinv", fxinv[lvl]);
    jacobianPoseShader.setFloat("fyinv", fyinv[lvl]);
    jacobianPoseShader.setFloat("cxinv", cxinv[lvl]);
    jacobianPoseShader.setFloat("cyinv", cyinv[lvl]);
    jacobianPoseShader.setFloat("dx", dx[lvl]);
    jacobianPoseShader.setFloat("dy", dy[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
}

HJPose mesh_vo::reduceHJPoseGPU(frame *_frame, int lvl)
{
    HJPose _hjpose;

    int src_lvl = lvl;
    int dst_lvl = lvl + 4;

    glActiveTexture(GL_TEXTURE0);
    glBindImageTexture( 0, _frame->jtra.gpuTexture, src_lvl, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE1);
    glBindImageTexture( 1, _frame->jrot.gpuTexture, src_lvl, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE2);
    glBindImageTexture( 2, _frame->error.gpuTexture, src_lvl, GL_FALSE, 0, GL_READ_ONLY, GL_R32F );
    glActiveTexture(GL_TEXTURE3);
    glBindImageTexture( 3, _frame->gradient1.gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE4);
    glBindImageTexture( 4, _frame->gradient2.gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE5);
    glBindImageTexture( 5, _frame->hessian1.gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE6);
    glBindImageTexture( 6, _frame->hessian2.gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE7);
    glBindImageTexture( 7, _frame->hessian3.gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE8);
    glBindImageTexture( 8, _frame->hessian4.gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE9);
    glBindImageTexture( 9, _frame->hessian5.gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE10);
    glBindImageTexture( 10, _frame->hessian6.gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );
    glActiveTexture(GL_TEXTURE11);
    glBindImageTexture( 11, _frame->error.gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glActiveTexture(GL_TEXTURE12);
    glBindImageTexture( 12, _frame->count.gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    // activate shader
    reduceHJPoseShader.use();
    glDispatchCompute(width[dst_lvl], height[dst_lvl], 1 );
    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    //_hjpose = reduceHJPoseGPU_v3(_frame, dst_lvl);
    //return _hjpose;

    _frame->gradient1.gpu_to_cpu(dst_lvl);
    _frame->gradient2.gpu_to_cpu(dst_lvl);
    _frame->hessian1.gpu_to_cpu(dst_lvl);
    _frame->hessian2.gpu_to_cpu(dst_lvl);
    _frame->hessian3.gpu_to_cpu(dst_lvl);
    _frame->hessian4.gpu_to_cpu(dst_lvl);
    _frame->hessian5.gpu_to_cpu(dst_lvl);
    _frame->hessian6.gpu_to_cpu(dst_lvl);
    _frame->error.gpu_to_cpu(dst_lvl);
    _frame->count.gpu_to_cpu(dst_lvl);

    for(int y = 0; y < height[dst_lvl]; y++)
        for(int x = 0; x < width[dst_lvl]; x++)
        {
            cv::Vec4f gradient1 = _frame->gradient1.cpuTexture[dst_lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f gradient2 = _frame->gradient2.cpuTexture[dst_lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian1  = _frame->hessian1.cpuTexture[dst_lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian2  = _frame->hessian2.cpuTexture[dst_lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian3  = _frame->hessian3.cpuTexture[dst_lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian4  = _frame->hessian4.cpuTexture[dst_lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian5  = _frame->hessian5.cpuTexture[dst_lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian6  = _frame->hessian6.cpuTexture[dst_lvl].at<cv::Vec4f>(y,x);
            float error = _frame->error.cpuTexture[dst_lvl].at<float>(y,x);
            float count = _frame->count.cpuTexture[dst_lvl].at<float>(y,x);

            _hjpose.J_pose(0) += gradient1[0];
            _hjpose.J_pose(1) += gradient1[1];
            _hjpose.J_pose(2) += gradient1[2];
            _hjpose.J_pose(3) += gradient2[0];
            _hjpose.J_pose(4) += gradient2[1];
            _hjpose.J_pose(5) += gradient2[2];

            _hjpose.H_pose(0,0) += hessian1[0];
            _hjpose.H_pose(1,0) += hessian1[1];
            _hjpose.H_pose(0,1) += hessian1[1];
            _hjpose.H_pose(2,0) += hessian1[2];
            _hjpose.H_pose(0,2) += hessian1[2];
            _hjpose.H_pose(3,0) += hessian1[3];
            _hjpose.H_pose(0,3) += hessian1[3];
            _hjpose.H_pose(4,0) += hessian2[0];
            _hjpose.H_pose(0,4) += hessian2[0];
            _hjpose.H_pose(5,0) += hessian2[1];
            _hjpose.H_pose(0,5) += hessian2[1];

            _hjpose.H_pose(1,1) += hessian2[2];
            _hjpose.H_pose(2,1) += hessian2[3];
            _hjpose.H_pose(1,2) += hessian2[3];
            _hjpose.H_pose(3,1) += hessian3[0];
            _hjpose.H_pose(1,3) += hessian3[0];
            _hjpose.H_pose(4,1) += hessian3[1];
            _hjpose.H_pose(1,4) += hessian3[1];
            _hjpose.H_pose(5,1) += hessian3[2];
            _hjpose.H_pose(1,5) += hessian3[2];

            _hjpose.H_pose(2,2) += hessian3[3];
            _hjpose.H_pose(3,2) += hessian4[0];
            _hjpose.H_pose(2,3) += hessian4[0];
            _hjpose.H_pose(4,2) += hessian4[1];
            _hjpose.H_pose(2,4) += hessian4[1];
            _hjpose.H_pose(5,2) += hessian4[2];
            _hjpose.H_pose(2,5) += hessian4[2];

            _hjpose.H_pose(3,3) += hessian4[3];
            _hjpose.H_pose(4,3) += hessian5[0];
            _hjpose.H_pose(3,4) += hessian5[0];
            _hjpose.H_pose(5,3) += hessian5[1];
            _hjpose.H_pose(3,5) += hessian5[1];

            _hjpose.H_pose(4,4) += hessian5[2];
            _hjpose.H_pose(5,4) += hessian5[3];
            _hjpose.H_pose(4,5) += hessian5[3];

            _hjpose.H_pose(5,5) += hessian6[0];

            _hjpose.count += count;
            _hjpose.error += error;
        }

    return _hjpose;
}

HJPose mesh_vo::reduceHJPoseGPU_v3(frame *_frame, int lvl)
{
    HJPose _hjpose;

    cv::Vec4f gradient1 = reduceVec4(&(_frame->gradient1), lvl);
    cv::Vec4f gradient2 = reduceVec4(&(_frame->gradient2), lvl);
    cv::Vec4f hessian1 = reduceVec4(&(_frame->hessian1), lvl);
    cv::Vec4f hessian2 = reduceVec4(&(_frame->hessian2), lvl);
    cv::Vec4f hessian3 = reduceVec4(&(_frame->hessian3), lvl);
    cv::Vec4f hessian4 = reduceVec4(&(_frame->hessian4), lvl);
    cv::Vec4f hessian5 = reduceVec4(&(_frame->hessian5), lvl);
    cv::Vec4f hessian6 = reduceVec4(&(_frame->hessian6), lvl);
    float error = reduceFloat(&(_frame->error), lvl);
    float count = reduceFloat(&(_frame->count), lvl);

    _hjpose.J_pose(0) += gradient1[0];
    _hjpose.J_pose(1) += gradient1[1];
    _hjpose.J_pose(2) += gradient1[2];
    _hjpose.J_pose(3) += gradient2[0];
    _hjpose.J_pose(4) += gradient2[1];
    _hjpose.J_pose(5) += gradient2[2];

    _hjpose.H_pose(0,0) += hessian1[0];
    _hjpose.H_pose(1,0) += hessian1[1];
    _hjpose.H_pose(0,1) += hessian1[1];
    _hjpose.H_pose(2,0) += hessian1[2];
    _hjpose.H_pose(0,2) += hessian1[2];
    _hjpose.H_pose(3,0) += hessian1[3];
    _hjpose.H_pose(0,3) += hessian1[3];
    _hjpose.H_pose(4,0) += hessian2[0];
    _hjpose.H_pose(0,4) += hessian2[0];
    _hjpose.H_pose(5,0) += hessian2[1];
    _hjpose.H_pose(0,5) += hessian2[1];

    _hjpose.H_pose(1,1) += hessian2[2];
    _hjpose.H_pose(2,1) += hessian2[3];
    _hjpose.H_pose(1,2) += hessian2[3];
    _hjpose.H_pose(3,1) += hessian3[0];
    _hjpose.H_pose(1,3) += hessian3[0];
    _hjpose.H_pose(4,1) += hessian3[1];
    _hjpose.H_pose(1,4) += hessian3[1];
    _hjpose.H_pose(5,1) += hessian3[2];
    _hjpose.H_pose(1,5) += hessian3[2];

    _hjpose.H_pose(2,2) += hessian3[3];
    _hjpose.H_pose(3,2) += hessian4[0];
    _hjpose.H_pose(2,3) += hessian4[0];
    _hjpose.H_pose(4,2) += hessian4[1];
    _hjpose.H_pose(2,4) += hessian4[1];
    _hjpose.H_pose(5,2) += hessian4[2];
    _hjpose.H_pose(2,5) += hessian4[2];

    _hjpose.H_pose(3,3) += hessian4[3];
    _hjpose.H_pose(4,3) += hessian5[0];
    _hjpose.H_pose(3,4) += hessian5[0];
    _hjpose.H_pose(5,3) += hessian5[1];
    _hjpose.H_pose(3,5) += hessian5[1];

    _hjpose.H_pose(4,4) += hessian5[2];
    _hjpose.H_pose(5,4) += hessian5[3];
    _hjpose.H_pose(4,5) += hessian5[3];

    _hjpose.H_pose(5,5) += hessian6[0];

    _hjpose.error = error;
    _hjpose.count = count;

    return _hjpose;
}

float mesh_vo::reduceFloat(data* _data, int lvl)
{
    float result = 0.0;

    int src_lvl = lvl;
    int dst_lvl = lvl + 5;

    glActiveTexture(GL_TEXTURE0);
    glBindImageTexture( 0, _data->gpuTexture, src_lvl, GL_FALSE, 0, GL_READ_ONLY, GL_R32F );

    glActiveTexture(GL_TEXTURE1);
    glBindImageTexture( 1, _data->gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );

    // activate shader
    reduceRGBAShader.use();
    glDispatchCompute(width[dst_lvl], height[dst_lvl], 1 );
    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    _data->gpu_to_cpu(dst_lvl);

    for(int y = 0; y < height[dst_lvl]; y++)
        for(int x = 0; x < width[dst_lvl]; x++)
        {
            result += _data->cpuTexture[dst_lvl].at<float>(y,x);
        }

    return result;
}

cv::Vec4f mesh_vo::reduceVec4(data* _data, int lvl)
{
    cv::Vec4f result;
    result[0] = 0.0;
    result[1] = 0.0;
    result[2] = 0.0;
    result[3] = 0.0;

    int src_lvl = lvl;
    int dst_lvl = lvl + 5;

    glActiveTexture(GL_TEXTURE0);
    glBindImageTexture( 0, _data->gpuTexture, src_lvl, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F );

    glActiveTexture(GL_TEXTURE1);
    glBindImageTexture( 1, _data->gpuTexture, dst_lvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F );

    // activate shader
    reduceRGBAShader.use();
    glDispatchCompute(width[dst_lvl], height[dst_lvl], 1 );
    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    _data->gpu_to_cpu(dst_lvl);

    for(int y = 0; y < height[dst_lvl]; y++)
        for(int x = 0; x < width[dst_lvl]; x++)
        {
            result += _data->cpuTexture[dst_lvl].at<cv::Vec4f>(y,x);
        }

    return result;
}

HJPose mesh_vo::reduceHJPoseGPUPerIndex(frame* _frame, int lvl, int ymin, int ymax)
{
    HJPose _hjpose;

    for(int y = ymin; y < ymax; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            cv::Vec4f d_I_d_tra = _frame->jtra.cpuTexture[lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f d_I_d_rot = _frame->jrot.cpuTexture[lvl].at<cv::Vec4f>(y,x);
            float residual = _frame->error.cpuTexture[lvl].at<float>(y,x);

            if(residual == 0.0)
                continue;

            Eigen::Matrix<float, 6, 1> J;
            J << d_I_d_tra[0], d_I_d_tra[1], d_I_d_tra[2], d_I_d_rot[0], d_I_d_rot[1], d_I_d_rot[2];

            _hjpose.error += residual*residual;
            _hjpose.count++;
            for(int i = 0; i < 6; i++)
            {
                _hjpose.J_pose(i) += J[i]*residual;
                for(int j = i; j < 6; j++)
                {
                    float jj = J[i]*J[j];
                    _hjpose.H_pose(i,j) += jj;
                    _hjpose.H_pose(j,i) += jj;
                }
            }
        }

    return _hjpose;
}

void mesh_vo::jacobianPoseTextureGPU_v2(frame *_frame, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _frame->gradient1.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, _frame->gradient2.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, _frame->hessian1.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, _frame->hessian2.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, _frame->hessian3.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, _frame->hessian4.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, _frame->hessian5.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT7, _frame->hessian6.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0,
                                GL_COLOR_ATTACHMENT1,
                                GL_COLOR_ATTACHMENT2,
                                GL_COLOR_ATTACHMENT3,
                                GL_COLOR_ATTACHMENT4,
                                GL_COLOR_ATTACHMENT5,
                                GL_COLOR_ATTACHMENT6,
                                GL_COLOR_ATTACHMENT7};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! JPose" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeData.image.gpuTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frame->image.gpuTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _frame->der.gpuTexture);

    // activate shader
    jacobianPoseShader_v2.use();

    jacobianPoseShader_v2.setMat4("framePose", eigen2glm_mat4((_frame->pose*keyframeData.pose.inverse()).matrix()));
    jacobianPoseShader_v2.setMat4("opencv2opengl", opencv2opengl);
    jacobianPoseShader_v2.setMat4("projection", projMat[lvl]);
    jacobianPoseShader_v2.setFloat("fx", fx[lvl]);
    jacobianPoseShader_v2.setFloat("fy", fy[lvl]);
    jacobianPoseShader_v2.setFloat("cx", cx[lvl]);
    jacobianPoseShader_v2.setFloat("cy", cy[lvl]);
    jacobianPoseShader_v2.setFloat("fxinv", fxinv[lvl]);
    jacobianPoseShader_v2.setFloat("fyinv", fyinv[lvl]);
    jacobianPoseShader_v2.setFloat("cxinv", cxinv[lvl]);
    jacobianPoseShader_v2.setFloat("cyinv", cyinv[lvl]);
    jacobianPoseShader_v2.setFloat("dx", dx[lvl]);
    jacobianPoseShader_v2.setFloat("dy", dy[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);


    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT7, 0, 0);

}

HJPose mesh_vo::reduceHJPoseGPU_v2(frame *_frame,int lvl)
{
    HJPose _hjpose;

    _frame->gradient1.gpu_to_cpu(lvl);
    _frame->gradient2.gpu_to_cpu(lvl);
    _frame->hessian1.gpu_to_cpu(lvl);
    _frame->hessian2.gpu_to_cpu(lvl);
    _frame->hessian3.gpu_to_cpu(lvl);
    _frame->hessian4.gpu_to_cpu(lvl);
    _frame->hessian5.gpu_to_cpu(lvl);
    _frame->hessian6.gpu_to_cpu(lvl);
    _frame->error.gpu_to_cpu(lvl);
    _frame->count.gpu_to_cpu(lvl);

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            cv::Vec4f gradient1 = _frame->gradient1.cpuTexture[lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f gradient2 = _frame->gradient2.cpuTexture[lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian1  = _frame->hessian1.cpuTexture[lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian2  = _frame->hessian2.cpuTexture[lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian3  = _frame->hessian3.cpuTexture[lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian4  = _frame->hessian4.cpuTexture[lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian5  = _frame->hessian5.cpuTexture[lvl].at<cv::Vec4f>(y,x);
            cv::Vec4f hessian6  = _frame->hessian6.cpuTexture[lvl].at<cv::Vec4f>(y,x);
            float error = _frame->error.cpuTexture[lvl].at<float>(y,x);
            float count = _frame->count.cpuTexture[lvl].at<float>(y,x);

            _hjpose.J_pose(0) += gradient1[0];
            _hjpose.J_pose(1) += gradient1[1];
            _hjpose.J_pose(2) += gradient1[2];
            _hjpose.J_pose(3) += gradient2[0];
            _hjpose.J_pose(4) += gradient2[1];
            _hjpose.J_pose(5) += gradient2[2];

            _hjpose.H_pose(0,0) += hessian1[0];
            _hjpose.H_pose(1,0) += hessian1[1];
            _hjpose.H_pose(0,1) += hessian1[1];
            _hjpose.H_pose(2,0) += hessian1[2];
            _hjpose.H_pose(0,2) += hessian1[2];
            _hjpose.H_pose(3,0) += hessian1[3];
            _hjpose.H_pose(0,3) += hessian1[3];
            _hjpose.H_pose(4,0) += hessian2[0];
            _hjpose.H_pose(0,4) += hessian2[0];
            _hjpose.H_pose(5,0) += hessian2[1];
            _hjpose.H_pose(0,5) += hessian2[1];

            _hjpose.H_pose(1,1) += hessian2[2];
            _hjpose.H_pose(2,1) += hessian2[3];
            _hjpose.H_pose(1,2) += hessian2[3];
            _hjpose.H_pose(3,1) += hessian3[0];
            _hjpose.H_pose(1,3) += hessian3[0];
            _hjpose.H_pose(4,1) += hessian3[1];
            _hjpose.H_pose(1,4) += hessian3[1];
            _hjpose.H_pose(5,1) += hessian3[2];
            _hjpose.H_pose(1,5) += hessian3[2];

            _hjpose.H_pose(2,2) += hessian3[3];
            _hjpose.H_pose(3,2) += hessian4[0];
            _hjpose.H_pose(2,3) += hessian4[0];
            _hjpose.H_pose(4,2) += hessian4[1];
            _hjpose.H_pose(2,4) += hessian4[1];
            _hjpose.H_pose(5,2) += hessian4[2];
            _hjpose.H_pose(2,5) += hessian4[2];

            _hjpose.H_pose(3,3) += hessian4[3];
            _hjpose.H_pose(4,3) += hessian5[0];
            _hjpose.H_pose(3,4) += hessian5[0];
            _hjpose.H_pose(5,3) += hessian5[1];
            _hjpose.H_pose(3,5) += hessian5[1];

            _hjpose.H_pose(4,4) += hessian5[2];
            _hjpose.H_pose(5,4) += hessian5[3];
            _hjpose.H_pose(4,5) += hessian5[3];

            _hjpose.H_pose(5,5) += hessian6[0];

            _hjpose.error += error;
            _hjpose.count += count;

        }

    return _hjpose;
}