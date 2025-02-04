#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "common/camera.h"

//#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/compute.h>
//#include <learnopengl/feedback_shader.h>
//#include <learnopengl/camera.h>

glm::mat4 create_glm_prj_matrix(float fx, float fy, float cx, float cy, float w, float h, float znear, float zfar)
{
    glm::mat4 projmat = glm::mat4(0.0f);

    projmat[0][0] = 2.0f*fx/w;
    projmat[1][1] = 2.0f*fy/h;
    projmat[2][0] = 1.0f - 2.0f*cx/w;
    projmat[2][1] = -1.0f + 2.0f*cy/h;
    projmat[2][2] = -(zfar + znear) / (zfar - znear);
    projmat[2][3] = -1.0f;
    projmat[3][2] = -2.0f * zfar * znear / (zfar - znear);

    return projmat;
}


class visualizer
{
public:
    visualizer(camera cam)
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


