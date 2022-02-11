#include "mesh_vo.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>
#include <Eigen/IterativeLinearSolvers>

#include <map>

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

glm::mat4 eigen2glm_mat4(Eigen::Matrix4f eigen_m)
{
    glm::mat4 glm_m;
    glm_m[0][0] = float(eigen_m(0,0));
    glm_m[1][0] = float(eigen_m(0,1));
    glm_m[2][0] = float(eigen_m(0,2));
    glm_m[3][0] = float(eigen_m(0,3));
    glm_m[0][1] = float(eigen_m(1,0));
    glm_m[1][1] = float(eigen_m(1,1));
    glm_m[2][1] = float(eigen_m(1,2));
    glm_m[3][1] = float(eigen_m(1,3));
    glm_m[0][2] = float(eigen_m(2,0));
    glm_m[1][2] = float(eigen_m(2,1));
    glm_m[2][2] = float(eigen_m(2,2));
    glm_m[3][2] = float(eigen_m(2,3));
    glm_m[0][3] = float(eigen_m(3,0));
    glm_m[1][3] = float(eigen_m(3,1));
    glm_m[2][3] = float(eigen_m(3,2));
    glm_m[3][3] = float(eigen_m(3,3));

    return glm_m;
}

glm::mat3 eigen2glm_mat3(Eigen::Matrix3f eigen_m)
{
    glm::mat3 glm_m;
    glm_m[0][0] = eigen_m(0,0);
    glm_m[1][0] = eigen_m(0,1);
    glm_m[2][0] = eigen_m(0,2);
    glm_m[0][1] = eigen_m(1,0);
    glm_m[1][1] = eigen_m(1,1);
    glm_m[2][1] = eigen_m(1,2);
    glm_m[0][2] = eigen_m(2,0);
    glm_m[1][2] = eigen_m(2,1);
    glm_m[2][2] = eigen_m(2,2);

    return glm_m;
}


mesh_vo::mesh_vo(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
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
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
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


    //generate buffers
    vwidth = 48;
    vheight = 48;

    //prealocate
    for(int y=0;y<vheight;y++)
    {
        for(int x=0;x<vwidth;x++)
        {
            scene_vertices.push_back(0.0);
            scene_vertices.push_back(0.0);
            scene_vertices.push_back(0.0);

            scene_vertices_updated.push_back(0.0);
            scene_vertices_updated.push_back(0.0);
            scene_vertices_updated.push_back(0.0);

            if(x>0 && y>0)
            {
                scene_indices.push_back(0);
                scene_indices.push_back(0);
                scene_indices.push_back(0);

                scene_indices.push_back(0);
                scene_indices.push_back(0);
                scene_indices.push_back(0);
            }
        }
    }

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
    frameData = frame(height[0], width[0]);
    for(int i = 0; i < MAX_FRAMES; i++)
        frameDataStack[i] = frame(height[0], width[0]);

    errorData = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    residualData = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    traData = data(height[0], width[0], 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    rotData = data(height[0], width[0], 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    Jpose1Data = data(height[0], width[0], 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    Jpose2Data = data(height[0], width[0], 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    Jpose3Data = data(height[0], width[0], 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    Jpose4Data = data(height[0], width[0], 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    Jpose5Data = data(height[0], width[0], 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    Jpose6Data = data(height[0], width[0], 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    Jpose7Data = data(height[0], width[0], 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    vertexIdData = data(height[0], width[0], 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    primitiveIdData = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    d_I_d_p0Data = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    d_I_d_p1Data = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    d_I_d_p2Data = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    errorShader.init("error.vs", "error.fs");
    errorShader.use();
    errorShader.setInt("keyframe", 0);
    errorShader.setInt("frame", 1);

    reduceFloatShader.init("reduceFloat.vs", "reduceFloat.fs");
    reduceFloatShader.use();
    reduceFloatShader.setInt("inTexture", 0);

    reduceVec4Shader.init("reduceVec4.vs", "reduceVec4.fs");
    reduceVec4Shader.use();
    reduceVec4Shader.setInt("inTexture", 0);

    calcJShader.init("JPose.vs", "JPose.fs");
    calcJShader.use();
    calcJShader.setInt("keyframe", 0);
    calcJShader.setInt("frame", 1);
    calcJShader.setInt("frameDer", 2);

    calcHJShader.init("HJPose.vs", "HJPose.fs");
    calcHJShader.use();
    calcHJShader.setInt("residualTexture", 0);
    calcHJShader.setInt("traTexture", 1);
    calcHJShader.setInt("rotTexture", 2);

    calcHJShader2.init("HJPose2.vs", "HJPose2.fs");
    calcHJShader2.use();
    calcHJShader2.setInt("keyframe", 0);
    calcHJShader2.setInt("frame", 1);
    calcHJShader2.setInt("frameDer", 2);

    calcHJMapShader.init("JMap.vs", "JMap.gs", "JMap.fs");
    calcHJMapShader.use();
    calcHJMapShader.setInt("keyframe", 0);
    calcHJMapShader.setInt("frame", 1);
    calcHJMapShader.setInt("frameDer", 2);

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


    //debugShader.init("debug.vs", "debug.gs", "debug.fs");
    //debugShader.use();
    //debugShader.setInt("keyframe", 0);
    //debugShader.setInt("frame", 1);}}

    //acc_H_depth = Eigen::MatrixXf::Zero(vwidth*vheight, vwidth*vheight);
    acc_H_depth = Eigen::SparseMatrix<float>(vwidth*vheight, vwidth*vheight);
    acc_J_depth = Eigen::VectorXf::Zero(vwidth*vheight);
    inc_depth = Eigen::VectorXf(vwidth*vheight);
    acc_count = Eigen::VectorXi(vwidth*vheight);

    occupancy = 1.0;
    lastFrameAdded = -1;

    //for profiling
    calcPoseTime = 0.0;
}

void mesh_vo::initWithRandomIdepth(cv::Mat _keyFrame, Sophus::SE3f _pose)
{
    //const float* maxGradients = new_frame->maxGradients();

    scene_vertices.clear();
    scene_indices.clear();

    for(int y=0;y<vheight;y++) //los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for(int x=0;x<vwidth;x++)
        {
            float idepth = 0.1 + (1.0-0.1) * float(y)/vheight;
            //float idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
            //float idepth = 0.5;

            float xi = (float(x)/float(vwidth-1))*width[0];
            float yi = (float(y)/float(vheight-1))*height[0];
            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
            Eigen::Vector3f r = Eigen::Vector3f(fxinv[0]*u(0) + cxinv[0], fyinv[0]*u(1) + cyinv[0], 1.0);
            Eigen::Vector3f p = r/idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(idepth);

            //            scene_vertices.push_back(xi);
            //            scene_vertices.push_back(yi);
            //            scene_vertices.push_back(idepth);

            if(x>0 && y>0)
            {
                scene_indices.push_back(x-1 +     y*(vwidth));
                scene_indices.push_back(x   + (y-1)*(vwidth));
                scene_indices.push_back(x-1 + (y-1)*(vwidth));

                scene_indices.push_back(x   +     y*(vwidth));
                scene_indices.push_back(x   + (y-1)*(vwidth));
                scene_indices.push_back(x-1 +     y*(vwidth));
            }
        }
    }

    glBindVertexArray(scene_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
    //glBufferData(GL_ARRAY_BUFFER, scene_vertices.size()*sizeof(float), scene_vertices.data(), GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

    // 3. copy our index array in a element buffer for OpenGL to use
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene_EBO);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, scene_indices.size()*sizeof(unsigned int), scene_indices.data(), GL_STREAM_DRAW);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, scene_indices.size()*sizeof(unsigned int), scene_indices.data());

    //save frame in gpu memory
    keyframeData.pose = _pose;
    calcIdepthGPU(keyframeData,0);
    keyframeData.idepth.gpu_to_cpu(0);
    keyframeData.idepth.generateMipmapsCPU(0);

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_keyFrame,keyframeData.image.cpuTexture[lvl],cv::Size(width[lvl], height[lvl]), cv::INTER_LANCZOS4);
        keyframeData.image.cpu_to_gpu(lvl);

        calcDerivativeCPU(keyframeData, lvl);
        keyframeData.der.cpu_to_gpu(lvl);
    }
}

void mesh_vo::initWithIdepth(cv::Mat _keyFrame, cv::Mat _idepth, Sophus::SE3f _pose)
{    
    scene_vertices.clear();
    scene_indices.clear();

    for(int y=0;y<vheight;y++) //los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for(int x=0;x<vwidth;x++)
        {
            float xi = (float(x)/float(vwidth-1))*width[0];
            float yi = (float(y)/float(vheight-1))*height[0];

            float idepth = _idepth.at<float>(yi,xi);
            /*
            if(idepth <= min_idepth)
                idepth = min_idepth;
            if(idepth > max_idepth)
                idepth = max_idepth;
                */
            if(idepth != idepth || idepth < 0.1 || idepth > 1.0)
                idepth = 0.1 + (1.0-0.1) * float(y)/vheight;

            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
            Eigen::Vector3f r = Eigen::Vector3f(fxinv[0]*u(0) + cxinv[0], fyinv[0]*u(1) + cyinv[0], 1.0);
            Eigen::Vector3f p = r/idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(idepth);

            //scene_vertices.push_back(xi);
            //scene_vertices.push_back(yi);
            //scene_vertices.push_back(idepth);

            if(x>0 && y>0)
            {
                scene_indices.push_back(x-1 +     y*(vwidth));
                scene_indices.push_back(x   + (y-1)*(vwidth));
                scene_indices.push_back(x-1 + (y-1)*(vwidth));

                scene_indices.push_back(x   +     y*(vwidth));
                scene_indices.push_back(x   + (y-1)*(vwidth));
                scene_indices.push_back(x-1 +     y*(vwidth));
            }
        }
    }

    glBindVertexArray(scene_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
    //glBufferData(GL_ARRAY_BUFFER, scene_vertices.size()*sizeof(float), scene_vertices.data(), GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

    // 3. copy our index array in a element buffer for OpenGL to use
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene_EBO);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, scene_indices.size()*sizeof(unsigned int), scene_indices.data(), GL_STREAM_DRAW);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, scene_indices.size()*sizeof(unsigned int), scene_indices.data());

    //save frame in gpu memory
    keyframeData.pose = _pose;
    calcIdepthGPU(keyframeData,0);

    keyframeData.idepth.gpu_to_cpu(0);
    keyframeData.idepth.generateMipmapsCPU(0);


    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_keyFrame,keyframeData.image.cpuTexture[lvl],cv::Size(width[lvl], height[lvl]), cv::INTER_LANCZOS4);
        keyframeData.image.cpu_to_gpu(lvl);

        calcDerivativeCPU(keyframeData, lvl);
        keyframeData.der.cpu_to_gpu(lvl);
    }
}

void mesh_vo::changeKeyframe(frame newkeyFrame)
{
    calcIdepthGPU(newkeyFrame, 0);
    newkeyFrame.idepth.gpu_to_cpu(0);

    while(true)
    {
        cv::Mat oldImage = newkeyFrame.idepth.cpuTexture[0];

        if(cv::countNonZero(oldImage) == width[0]*height[0])
            break;

        cv::Mat newImage;
        oldImage.copyTo(newImage);

        for(int y = 0; y < height[0]; y++)
            for(int x = 0; x < width[0]; x++)
            {
                float idepth = oldImage.at<float>(y,x);
                if(idepth <= 0.0)
                {
                    float newIdepth = 0.0;
                    int count = 0;
                    for(int yi = -1; yi < 2; yi++)
                        for(int xi = -1; xi < 2; xi++)
                        {
                            if(oldImage.at<float>(y+yi,x+xi) > 0.0)
                            {
                                newIdepth += oldImage.at<float>(y+yi,x+xi);
                                count ++;
                            }
                        }
                    if(count > 0)
                      newImage.at<float>(y,x) = newIdepth/count;
                }
            }
        newImage.copyTo(newkeyFrame.idepth.cpuTexture[0]);
    }

    scene_vertices.clear();
    scene_indices.clear();

    for(int y=0;y<vheight;y++) //los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for(int x=0;x<vwidth;x++)
        {
            float xi = (float(x)/float(vwidth-1))*width[0];
            float yi = (float(y)/float(vheight-1))*height[0];

            float idepth = newkeyFrame.idepth.cpuTexture[0].at<float>(yi,xi);

            if(idepth < min_idepth || idepth > max_idepth)
                idepth = 0.1 + (1.0-0.1) * float(y)/vheight;

            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
            Eigen::Vector3f r = Eigen::Vector3f(fxinv[0]*u(0) + cxinv[0], fyinv[0]*u(1) + cyinv[0], 1.0);
            Eigen::Vector3f p = r/idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(idepth);

            //scene_vertices.push_back(xi);
            //scene_vertices.push_back(yi);
            //scene_vertices.push_back(idepth);

            if(x>0 && y>0)
            {
                scene_indices.push_back(x-1 +     y*(vwidth));
                scene_indices.push_back(x   + (y-1)*(vwidth));
                scene_indices.push_back(x-1 + (y-1)*(vwidth));

                scene_indices.push_back(x   +     y*(vwidth));
                scene_indices.push_back(x   + (y-1)*(vwidth));
                scene_indices.push_back(x-1 +     y*(vwidth));
            }
        }
    }

    glBindVertexArray(scene_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
    //glBufferData(GL_ARRAY_BUFFER, scene_vertices.size()*sizeof(float), scene_vertices.data(), GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

    // 3. copy our index array in a element buffer for OpenGL to use
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene_EBO);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, scene_indices.size()*sizeof(unsigned int), scene_indices.data(), GL_STREAM_DRAW);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, scene_indices.size()*sizeof(unsigned int), scene_indices.data());

    //save frame in gpu memory
    keyframeData.pose = newkeyFrame.pose;
    calcIdepthGPU(keyframeData,0);
    keyframeData.idepth.gpu_to_cpu(0);
    keyframeData.idepth.generateMipmapsCPU(0);

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        copyCPU(newkeyFrame.image, keyframeData.image,lvl);
        copyGPU(newkeyFrame.image, keyframeData.image, lvl);

        copyCPU(newkeyFrame.der, keyframeData.der, lvl);
        copyGPU(newkeyFrame.der, keyframeData.der, lvl);
    }
}

void mesh_vo::addFrameToStack(frame &_frame)
{
    lastFrameAdded++;
    if(lastFrameAdded >= MAX_FRAMES)
        lastFrameAdded = 0;

    //copy texture and generate its mipmaps
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        copyCPU(_frame.image, frameDataStack[lastFrameAdded].image,lvl);
        copyGPU(_frame.image, frameDataStack[lastFrameAdded].image, lvl);

        copyCPU(_frame.der, frameDataStack[lastFrameAdded].der, lvl);
        copyGPU(_frame.der, frameDataStack[lastFrameAdded].der, lvl);
    }

    frameDataStack[lastFrameAdded].pose = _frame.pose;
    frameDataStack[lastFrameAdded].init = true;
}

float mesh_vo::calcErrorGPU(frame &_frame, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    //return calcResidual_CPU(frame, framePose, lvl);

    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, errorData.gpuTexture, lvl);

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
    glBindTexture(GL_TEXTURE_2D, _frame.image.gpuTexture);


    // activate shader
    errorShader.use();

    errorShader.setMat4("framePose", eigen2glm_mat4((_frame.pose*keyframeData.pose.inverse()).matrix()));
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


    //int new_lvl[MAX_LEVELS] = {5,6,7,8,9};
    //reduceFloat(errorTexture, lvl, new_lvl[lvl]);
    //int new_lvl[MAX_LEVELS] = {0,1,2,3,4};

    errorData.generateMipmapsGPU(lvl);

    int new_lvl = lvl+0;
    errorData.gpu_to_cpu(new_lvl);

    float error = 0.0;
    int count = 0;
    for(int x = 0; x < width[new_lvl]; x++)
        for(int y = 0; y < height[new_lvl]; y++)
        {
            float res = errorData.cpuTexture[new_lvl].at<float>(y,x);

            //if(x == width[new_lvl]/2 && y == height[new_lvl]/2)
            //std::cout << "res " << res << std::endl;
            if(res == error_value)
                //    res = 255.0*255.0;
                continue;

            count++;
            error += res;
        }

    if(count > 0)//width[new_lvl]*height[new_lvl]*0.7)
        error /= count;
    else
    {
        std::cout << "some problem in calcErrorGPU, maybe images dont overlap" << std::endl;
        error = 1230000000000000000000000000.0f;
    }

    //std::cout << "max error GPU " << max_error << std::endl;

    //std::cout << "lvl " << lvl << " new_lvl " << new_lvl << " error " << error << std::endl;

    return error;
}

float mesh_vo::calcErrorCPU(frame &_frame, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    float max_error = 0.0;

    float residual = 0.0;
    int num = 0;

    Sophus::SE3f relativePose = _frame.pose*keyframeData.pose.inverse();

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            uchar vkf = keyframeData.image.cpuTexture[lvl].at<uchar>(y,x);
            float keyframeId = keyframeData.idepth.cpuTexture[lvl].at<float>(y,x);

            if(keyframeId <= 0.0)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(fxinv[lvl]*(x+0.5) + cxinv[lvl], fyinv[lvl]*(y+0.5) + cyinv[lvl],1.0)/keyframeId;
            Eigen::Vector3f pointFrame = relativePose*poinKeyframe;

            //std::cout << "pointKeyframe " << poinKeyframe << std::endl;
            //std::cout << "pointFrame " << pointFrame << std::endl;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = Eigen::Vector3f(fx[lvl]*pointFrame(0)/pointFrame(2) + cx[lvl], fy[lvl]*pointFrame(1)/pointFrame(2) + cy[lvl], 1.0);

            if(pixelFrame(0) < 0.0 || pixelFrame(0) > width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) > height[lvl])
                continue;

            //std::cout << "pixelFrame " << pixelFrame << std::endl;
            uchar vf = _frame.image.cpuTexture[lvl].at<uchar>(pixelFrame(1), pixelFrame(0));

            float error = (vkf-vf)*(vkf-vf);

            errorData.cpuTexture[lvl].at<float>(y,x) = error;

            if(error > max_error)
                max_error = error;

            residual += error;
            num++;
        }

    //std::cout << "max error CPU " << max_error << std::endl;

    //cv::namedWindow("calcResidual debug", cv::WINDOW_NORMAL);
    //cv::imshow("calcResidual debug", debug);
    //cv::waitKey(30);

    return residual/num;
}

void mesh_vo::calcPose(frame &_frame, Sophus::SE3f initialGuessPose)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    Sophus::SE3f bestPose = initialGuessPose;
    _frame.pose = bestPose;

    for(int lvl=4; lvl >= 1; lvl--)
    {

        float last_error = calcErrorCPU(_frame,lvl);
        //float last_error = calcErrorGPU(_frame, lvl);

        //std::cout << "lvl " << lvl << " " << last_error << std::endl;

        for(int it = 0; it < maxIterations[lvl]; it++)
        {
            acc_J_pose.setZero();
            acc_H_pose.setZero();

            calcHJPoseCPU(_frame ,lvl);
            //calcHJPose(_frameTexture, _frameDerivativeTexture, framePose, lvl);
            //calcHJPose2(keyframeTexture, keyframeDerivativeTexture, frameTexture, frameDerivativeTexture, framePose, lvl);
            //showTecalcHJPosexture(residualTexture, lvl);

            float lambda = 0.0;
            int n_try = 0;
            while(true)
            {
                Eigen::Matrix<float, 6, 6> acc_H_pose_lambda;
                acc_H_pose_lambda = acc_H_pose;

                for(int j = 0; j < 6; j++)
                    acc_H_pose_lambda(j,j) *= 1.0 + lambda;

                inc_pose = acc_H_pose_lambda.ldlt().solve(acc_J_pose);

                //std::cout << "acc_J_pose " << acc_J_pose << std::endl;

                //Sophus::SE3f new_pose = framePose.inverse()*Sophus::SE3f::exp(inc).inverse();
                //Sophus::SE3f new_pose = Sophus::SE3f::exp(inc) * framePose;
                //Sophus::SE3f new_pose = Sophus::SE3f::exp(inc).inverse() * framePose;
                //Sophus::SE3f new_pose = framePose*Sophus::SE3f::exp(inc);

                Sophus::SE3f new_pose = bestPose*Sophus::SE3f::exp(inc_pose).inverse();
                _frame.pose = new_pose;

                //std::cout << "new_pose " << new_pose.matrix() << std::endl;

                //float error = calcErrorGPU(_frame, lvl);
                float error = calcErrorCPU(_frame,lvl);

                //std::cout << "new error " << error << std::endl;

                if(error < last_error)
                {
                    //accept update, decrease lambda
                    bestPose = new_pose;

                    float p = error / last_error;

                    if(lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

                    last_error = error;

                    if( p >  0.999f)
                    {
                        //std::cout << "lvl " << lvl << " converged after " << it << " itarations and " << t_try << " total tries, and lambda " << lambda << std::endl;
                        //if converged, do next level
                        it = maxIterations[lvl];
                    }

                    //if update accepted, do next iteration
                    break;
                }
                else
                {
                    _frame.pose = bestPose;

                    n_try++;

                    if(lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= std::pow(2.0, n_try);

                    //reject update, increase lambda, use un-updated data
                    //std::cout << "update rejected " << std::endl;

                    if(!(inc_pose.dot(inc_pose) > 1e-8))
                        //if(!(inc.dot(inc) > 1e-6))
                    {
                        //std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
                        //if too small, do next level!
                        it = maxIterations[lvl];
                        break;
                    }
                }
            }
        }
    }
}

void mesh_vo::updateMap()
{
    /*
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == false)
            return;
    }
    */

    for(int lvl = 0; lvl >= 0; lvl--)
    {
        float last_error = 0.0;
        for(int i = 0; i < MAX_FRAMES; i++)
        {
            //last_error += calcErrorCPU(frameDataStack[i], lvl);
            if(frameDataStack[i].init == true)
              last_error += calcErrorGPU(frameDataStack[i], lvl);
        }

        //showGPU(errorData,lvl);
        //cv::waitKey(1000.0);
        //return;

        std::cout << "lvl " << lvl << " initial error " << last_error << std::endl;

        int maxIterations = 10;
        float lambda = 0.0;
        for(int it = 0; it < maxIterations; it++)
        {
            //acc_H_map.setZero();
            //acc_J_map.setZero();
            //inc_map.setZero();

            acc_H_depth.setZero();
            acc_J_depth.setZero();
            inc_depth.setZero();
            acc_count.setZero();

            for(int i = 0; i < MAX_FRAMES; i++)
            {
                //showGPU(frameDataStack[i].image,0);
                //calcPose(frameDataStack[i],frameDataStack[i].pose);
                if(frameDataStack[i].init == true)
                  calcHJMapGPU(frameDataStack[i], lvl,lvl+0);
            }

            showGPU(primitiveIdData,lvl);
            cv::waitKey(100.0);
            //return;
            //std::cout << "acc_J_map " << acc_J_map << std::endl;

            int n_try = 0;
            while(true)
            {

                //Eigen::MatrixXf acc_H_depth_lambda = acc_H_depth;
                Eigen::SparseMatrix<float> acc_H_depth_lambda = acc_H_depth;

                std::vector<float> diagonal(0.0);
                for(int j = 0; j < acc_H_depth_lambda.rows(); j++)
                {
                    //if(acc_count(j) > 0)
                    {
                        //acc_J_depth(j) /= acc_count(j);
                        acc_H_depth_lambda.coeffRef(j,j) *= (1.0+lambda);///acc_count(j);
                    }
                    diagonal.push_back(acc_H_depth_lambda.coeffRef(j,j));
                    //diagonal.push_back(fabs(acc_J_depth(j)));

                }

/*
                acc_H_depth_lambda.makeCompressed();
                //Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
                //Eigen::SparseLU<Eigen::SparseMatrix<float> > solver;
                //Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
                Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;
                solver.analyzePattern(acc_H_depth_lambda);
                //std::cout << solver.info() << std::endl;
                solver.factorize(acc_H_depth_lambda);
                //std::cout << solver.lastErrorMessage() << std::endl;
                inc_depth = -solver.solve(acc_J_depth);
                //inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                //inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                //inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);
*/


                std::cout << "lambda " << lambda << std::endl;
                std::sort(diagonal.begin(), diagonal.end(),std::greater<float>());
                std::cout << "ini " << diagonal.at(0) << " end " << diagonal.at(diagonal.size()-1) << std::endl;
                float min_value = diagonal.at(int(diagonal.size()*0.9));
                std::cout << "min_value " << min_value << std::endl;


                for(int j = 0; j < int(acc_J_depth.size()); j++)
                {
                    float h = acc_H_depth_lambda.coeffRef(j,j);
                    float h2 = fabs(acc_J_depth(j));
                    //if(h > min_value)// && h2 > min_value)
                    //if(acc_count(j) > 20)
                    if(fabs(acc_J_depth(j)) > 0.0)
                    {
                        //inc_depth(j) = -acc_J_depth(j)/h;
                        inc_depth(j) = -(1.0/(1.0+lambda))*acc_J_depth(j)/fabs(acc_J_depth(j));
                        //std::cout << "update" << std::endl;
                    }
                }

                //std::cout << acc_H_depth << std::endl;
                //std::cout << "acc_J_depth " << std::endl;
                //std::cout << acc_J_depth << std::endl;
                //std::cout << "inc_depth" << std::endl;
                //std::cout << inc_depth << std::endl;

                for(int index=0; index < int(scene_vertices.size()); index++)
                {
                    scene_vertices_updated[index] = scene_vertices[index];// + inc_depth(index);
                }


                for(int index=0; index < int(inc_depth.size()); index++)
                {
                    /*
                float max_depth = 1.0/min_idepth;
                float min_depth = 1.0/max_idepth;
                float max_depth_jump = (max_depth - min_depth)*0.1;

                if(inc_depth(index) > log(max_depth_jump))
                    inc_depth(index) = log(max_depth_jump);
                if(inc_depth(index) < log(-max_depth_jump))
                    inc_depth(index) = log(-max_depth_jump);
                */

                    if(inc_depth(index)!=inc_depth(index))
                    {
                        std::cout << "some nand in inc_depth" << std::endl;
                    }
/*
                    float max_inc = 0.1;
                    if(inc_depth(index) > max_inc)
                        inc_depth(index) = max_inc;
                    if(inc_depth(index) < -max_inc)
                        inc_depth(index) = -max_inc;
*/
                    scene_vertices_updated[index*3+2] = scene_vertices[index*3+2] + inc_depth(index);

                    //if(scene_vertices_updated[index*3+2] != scene_vertices_updated[index*3+2])
                    //    std::cout << "some nand in updating scene_vertices" << std::endl;


                    if(scene_vertices_updated[index*3+2] < min_idepth)
                        scene_vertices_updated[index*3+2] = scene_vertices[index*3+2];
                    if(scene_vertices_updated[index*3+2] > max_idepth)
                        scene_vertices_updated[index*3+2] = scene_vertices[index*3+2];

                    //if(scene_vertices_updated[index*3+2] > log(max_depth))
                    //    scene_vertices_updated[index*3+2] = log(max_depth);

                    //if(scene_vertices_updated[index*3+2] < 0.0)
                    //    std::cout << "negative depth " << index << " " << scene_vertices_updated[index*3+2] << std::endl;
                }


                glBindVertexArray(scene_VAO);
                glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices_updated.size(), scene_vertices_updated.data());

                //calcIdepthGPU(keyframeData,0);
                //keyframeData.idepth.gpu_to_cpu(0);
                //keyframeData.idepth.generateMipmapsCPU(0);

                float error = 0.0;
                for(int i=0; i < MAX_FRAMES; i++)
                {
                    //error += calcErrorCPU(frameDataStack[i], lvl);
                    if(frameDataStack[i].init == true)
                      error += calcErrorGPU(frameDataStack[i],lvl);
                }

                std::cout << "lvl " << lvl << " new error " << error << std::endl;

                if(error < last_error)
                {
                    //accept update, decrease lambda
                    std::cout << "update accepted " << std::endl;


                    for(int index=0; index < int(scene_vertices.size()); index++)
                    {
                        scene_vertices[index] = scene_vertices_updated[index];
                    }

                    /*
                    //update frames pose
                    calcIdepthGPU(keyframeData,0);
                    keyframeData.idepth.gpu_to_cpu(0);
                    keyframeData.idepth.generateMipmapsCPU(0);

                    for(int i=0; i < MAX_FRAMES; i++)
                    {
                       calcPose(frameDataStack[i],frameDataStack[i].pose);
                    }
                    */


                    float p = error / last_error;

                    if(lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

                    last_error = error;

                    if( p > 0.999f)
                    {
                        std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
                        //if converged, do next level
                        it = maxIterations;
                    }

                    //if update accepted, do next iteration
                    break;
                }
                else
                {
                    n_try++;

                    if(lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0;// std::pow(2.0, n_try);

                    glBindVertexArray(scene_VAO);
                    glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
                    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

                    //calcIdepthGPU(keyframeData,0);
                    //keyframeData.idepth.gpu_to_cpu(0);
                    //keyframeData.idepth.generateMipmapsCPU(0);

                    //reject update, increase lambda, use un-updated data
                    std::cout << "update rejected " << std::endl;


                    if(inc_depth.dot(inc_depth) < 1e-32)
                    {
                        std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations with lambda " << lambda << std::endl;
                        //if too small, do next level!
                        it = maxIterations;
                        break;
                    }
                }
            }
        }
    }
}

void mesh_vo::calcHJPoseGPU(frame _frame, int lvl)
{   
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, residualData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, traData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, rotData.gpuTexture, lvl);

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
    glBindTexture(GL_TEXTURE_2D, _frame.image.gpuTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _frame.der.gpuTexture);

    // activate shader
    calcJShader.use();

    calcJShader.setMat4("framePose", eigen2glm_mat4((_frame.pose*keyframeData.pose.inverse()).matrix()));
    calcJShader.setMat4("opencv2opengl", opencv2opengl);
    calcJShader.setMat4("projection", projMat[lvl]);
    calcJShader.setFloat("fx", fx[lvl]);
    calcJShader.setFloat("fy", fy[lvl]);
    calcJShader.setFloat("cx", cx[lvl]);
    calcJShader.setFloat("cy", cy[lvl]);
    calcJShader.setFloat("fxinv", fxinv[lvl]);
    calcJShader.setFloat("fyinv", fyinv[lvl]);
    calcJShader.setFloat("cxinv", cxinv[lvl]);
    calcJShader.setFloat("cyinv", cyinv[lvl]);
    calcJShader.setFloat("dx", dx[lvl]);
    calcJShader.setFloat("dy", dy[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);


    int new_lvl[MAX_LEVELS] = {2,3,4,5,6};
    int new_lvl2[MAX_LEVELS] = {9,9,9,9,9};

    glViewport(0,0,width[new_lvl[lvl]],height[new_lvl[lvl]]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, Jpose1Data.gpuTexture, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, Jpose2Data.gpuTexture, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, Jpose3Data.gpuTexture, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, Jpose4Data.gpuTexture, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, Jpose5Data.gpuTexture, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, Jpose6Data.gpuTexture, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, Jpose6Data.gpuTexture, new_lvl[lvl]);

    unsigned int drawbuffers2[]={GL_COLOR_ATTACHMENT0,
                                 GL_COLOR_ATTACHMENT1,
                                 GL_COLOR_ATTACHMENT2,
                                 GL_COLOR_ATTACHMENT3,
                                 GL_COLOR_ATTACHMENT4,
                                 GL_COLOR_ATTACHMENT5,
                                 GL_COLOR_ATTACHMENT6};
    glDrawBuffers(sizeof(drawbuffers2)/sizeof(unsigned int), drawbuffers2);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! HJPose" << std::endl;

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    //glClearColor(-1.0f, -1.0f, -1.0f, -1.0f);
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, residualData.gpuTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, traData.gpuTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, rotData.gpuTexture);

    calcHJShader.use();
    calcHJShader.setInt("lvlin", lvl);
    calcHJShader.setInt("lvlout", new_lvl[lvl]);

    glBindVertexArray(frame_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, 0, 0);

    /*
    reduceVec4(JposeTexture1, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, (float *)JposeMat1[new_lvl2[lvl]].data);
    reduceVec4(JposeTexture2, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, (float *)JposeMat2[new_lvl2[lvl]].data);
    reduceVec4(JposeTexture3, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, (float *)JposeMat3[new_lvl2[lvl]].data);
    reduceVec4(JposeTexture4, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, (float *)JposeMat4[new_lvl2[lvl]].data);
    reduceVec4(JposeTexture5, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, (float *)JposeMat5[new_lvl2[lvl]].data);
    reduceVec4(JposeTexture6, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, (float *)JposeMat6[new_lvl2[lvl]].data);
    reduceVec4(JposeTexture7, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, (float *)JposeMat7[new_lvl2[lvl]].data);
    */

    int count = 0;
    for(int x = 0; x < width[new_lvl2[lvl]]; x++)
        for(int y = 0; y < height[new_lvl2[lvl]]; y++)
        {
            /*
            if(residual == 0.0f)
                continue;

            if(J_pose.norm() == 0.0f)
                continue;
            */

            Eigen::Matrix<float, 6, 1> J_pose;
            //J_pose << JposeMat1[new_lvl2[lvl]].at<cv::Vec4f>(y,x)[0], JposeMat1[new_lvl2[lvl]].at<cv::Vec4f>(y,x)[1], JposeMat1[new_lvl2[lvl]].at<cv::Vec4f>(y,x)[2], JposeMat1[new_lvl2[lvl]].at<cv::Vec4f>(y,x)[3], JposeMat2[new_lvl2[lvl]].at<cv::Vec4f>(y,x)[0], JposeMat2[new_lvl2[lvl]].at<cv::Vec4f>(y,x)[1];

            if(J_pose.norm() == 0.0f)
                continue;

            count++;

            acc_J_pose += J_pose;

            /*
            acc_J_pose(0) += j_pose_data1[index*4];
            acc_J_pose(1) += j_pose_data1[index*4+1];
            acc_J_pose(2) += j_pose_data1[index*4+2];
            acc_J_pose(3) += j_pose_data1[index*4+3];
            acc_J_pose(4) += j_pose_data2[index*4];
            acc_J_pose(5) += j_pose_data2[index*4+1];
            */

            /*
            acc_H_pose(0,0) += j_pose_data2[index*4+2];

            acc_H_pose(0,1) += j_pose_data2[index*4+3];
            acc_H_pose(1,0) += j_pose_data2[index*4+3];
            acc_H_pose(0,2) += j_pose_data3[index*4];
            acc_H_pose(2,0) += j_pose_data3[index*4];
            acc_H_pose(0,3) += j_pose_data3[index*4+1];
            acc_H_pose(3,0) += j_pose_data3[index*4+1];
            acc_H_pose(0,4) += j_pose_data3[index*4+2];
            acc_H_pose(4,0) += j_pose_data3[index*4+2];
            acc_H_pose(0,5) += j_pose_data3[index*4+3];
            acc_H_pose(5,0) += j_pose_data3[index*4+3];

            acc_H_pose(1,1) += j_pose_data4[index*4];

            acc_H_pose(1,2) += j_pose_data4[index*4+1];
            acc_H_pose(2,1) += j_pose_data4[index*4+1];
            acc_H_pose(1,3) += j_pose_data4[index*4+2];
            acc_H_pose(3,1) += j_pose_data4[index*4+2];
            acc_H_pose(1,4) += j_pose_data4[index*4+3];
            acc_H_pose(4,1) += j_pose_data4[index*4+3];
            acc_H_pose(1,5) += j_pose_data5[index*4];
            acc_H_pose(5,1) += j_pose_data5[index*4];

            acc_H_pose(2,2) += j_pose_data5[index*4+1];

            acc_H_pose(2,3) += j_pose_data5[index*4+2];
            acc_H_pose(3,2) += j_pose_data5[index*4+2];
            acc_H_pose(2,4) += j_pose_data5[index*4+3];
            acc_H_pose(4,2) += j_pose_data5[index*4+3];
            acc_H_pose(2,5) += j_pose_data6[index*4];
            acc_H_pose(5,2) += j_pose_data6[index*4];

            acc_H_pose(3,3) += j_pose_data6[index*4+1];

            acc_H_pose(3,4) += j_pose_data6[index*4+2];
            acc_H_pose(4,3) += j_pose_data6[index*4+2];
            acc_H_pose(3,5) += j_pose_data6[index*4+3];
            acc_H_pose(5,3) += j_pose_data6[index*4+3];

            acc_H_pose(4,4) += j_pose_data7[index*4];

            acc_H_pose(4,5) += j_pose_data7[index*4+1];
            acc_H_pose(5,5) += j_pose_data7[index*4+1];

            acc_H_pose(5,5) += j_pose_data7[index*4+2];
            */
        }

    /*
    //old implementation, i cannot "reduce" it in gpu i think
    glBindTexture(GL_TEXTURE_2D, residualTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, residual_cpu_data);

    glBindTexture(GL_TEXTURE_2D, traTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RGB, GL_FLOAT, tra_cpu_data);

    glBindTexture(GL_TEXTURE_2D, rotTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RGB, GL_FLOAT, rot_cpu_data);

    for(int index = 0; index < width[lvl]*height[lvl]; index+=1)
    {
        Eigen::Matrix<float, 6, 1> J_pose;
        J_pose << tra_cpu_data[index*3], tra_cpu_data[index*3+1], tra_cpu_data[index*3+2], rot_cpu_data[index*3], rot_cpu_data[index*3+1], rot_cpu_data[index*3+2];


        //for(int i = 0; i < 6; i++)
        //{
        //    if(J_pose[i]!=J_pose[i])
        //        std::cout << "nand " << J_pose[i] << std::endl;
        //}


        float residual = residual_cpu_data[index];

        //std::cout << "J_pose " << J_pose << std::endl;
        //std::cout << "residual " << residual << std::endl;


        if(residual == 0.0f)
            continue;

        if(J_pose.norm() == 0.0f)
            continue;


        acc_J_pose += J_pose*residual;

        for(int i = 0; i < 6; i++)
        {
            acc_H_pose(i,i) += J_pose[i]*J_pose[i];

            for(int j = i+1; j < 6; j++)
            {
                float aux = J_pose[i]*J_pose[j];
                acc_H_pose(i,j) += aux;
                acc_H_pose(j,i) += aux;
            }
        }


        //for(int i = 0; i < 6; i++)
        //{
        //    for(int j = 0; j < 6; j++)
        //    {
        //        acc_H_pose(i,j) += J_pose[i]*J_pose[j];
        //    }
        //}


        //std::cout << "acc_J_pose " << acc_J_pose << std::endl;
        //std::cout << "acc_J_pose " << acc_H_pose << std::endl;
    }
    */
}
/*
void mesh_vo::reduceFloatGPU(data _data, int src_lvl, int dst_lvl)
{

    //reduce by mipmap generation
    //glBindTexture(GL_TEXTURE_2D, texture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, src_lvl);
    //glGenerateMipmap(GL_TEXTURE_2D);


    //reduce by shader call
    int swp = 2;
    unsigned int in_texture = reduceFloatTexture;
    unsigned int out_texture = texture;

    int step = 1;

    for(int in_lvl = src_lvl, out_lvl = std::min(src_lvl+step,dst_lvl); in_lvl < dst_lvl; in_lvl=out_lvl, out_lvl = std::min(out_lvl+step,dst_lvl))
    {
        //std::cout << "in_lvl " << in_lvl << " out_lvl " << out_lvl << std::endl;

        if(swp % 2 == 0)
        {
            in_texture = texture;
            out_texture = reduceFloatTexture;
        }
        else
        {
            in_texture = reduceFloatTexture;
            out_texture = texture;
        }
        swp++;

        glViewport(0,0,width[out_lvl],height[out_lvl]);

        //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, out_texture, out_lvl);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcResidual" << std::endl;

        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, in_texture);

        reduceFloatShader.use();
        reduceFloatShader.setInt("lvlin", in_lvl);
        reduceFloatShader.setInt("lvlout", out_lvl);

        glBindVertexArray(frame_VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    if(out_texture != texture)
    {
        glCopyTexSubImage2D(GL_TEXTURE_2D, dst_lvl, 0, 0, 0, 0, width[dst_lvl], height[dst_lvl]);
    }
}
*/

/*
void mesh_vo::reduceVec4(unsigned int texture, int src_lvl, int dst_lvl)
{

    //reduce by mipmap generation
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, src_lvl);
    glGenerateMipmap(GL_TEXTURE_2D);

    //reduce by shader call
    int swp = 2;
    unsigned int in_texture = reduceVec4Texture;
    unsigned int out_texture = texture;

    int step = 1;

    for(int in_lvl = src_lvl, out_lvl = std::min(src_lvl+step,dst_lvl); in_lvl < dst_lvl; in_lvl=out_lvl, out_lvl = std::min(out_lvl+step,dst_lvl))
    {
        //std::cout << "in_lvl " << in_lvl << " out_lvl " << out_lvl << std::endl;

        if(swp % 2 == 0)
        {
            in_texture = texture;
            out_texture = reduceVec4Texture;
        }
        else
        {
            in_texture = reduceVec4Texture;
            out_texture = texture;
        }
        swp++;

        glViewport(0,0,width[out_lvl],height[out_lvl]);

        //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, out_texture, out_lvl);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcResidual" << std::endl;

        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, in_texture);

        reduceVec4Shader.use();
        reduceVec4Shader.setInt("lvlin", in_lvl);
        reduceVec4Shader.setInt("lvlout", out_lvl);

        glBindVertexArray(frame_VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    if(out_texture != texture)
    {
        glCopyTexSubImage2D(GL_TEXTURE_2D, dst_lvl, 0, 0, 0, 0, width[dst_lvl], height[dst_lvl]);
    }
}
*/

/*
void mesh_vo::calcHJPoseGPU2(frame &_frame, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    int new_lvl[MAX_LEVELS] = {9,9,9,9,9};

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, Jpose1Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, Jpose2Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, Jpose3Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, Jpose4Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, Jpose5Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, Jpose6Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, Jpose7Data.gpuTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0,
                                GL_COLOR_ATTACHMENT1,
                                GL_COLOR_ATTACHMENT2,
                                GL_COLOR_ATTACHMENT3,
                                GL_COLOR_ATTACHMENT4,
                                GL_COLOR_ATTACHMENT5,
                                GL_COLOR_ATTACHMENT6};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcHJPose" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeData.image.gpuTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frame.image.gpuTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _frame.der.gpuTexture);

    // activate shader
    calcHJShader2.use();

    calcHJShader2.setMat4("framePose", eigen2glm_mat4(_frame.pose.matrix()));
    calcHJShader2.setMat4("opencv2opengl", opencv2opengl);
    calcHJShader2.setMat4("projection", projMat[lvl]);
    calcHJShader2.setFloat("fx", fx[lvl]);
    calcHJShader2.setFloat("fy", fy[lvl]);
    calcHJShader2.setFloat("cx", cx[lvl]);
    calcHJShader2.setFloat("cy", cy[lvl]);
    calcHJShader2.setFloat("fxinv", fxinv[lvl]);
    calcHJShader2.setFloat("fyinv", fyinv[lvl]);
    calcHJShader2.setFloat("cxinv", cxinv[lvl]);
    calcHJShader2.setFloat("cyinv", cyinv[lvl]);
    calcHJShader2.setFloat("dx", dx[lvl]);
    calcHJShader2.setFloat("dy", dy[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, 0, 0);

    //glActiveTexture(GL_TEXTURE0);
    //glBindTexture(GL_TEXTURE_2D, residualTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    //glGenerateMipmap(GL_TEXTURE_2D);

    //glBindTexture(GL_TEXTURE_2D, traTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    //glGenerateMipmap(GL_TEXTURE_2D);

    //glBindTexture(GL_TEXTURE_2D, rotTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    //glGenerateMipmap(GL_TEXTURE_2D);

    //glBindTexture(GL_TEXTURE_2D, residualTexture);

    Jpose1Data.generateMipmapsGPU(lvl);
    Jpose1Data.gpu_to_cpu(new_lvl[lvl]);

    Jpose2Data.generateMipmapsGPU(lvl);
    Jpose2Data.gpu_to_cpu(new_lvl[lvl]);

    Jpose3Data.generateMipmapsGPU(lvl);
    Jpose3Data.gpu_to_cpu(new_lvl[lvl]);

    Jpose4Data.generateMipmapsGPU(lvl);
    Jpose4Data.gpu_to_cpu(new_lvl[lvl]);

    Jpose5Data.generateMipmapsGPU(lvl);
    Jpose5Data.gpu_to_cpu(new_lvl[lvl]);

    Jpose6Data.generateMipmapsGPU(lvl);
    Jpose6Data.gpu_to_cpu(new_lvl[lvl]);

    Jpose7Data.generateMipmapsGPU(lvl);
    Jpose7Data.gpu_to_cpu(new_lvl[lvl]);

    for(int x = 0; x < width[new_lvl[lvl]]; x++)
        for(int y = 0; y < height[new_lvl[lvl]]; y++)
        {

        //if(residual == 0.0f)
        //    continue;

        //if(J_pose.norm() == 0.0f)
        //    continue;


            Eigen::Matrix<float, 6, 1> J_pose;
            J_pose << Jpose1Data.cpuTexture[] j_pose_data1[index*4], j_pose_data1[index*4+1], j_pose_data1[index*4+2], j_pose_data1[index*4+3], j_pose_data2[index*4], j_pose_data2[index*4+1];

            if(J_pose.norm() == 0.0f)
                continue;

            acc_J_pose += J_pose;


        //acc_J_pose(0) += j_pose_data1[index*4];
        //acc_J_pose(1) += j_pose_data1[index*4+1];
        //acc_J_pose(2) += j_pose_data1[index*4+2];
        //acc_J_pose(3) += j_pose_data1[index*4+3];
        //acc_J_pose(4) += j_pose_data2[index*4];
        //acc_J_pose(5) += j_pose_data2[index*4+1];


            acc_H_pose(0,0) += j_pose_data2[index*4+2];

            acc_H_pose(0,1) += j_pose_data2[index*4+3];
            acc_H_pose(1,0) += j_pose_data2[index*4+3];
            acc_H_pose(0,2) += j_pose_data3[index*4];
            acc_H_pose(2,0) += j_pose_data3[index*4];
            acc_H_pose(0,3) += j_pose_data3[index*4+1];
            acc_H_pose(3,0) += j_pose_data3[index*4+1];
            acc_H_pose(0,4) += j_pose_data3[index*4+2];
            acc_H_pose(4,0) += j_pose_data3[index*4+2];
            acc_H_pose(0,5) += j_pose_data3[index*4+3];
            acc_H_pose(5,0) += j_pose_data3[index*4+3];

            acc_H_pose(1,1) += j_pose_data4[index*4];

            acc_H_pose(1,2) += j_pose_data4[index*4+1];
            acc_H_pose(2,1) += j_pose_data4[index*4+1];
            acc_H_pose(1,3) += j_pose_data4[index*4+2];
            acc_H_pose(3,1) += j_pose_data4[index*4+2];
            acc_H_pose(1,4) += j_pose_data4[index*4+3];
            acc_H_pose(4,1) += j_pose_data4[index*4+3];
            acc_H_pose(1,5) += j_pose_data5[index*4];
            acc_H_pose(5,1) += j_pose_data5[index*4];

            acc_H_pose(2,2) += j_pose_data5[index*4+1];

            acc_H_pose(2,3) += j_pose_data5[index*4+2];
            acc_H_pose(3,2) += j_pose_data5[index*4+2];
            acc_H_pose(2,4) += j_pose_data5[index*4+3];
            acc_H_pose(4,2) += j_pose_data5[index*4+3];
            acc_H_pose(2,5) += j_pose_data6[index*4];
            acc_H_pose(5,2) += j_pose_data6[index*4];

            acc_H_pose(3,3) += j_pose_data6[index*4+1];

            acc_H_pose(3,4) += j_pose_data6[index*4+2];
            acc_H_pose(4,3) += j_pose_data6[index*4+2];
            acc_H_pose(3,5) += j_pose_data6[index*4+3];
            acc_H_pose(5,3) += j_pose_data6[index*4+3];

            acc_H_pose(4,4) += j_pose_data7[index*4];

            acc_H_pose(4,5) += j_pose_data7[index*4+1];
            acc_H_pose(5,5) += j_pose_data7[index*4+1];

            acc_H_pose(5,5) += j_pose_data7[index*4+2];
        }
}
*/

void mesh_vo::calcHJPoseCPU(frame &_frame, int lvl)
{
    acc_J_pose.setZero();
    acc_H_pose.setZero();

    Sophus::SE3f relativePose = _frame.pose*keyframeData.pose.inverse();

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            uchar vkf = keyframeData.image.cpuTexture[lvl].at<uchar>(y,x);
            float keyframeId = keyframeData.idepth.cpuTexture[lvl].at<float>(y,x);

            //std::cout << "keyframeId " << keyframeId << std::endl;

            if(keyframeId <= 0.0)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(fxinv[lvl]*x + cxinv[lvl],fyinv[lvl]*y + cyinv[lvl],1.0)/keyframeId;
            Eigen::Vector3f pointFrame = relativePose*poinKeyframe;

            //std::cout << "pointFrame " << pointFrame << std::endl;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = Eigen::Vector3f(fx[lvl]*pointFrame(0)/pointFrame(2) + cx[lvl], fy[lvl]*pointFrame(1)/pointFrame(2) + cy[lvl], 1.0);

            //std::cout << "pixelFrame " << std::endl;

            if(pixelFrame(0) < 0.0 || pixelFrame(0) >= width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= height[lvl])
                continue;

            uchar vf = _frame.image.cpuTexture[lvl].at<uchar>(pixelFrame(1), pixelFrame(0));
            cv::Vec2f der = _frame.der.cpuTexture[lvl].at<cv::Vec2f>(pixelFrame(1),pixelFrame(0));

            Eigen::Vector2f d_f_d_uf(der.val[0],der.val[1]);

            //std::cout << "vf " << vf << " der " << der << std::endl;

            float id = 1.0/pointFrame(2);

            float v0 = d_f_d_uf(0) * fx[lvl] * id;
            float v1 = d_f_d_uf(1) * fy[lvl] * id;
            float v2 = -(v0 * pointFrame(0) + v1 * pointFrame(1)) * id;

            Eigen::Vector3f d_I_d_tra = Eigen::Vector3f(v0, v1, v2);
            Eigen::Vector3f d_I_d_rot = Eigen::Vector3f( -pointFrame(2) * v1 + pointFrame(1) * v2, pointFrame(2) * v0 - pointFrame(0) * v2, -pointFrame(1) * v0 + pointFrame(0) * v1);

            float residual = (vf - vkf);

            Eigen::Matrix<float, 6, 1> J_pose;
            J_pose << d_I_d_tra(0), d_I_d_tra(1), d_I_d_tra(2), d_I_d_rot(0), d_I_d_rot(1), d_I_d_rot(2);

            acc_J_pose += J_pose*residual;

            for(int i = 0; i < 6; i++)
            {
                for(int j = 0; j < 6; j++)
                {
                    acc_H_pose(i,j) += J_pose[i]*J_pose[j];
                }
            }
        }
}

void mesh_vo::calcHJMapGPU(frame &_frame, int lvl, int srclvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vertexIdData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, primitiveIdData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, residualData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, d_I_d_p0Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, d_I_d_p1Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, d_I_d_p2Data.gpuTexture, lvl);

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
    calcHJMapShader.use();

    calcHJMapShader.setMat4("framePose", eigen2glm_mat4((_frame.pose*keyframeData.pose.inverse()).matrix()));
    calcHJMapShader.setMat4("opencv2opengl", opencv2opengl);
    calcHJMapShader.setMat4("projection", projMat[lvl]);
    calcHJMapShader.setFloat("fx", fx[lvl]);
    calcHJMapShader.setFloat("fy", fy[lvl]);
    calcHJMapShader.setFloat("cx", cx[lvl]);
    calcHJMapShader.setFloat("cy", cy[lvl]);
    calcHJMapShader.setFloat("fxinv", fxinv[lvl]);
    calcHJMapShader.setFloat("fyinv", fyinv[lvl]);
    calcHJMapShader.setFloat("cxinv", cxinv[lvl]);
    calcHJMapShader.setFloat("cyinv", cyinv[lvl]);
    calcHJMapShader.setFloat("dx", dx[lvl]);
    calcHJMapShader.setFloat("dy", dy[lvl]);
    calcHJMapShader.setInt("srclvl",srclvl);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, 0, 0);

    int new_lvl = lvl + 0;

    vertexIdData.generateMipmapsGPU(lvl);
    primitiveIdData.generateMipmapsGPU(lvl);
    residualData.generateMipmapsGPU(lvl);
    d_I_d_p0Data.generateMipmapsGPU(lvl);
    d_I_d_p1Data.generateMipmapsGPU(lvl);
    d_I_d_p2Data.generateMipmapsGPU(lvl);

    vertexIdData.gpu_to_cpu(new_lvl);
    primitiveIdData.gpu_to_cpu(new_lvl);
    residualData.gpu_to_cpu(new_lvl);
    d_I_d_p0Data.gpu_to_cpu(new_lvl);
    d_I_d_p1Data.gpu_to_cpu(new_lvl);
    d_I_d_p2Data.gpu_to_cpu(new_lvl);

    for(int y = 0; y < height[new_lvl]; y++)
        for(int x = 0; x < width[new_lvl]; x++)
        {
            //std::cout << "pixel index " << index << std::endl;
            //acumular para cada pixel
            float error = residualData.cpuTexture[new_lvl].at<float>(y,x);

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
            vertexID[0] = int(vertexIdData.cpuTexture[new_lvl].at<cv::Vec3f>(y,x)[0]);
            vertexID[1] = int(vertexIdData.cpuTexture[new_lvl].at<cv::Vec3f>(y,x)[1]);
            vertexID[2] = int(vertexIdData.cpuTexture[new_lvl].at<cv::Vec3f>(y,x)[2]);

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
            J[0] = d_I_d_p0Data.cpuTexture[new_lvl].at<float>(y,x);
            J[1] = d_I_d_p1Data.cpuTexture[new_lvl].at<float>(y,x);
            J[2] = d_I_d_p2Data.cpuTexture[new_lvl].at<float>(y,x);

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

            //ahora si, actualizo las matrices usando los indices de cada vertice
            for(int i = 0; i < 3; i++)
            {
                acc_J_depth(vertexID[i]) += J[i]*error;
                acc_count(vertexID[i]) += 1;

                for(int j = 0; j < 3; j++)
                {
                    //acc_H_depth(vertexID[i],vertexID[j]) += J[i]*J[j];
                    acc_H_depth.coeffRef(vertexID[i],vertexID[j]) += J[i]*J[j];
                }
            }
        }
}

void mesh_vo::visual_odometry(cv::Mat _frame)
{
    glfwMakeContextCurrent(frameWindow);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    showGPU(keyframeData.idepth,0);

    //save frame in gpu memory, calc derivavites y mipmaps
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_frame,frameData.image.cpuTexture[lvl],cv::Size(width[lvl], height[lvl]), cv::INTER_LANCZOS4);
        frameData.image.cpu_to_gpu(lvl);

        calcDerivativeCPU(frameData,lvl);
        frameData.der.cpu_to_gpu(lvl);
    }

    tic_toc t;
    t.tic();
    //frameData.pose = _globalPose*keyframeData.pose.inverse();
    calcPose(frameData, frameData.pose);//*Sophus::SE3f::exp(inc_pose).inverse());
    glFinish();
    if(calcPoseTime == 0.0)
        calcPoseTime = t.toc();
    else
        calcPoseTime = calcPoseTime*0.9 + t.toc()*0.1;
    std::cout << "estimated pose " << std::endl;
    std::cout << frameData.pose.matrix() << std::endl;
    std::cout << "clacPose time " << calcPoseTime << std::endl;

    float new_occupancy = calcOccupancyCPU(frameData, 0);

    std::cout << "occ " << new_occupancy << std::endl;

    //showTexture(keyframeIdepthTexture, 0);
    //cv::imshow("idepth", keyframeIdepthMat[0]);
    //cv::waitKey(30);
    //return;

    float diff = fabs(occupancy - new_occupancy);
    occupancy = new_occupancy;

    float norm = (frameData.pose.translation() - keyframeData.pose.translation()).norm();

    if(occupancy < 0.8)
    {
        changeKeyframe(frameData);
        updateMap();
        calcIdepthGPU(keyframeData,0);
        showGPU(keyframeData.idepth,0);
        keyframeData.idepth.gpu_to_cpu(0);
        keyframeData.idepth.generateMipmapsCPU(0);

        occupancy = 1.0;

        /*
        for(int s = 0; s < MAX_FRAMES; s++)
        {
            if(frameDataStack[s].init == true)
              calcPose(frameDataStack[s],frameDataStack[s].pose);
        }
        */

        return;
    }

    //if(diff > 0.01 && norm > 0.01)
    {
        //std::cout << "sup diff " << diff << " add frame and update map" << std::endl;
        addFrameToStack(frameData);
        updateMap();
        calcIdepthGPU(keyframeData,0);
        showGPU(keyframeData.idepth,0);
        keyframeData.idepth.gpu_to_cpu(0);
        keyframeData.idepth.generateMipmapsCPU(0);

        for(int s = 0; s < MAX_FRAMES; s++)
        {
            if(frameDataStack[s].init == true)
              calcPose(frameDataStack[s],frameDataStack[s].pose);
        }
    }
}

void mesh_vo::mapping(cv::Mat _frame, Sophus::SE3f _globalPose)
{
    tic_toc t;

    glfwMakeContextCurrent(frameWindow);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    calcIdepthGPU(keyframeData,0);
    showGPU(keyframeData.idepth,0);

    t.tic();
    //save frame in gpu memory, calc derivavites y mipmaps
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_frame,frameData.image.cpuTexture[lvl],cv::Size(width[lvl], height[lvl]), cv::INTER_LANCZOS4);
        frameData.image.cpu_to_gpu(lvl);

        calcDerivativeCPU(frameData,lvl);
        frameData.der.cpu_to_gpu(lvl);
    }
    glFinish();
    std::cout << "save frame time " << t.toc() << std::endl;

    frameData.pose = _globalPose;//*keyframeData.pose.inverse();
    //trackedPose = calcPose(frameTexture, frameDerivativeTexture, trackedPose*Sophus::SE3f::exp(inc_pose).inverse());

    //calcErrorGPU(frameData,0);
    //showGPU(errorData,0);
    //return;

    t.tic();
    float new_occupancy = calcOccupancyCPU(frameData, 0);
    glFinish();
    std::cout << "occupancy time " << t.toc() << std::endl;

    float diff = fabs(occupancy - new_occupancy);
    occupancy = new_occupancy;

    if(occupancy < 0.8)
    {
        t.tic();
        changeKeyframe(frameData);
        updateMap();

        //calcIdepthGPU(keyframeData,0);
        //showGPU(keyframeData.idepth,0);
        //keyframeData.idepth.gpu_to_cpu(0);
        //keyframeData.idepth.generateMipmapsCPU(0);

        glFinish();
        std::cout << "change keyframe time " << t.toc() << std::endl;

        occupancy = 1.0;
        return;
    }

    float norm = (frameData.pose.translation() - keyframeData.pose.translation()).norm();

    if(diff > 0.01)
    {
        t.tic();
        addFrameToStack(frameData);
        glFinish();
        std::cout << "add frame stack time " << t.toc() << std::endl;

        t.tic();
        updateMap();
        //calcIdepthGPU(keyframeData,0);
        //showGPU(keyframeData.idepth,0);
        //keyframeData.idepth.gpu_to_cpu(0);
        //keyframeData.idepth.generateMipmapsCPU(0);
        glFinish();
        std::cout << "update map time " << t.toc() << std::endl;
    }
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

float mesh_vo::calcOccupancyCPU(frame &_frame, int lvl)
{
    int count = 0;

    Sophus::SE3f relativePose = _frame.pose*keyframeData.pose.inverse();

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            float keyframeId = keyframeData.idepth.cpuTexture[lvl].at<float>(y,x);

            if(keyframeId <= 0.0)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(fxinv[lvl]*(float(x)+0.5) + cxinv[lvl], fyinv[lvl]*(float(y)+0.5) + cyinv[lvl],1.0)/keyframeId;
            Eigen::Vector3f pointFrame = relativePose*poinKeyframe;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = Eigen::Vector3f(fx[lvl]*pointFrame(0)/pointFrame(2) + cx[lvl], fy[lvl]*pointFrame(1)/pointFrame(2) + cy[lvl], 1.0);

            if(pixelFrame(0) < 0.0 || pixelFrame(0) >= width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= height[lvl])
                continue;

            count++;
        }

    return float(count)/(width[lvl]*height[lvl]);
}


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

void mesh_vo::showCPU(data &_data, int lvl)
{
    cv::Mat toShow;
    cv::resize(_data.cpuTexture[lvl],toShow,cv::Size(width[0],height[0]));
    cv::imshow("showCPU", toShow);
    cv::waitKey(30);
}

void mesh_vo::copyCPU(data &_src, data &_dst, int lvl)
{
    _src.cpuTexture[lvl].copyTo(_dst.cpuTexture[lvl]);
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

void mesh_vo::calcDerivativeCPU(frame &_frame, int lvl)
{
    for(int y = 1; y < height[lvl]-1; y++)
        for(int x = 1; x < width[lvl]-1; x++)
        {
            cv::Vec2f d;
            d.val[0] = (_frame.image.cpuTexture[lvl].at<uchar>(y,x+1) - _frame.image.cpuTexture[lvl].at<uchar>(y,x-1))/2.0;
            d.val[1] = (_frame.image.cpuTexture[lvl].at<uchar>(y+1,x) - _frame.image.cpuTexture[lvl].at<uchar>(y-1,x))/2.0;

            _frame.der.cpuTexture[lvl].at<cv::Vec2f>(y,x) = d;
        }
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






