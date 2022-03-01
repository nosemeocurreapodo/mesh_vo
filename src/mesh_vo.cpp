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
    //prealocate
    for(int y=0;y<VERTEX_HEIGH;y++)
    {
        for(int x=0;x<VERTEX_WIDTH;x++)
        {
            scene_vertices.push_back(0.0);
            scene_vertices.push_back(0.0);
            scene_vertices.push_back(0.0);

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

    vertexIdData = data(height[0], width[0], 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    errorData = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    traData = data(height[0], width[0], 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    rotData = data(height[0], width[0], 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    d_I_d_p0Data = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    d_I_d_p1Data = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    d_I_d_p2Data = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    debugData = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    view3DData = data(height[0], width[0], 4, GL_UNSIGNED_BYTE, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    vertexViewCountShader.init("isVertexInView.vs","isVertexInView.gs","isVertexInView.fs");
    vertexViewCountShader.use();

    errorShader.init("error.vs", "error.gs", "error.fs");
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

void mesh_vo::initWithRandomIdepth(cv::Mat _keyFrame, Sophus::SE3f _pose)
{
    //const float* maxGradients = new_frame->maxGradients();

    scene_vertices.clear();

    for(int y=0;y<VERTEX_HEIGH;y++) //los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for(int x=0;x<VERTEX_WIDTH;x++)
        {
            float idepth = 0.1 + (1.0-0.1) * float(y)/VERTEX_HEIGH;
            //float idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
            //float idepth = 0.5;

            float xi = (float(x)/float(VERTEX_WIDTH-1))*width[0];
            float yi = (float(y)/float(VERTEX_HEIGH-1))*height[0];
            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
            Eigen::Vector3f r = Eigen::Vector3f(fxinv[0]*u(0) + cxinv[0], fyinv[0]*u(1) + cyinv[0], 1.0);
            Eigen::Vector3f p = r/idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(idepth);
        }
    }

    setTriangles();

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

    for(int y=0;y<VERTEX_HEIGH;y++) //los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for(int x=0;x<VERTEX_WIDTH;x++)
        {
            float xi = (float(x)/float(VERTEX_WIDTH-1))*width[0];
            float yi = (float(y)/float(VERTEX_HEIGH-1))*height[0];

            float idepth = _idepth.at<float>(yi,xi);
            /*
            if(idepth <= min_idepth)
                idepth = min_idepth;
            if(idepth > max_idepth)
                idepth = max_idepth;
                */
            if(idepth != idepth || idepth < 0.1 || idepth > 1.0)
                idepth = 0.1 + (1.0-0.1) * float(y)/VERTEX_HEIGH;

            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
            Eigen::Vector3f r = Eigen::Vector3f(fxinv[0]*u(0) + cxinv[0], fyinv[0]*u(1) + cyinv[0], 1.0);
            Eigen::Vector3f p = r/idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(idepth);
        }
    }

    setTriangles();

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

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_keyFrame,keyframeData.image.cpuTexture[lvl],cv::Size(width[lvl], height[lvl]), cv::INTER_LANCZOS4);
        keyframeData.image.cpu_to_gpu(lvl);

        calcDerivativeCPU(keyframeData, lvl);
        keyframeData.der.cpu_to_gpu(lvl);
    }
}

void mesh_vo::changeKeyframe(frame &newkeyFrame, int lvl, float min_occupancy)
{
    calcIdepthGPU(newkeyFrame, lvl);
    newkeyFrame.idepth.gpu_to_cpu(lvl);

    while(true)
    {
        cv::Mat oldImage = newkeyFrame.idepth.cpuTexture[lvl];

        if(cv::countNonZero(oldImage) == width[lvl]*height[lvl])
            break;

        cv::Mat newImage;
        oldImage.copyTo(newImage);

        for(int y = 0; y < height[lvl]; y++)
            for(int x = 0; x < width[lvl]; x++)
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
        newImage.copyTo(newkeyFrame.idepth.cpuTexture[lvl]);
    }

    scene_vertices.clear();

    for(int y=0;y<VERTEX_HEIGH;y++) //los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for(int x=0;x<VERTEX_WIDTH;x++)
        {
            float xi = (float(x)/float(VERTEX_WIDTH-1))*width[lvl];
            float yi = (float(y)/float(VERTEX_HEIGH-1))*height[lvl];

            float idepth = newkeyFrame.idepth.cpuTexture[lvl].at<float>(yi,xi);

            if(idepth < min_idepth || idepth > max_idepth)
                idepth = 0.1 + (1.0-0.1) * float(y)/VERTEX_HEIGH;

            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
            Eigen::Vector3f r = Eigen::Vector3f(fxinv[lvl]*u(0) + cxinv[lvl], fyinv[lvl]*u(1) + cyinv[lvl], 1.0);
            Eigen::Vector3f p = r/idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(idepth);
        }
    }

    setTriangles();

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

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        copyCPU(newkeyFrame.image, keyframeData.image,lvl);
        copyGPU(newkeyFrame.image, keyframeData.image, lvl);

        copyCPU(newkeyFrame.der, keyframeData.der, lvl);
        copyGPU(newkeyFrame.der, keyframeData.der, lvl);
    }

    for(int index = 0; index < MAX_FRAMES; index++)
    {
        if(frameDataStack[index].init == true)
        {
            float occ = calcOccupancyCPU(frameDataStack[index],1);
            std::cout << "frame " << index << " " << occ << std::endl;
            if(occ < min_occupancy)
            {
                frameDataStack[index].init = false;

            }
        }
    }
}

void mesh_vo::setTriangles()
{
    scene_indices.clear();

    for(int y=0;y<VERTEX_HEIGH;y++)
    {
        for(int x=0;x<VERTEX_WIDTH;x++)
        {
            if(x>0 && y>0)
            {
                //if(((x % 2 == 0)))
                if(((x % 2 == 0) && (y % 2 == 0)) || ((x % 2 != 0) && (y % 2 != 0)))
                //if(rand() > 0.5*RAND_MAX)
                {
                    scene_indices.push_back(x-1 +     y*(VERTEX_WIDTH));
                    scene_indices.push_back(x   + (y-1)*(VERTEX_WIDTH));
                    scene_indices.push_back(x-1 + (y-1)*(VERTEX_WIDTH));

                    scene_indices.push_back(x   +     y*(VERTEX_WIDTH));
                    scene_indices.push_back(x   + (y-1)*(VERTEX_WIDTH));
                    scene_indices.push_back(x-1 +     y*(VERTEX_WIDTH));
                }
                else
                {
                    scene_indices.push_back(x   +     y*(VERTEX_WIDTH));
                    scene_indices.push_back(x-1 + (y-1)*(VERTEX_WIDTH));
                    scene_indices.push_back(x-1 +     y*(VERTEX_WIDTH));

                    scene_indices.push_back(x   +     y*(VERTEX_WIDTH));
                    scene_indices.push_back(x   + (y-1)*(VERTEX_WIDTH));
                    scene_indices.push_back(x-1 + (y-1)*(VERTEX_WIDTH));
                }
            }
        }
    }
}

void mesh_vo::addFrameToStack(frame &_frame)
{

    float frameError = errorGPU(_frame, 1);
    float minError = 100000000000.0;
    int minErrorIndex = -1;
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == true)
        {
            float error = errorGPU(frameDataStack[i],1);
            if(error < minError)
            {
                minError = error;
                minErrorIndex = i;
            }
        }
        else
        {
            minErrorIndex = i;
            break;
        }
    }

    if(minErrorIndex >= 0 && frameError > minErrorIndex)
    {
        std::cout << "add frame in index " << minErrorIndex << std::endl;
        //copy texture and generate its mipmaps
        for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            copyCPU(_frame.image, frameDataStack[minErrorIndex].image,lvl);
            copyGPU(_frame.image, frameDataStack[minErrorIndex].image, lvl);

            copyCPU(_frame.der, frameDataStack[minErrorIndex].der, lvl);
            copyGPU(_frame.der, frameDataStack[minErrorIndex].der, lvl);
        }

        frameDataStack[minErrorIndex].pose = _frame.pose;
        frameDataStack[minErrorIndex].init = true;
    }
    else
    {
        std::cout << "frame not added " << std::endl;
    }
    return;



    vertexViewCountGPU(_frame,1);
    float newGoodViewPercentage = goodVertexViewPercentage(_frame);

    //do view table
    bool isViewBy[VERTEX_HEIGH*VERTEX_WIDTH][MAX_FRAMES];
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == true)
        {
            vertexViewCountGPU(frameDataStack[i],1);
        }
        for(int j = 0; j < VERTEX_HEIGH*VERTEX_WIDTH; j++)
        {
            isViewBy[j][i] = false;

            if(frameDataStack[i].init == true)
            {
                if(frameDataStack[i].vertexViewCount(j) > 0)
                    isViewBy[j][i] = true;
            }
        }
    }

    //check if new frame adds some new information
    bool newInformation = false;
    for(int j = 0; j < VERTEX_HEIGH*VERTEX_WIDTH; j++)
    {
        bool viewedByNone = true;
        for(int i = 0; i < MAX_FRAMES; i++)
        {
            if(frameDataStack[i].init == true && frameDataStack[i].vertexViewCount(j) > 0)
                viewedByNone = false;
        }
        if(viewedByNone == true && _frame.vertexViewCount(j) > 0)
            newInformation = true;
    }

    //select frame in stack to replace
    int replaceFrameIndex = -1;
    if(newInformation == true)
    {
        std::cout << "frame has some new information " << std::endl;
        std::cout << "new good percentage: " << newGoodViewPercentage << std::endl;
        for(int i = 0; i < MAX_FRAMES; i++)
        {
            if(frameDataStack[i].init == true)
            {
                float lastGoodViewPercentage = goodVertexViewPercentage(frameDataStack[i]);
                std::cout << "last good percentage " << lastGoodViewPercentage << std::endl;
                if(newGoodViewPercentage > lastGoodViewPercentage)
                    replaceFrameIndex = i;
            }
            else
            {
                replaceFrameIndex = i;
                break;
            }
        }
    }
    else
    {
        std::cout << "frame does not add new information, skipping" << std::endl;
    }

    if(replaceFrameIndex >= 0)
    {
        std::cout << "add frame in index " << replaceFrameIndex << std::endl;
        //copy texture and generate its mipmaps
        for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            copyCPU(_frame.image, frameDataStack[replaceFrameIndex].image,lvl);
            copyGPU(_frame.image, frameDataStack[replaceFrameIndex].image, lvl);

            copyCPU(_frame.der, frameDataStack[replaceFrameIndex].der, lvl);
            copyGPU(_frame.der, frameDataStack[replaceFrameIndex].der, lvl);
        }

        frameDataStack[replaceFrameIndex].pose = _frame.pose;
        frameDataStack[replaceFrameIndex].init = true;
    }
    else
    {
        std::cout << "frame not added " << std::endl;
    }
}

float mesh_vo::errorMesh()
{
    float error = 0;
    for(int i = 0; i < int(scene_indices.size()); i+=3)
    {
        //std::cout << "triangle" << std::endl;
        float idepth[3];
        for(int j = 0; j < 3; j++)
        {
            int vertexIndex = scene_indices.at(i+j);
            idepth[j] = scene_vertices.at(vertexIndex*3+2);
        }
        float diff1 = idepth[0]-idepth[1];
        float diff2 = idepth[0]-idepth[2];
        float diff3 = idepth[1]-idepth[2];

        error += diff1*diff1 + diff2*diff2 + diff3*diff3;
    }
    return MESH_REGU*error/(VERTEX_HEIGH*VERTEX_WIDTH);
    /*
    float error = 0;
    for(int i = 0; i < int(scene_indices.size()); i+=3)
    {
        //std::cout << "triangle" << std::endl;
        Eigen::Vector3f vertex[3];
        for(int j = 0; j < 3; j++)
        {
            int vertexIndex = scene_indices.at(i+j);
            float rx = scene_vertices.at(vertexIndex*3+0);
            float ry = scene_vertices.at(vertexIndex*3+1);
            float rz = scene_vertices.at(vertexIndex*3+2);
            vertex[j] = Eigen::Vector3f(rx,ry,1.0)/rz;
            //std::cout << "vertex " << j << " " << vertex[j] << std::endl;
            if(vertex[j](0) != vertex[j](0) || vertex[j](1) != vertex[j](1) || vertex[j](2) != vertex[j](2))
            {
                std::cout << "some nand " << std::endl;
                std::cout << i << std::endl;
                std::cout << j << std::endl;
                std::cout << vertexIndex << std::endl;
                std::cout << rx << " " << ry << " " << rz << std::endl;
                std::cout << vertex[j] << std::endl;
            }
        }
        Eigen::Vector3f diff1 = vertex[0]-vertex[1];
        Eigen::Vector3f diff2 = vertex[0]-vertex[2];
        Eigen::Vector3f diff3 = vertex[1]-vertex[2];

        error += diff1.dot(diff1) + diff2.dot(diff2) + diff3.dot(diff3);
    }
    return 10.0*error/(VERTEX_HEIGH*VERTEX_WIDTH);
    */
}

void mesh_vo::HJMesh()
{
    for(int i = 0; i < int(scene_indices.size()); i+=3)
    {
        int vertexIndex[3];
        float idepth[3];
        float J[3];
        for(int j = 0; j < 3; j++)
        {
            vertexIndex[j] = scene_indices.at(i+j);
            idepth[j] = scene_vertices.at(vertexIndex[j]*3+2);
            J[j] = 1.0;
        }
        for(int j = 0; j < 3; j++)
        {
            float diff1;
            float diff2;
            if(j == 0)
            {
                diff1 = idepth[0] - idepth[1];
                diff2 = idepth[0] - idepth[2];
            }
            if(j == 1)
            {
                diff1 = idepth[1] - idepth[0];
                diff2 = idepth[1] - idepth[2];
            }
            if(j == 2)
            {
                diff1 = idepth[2] - idepth[0];
                diff2 = idepth[2] - idepth[1];
            }

            J_depth(vertexIndex[j]) += (MESH_REGU/(VERTEX_HEIGH*VERTEX_WIDTH))*(diff1+diff2)*J[j];
            J_joint(MAX_FRAMES*6+vertexIndex[j]) += (MESH_REGU/(VERTEX_HEIGH*VERTEX_WIDTH))*(diff1+diff2)*J[j];

            for(int k = 0; k < 3; k++)
            {
                H_depth.coeffRef(vertexIndex[j],vertexIndex[k]) += (MESH_REGU/(VERTEX_WIDTH*VERTEX_HEIGH))*2.0*J[j]*J[k];
                H_joint(MAX_FRAMES*6+vertexIndex[j],MAX_FRAMES*6+vertexIndex[k]) += (MESH_REGU/(VERTEX_WIDTH*VERTEX_HEIGH))*2.0*J[j]*J[k];
            }

        }
    }
    /*
    for(int i = 0; i < int(scene_indices.size()); i+=3)
    {
        int vertexIndex[3];
        Eigen::Vector3f vertex[3];
        Eigen::Vector3f J[3];
        for(int j = 0; j < 3; j++)
        {
            vertexIndex[j] = scene_indices.at(i+j);
            float rx = scene_vertices.at(vertexIndex[j]*3+0);
            float ry = scene_vertices.at(vertexIndex[j]*3+1);
            float rz = scene_vertices.at(vertexIndex[j]*3+2);
            vertex[j] = Eigen::Vector3f(rx,ry,1.0)/rz;
            J[j] = Eigen::Vector3f(-rx/(rz*rz),-ry*(rz*rz),-1.0/(rz*rz));
        }
        for(int j = 0; j < 3; j++)
        {
            Eigen::Vector3f diff1;
            Eigen::Vector3f diff2;
            if(j == 0)
            {
                diff1 = vertex[0] - vertex[1];
                diff2 = vertex[0] - vertex[2];
            }
            if(j == 1)
            {
                diff1 = vertex[1] - vertex[0];
                diff2 = vertex[1] - vertex[2];
            }
            if(j == 2)
            {
                diff1 = vertex[2] - vertex[0];
                diff2 = vertex[2] - vertex[1];
            }
            J_depth(vertexIndex[j]) += 0.0*(diff1+diff2).dot(J[j]);
            for(int k = 0; k < 3; k++)
                H_depth.coeffRef(vertexIndex[j],vertexIndex[k]) = 0.0*1.0*J[j].dot(J[k]);
        }
    }
    */
}

float mesh_vo::errorCPU(frame &_frame, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

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

            residual += error;
            num++;
        }

    if(num > 0)
        residual /= num;

    return residual;
}

float mesh_vo::errorStackCPU(int lvl)
{
    float error = 0.0;
    int count = 0;
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == true)
        {
            error += errorCPU(frameDataStack[i],lvl);
            count++;
        }
    }
    if(count > 0)
        error /= count;
    return error;
}

float mesh_vo::errorGPU(frame &_frame, int lvl)
{
    errorTextureGPU(_frame, lvl);
    return reduceErrorGPU(lvl);
}

float mesh_vo::errorStackGPU(int lvl)
{
    float error = 0.0;
    int count = 0;
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == true)
        {
            error += errorGPU(frameDataStack[i],lvl);
            count++;
        }
    }
    if(count > 0)
        error /= count;
    return error;
}

void mesh_vo::errorTextureGPU(frame &_frame, int lvl)
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

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
}

float mesh_vo::reduceErrorGPU(int lvl)
{
    //int new_lvl[MAX_LEVELS] = {5,6,7,8,9};
    //reduceFloat(errorTexture, lvl, new_lvl[lvl]);
    //int new_lvl[MAX_LEVELS] = {0,1,2,3,4};

    //errorData.generateMipmapsGPU(lvl);

    errorData.gpu_to_cpu(lvl);

    float error = 0.0;
    int count = 0;
    for(int x = 0; x < width[lvl]; x++)
        for(int y = 0; y < height[lvl]; y++)
        {
            float res = errorData.cpuTexture[lvl].at<float>(y,x);

            if(res < 0.0)
            {
                continue;
            }

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
    return error;
}

void mesh_vo::errorVertexGPU(frame &_frame, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    //return calcResidual_CPU(frame, framePose, lvl);

    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vertexIdData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, errorData.gpuTexture, lvl);

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

float mesh_vo::reduceErrorVertexGPU(int lvl)
{
    //int new_lvl[MAX_LEVELS] = {5,6,7,8,9};
    //reduceFloat(errorTexture, lvl, new_lvl[lvl]);
    //int new_lvl[MAX_LEVELS] = {0,1,2,3,4};

    //errorData.generateMipmapsGPU(lvl);

    errorData.gpu_to_cpu(lvl);
    vertexIdData.gpu_to_cpu(lvl);

    float error = 0.0;
    int count = 0;
    for(int x = 0; x < width[lvl]; x++)
        for(int y = 0; y < height[lvl]; y++)
        {
            float res = errorData.cpuTexture[lvl].at<float>(y,x);

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

void mesh_vo::optPose(frame &_frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    Sophus::SE3f bestPose = _frame.pose;

    for(int lvl=4; lvl >= 1; lvl--)
    {

        //float last_error = errorCPU(_frame,lvl);
        float last_error = errorGPU(_frame, lvl);

        //std::cout << "lvl " << lvl << " " << last_error << std::endl;

        for(int it = 0; it < maxIterations[lvl]; it++)
        {
            //HJPoseCPU(_frame ,lvl);
            //std::cout << "J_pose cpu: " << acc_J_pose << std::endl;
            HJPoseGPU(_frame, lvl);
            //std::cout << "J_pose gpu: " << acc_J_pose << std::endl;

            float lambda = 0.0;
            int n_try = 0;
            while(true)
            {
                Eigen::Matrix<float, 6, 6> acc_H_pose_lambda;
                acc_H_pose_lambda = H_pose;

                for(int j = 0; j < 6; j++)
                    acc_H_pose_lambda(j,j) *= 1.0 + lambda;

                inc_pose = acc_H_pose_lambda.ldlt().solve(J_pose);

                //std::cout << "acc_J_pose " << acc_J_pose << std::endl;

                //Sophus::SE3f new_pose = framePose.inverse()*Sophus::SE3f::exp(inc).inverse();
                //Sophus::SE3f new_pose = Sophus::SE3f::exp(inc) * framePose;
                //Sophus::SE3f new_pose = Sophus::SE3f::exp(inc).inverse() * framePose;
                //Sophus::SE3f new_pose = framePose*Sophus::SE3f::exp(inc);

                _frame.pose = bestPose*Sophus::SE3f::exp(inc_pose).inverse();

                //std::cout << "new_pose " << new_pose.matrix() << std::endl;

                //float error = errorCPU(_frame,lvl);
                float error = errorGPU(_frame, lvl);

                //std::cout << "new error " << error << std::endl;

                if(error < last_error)
                {
                    //accept update, decrease lambda
                    bestPose = _frame.pose;

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

void mesh_vo::optPoseMapJoint()
{
/*
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == false)
            return;
    }
*/
    tic_toc t;

    for(int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();

        Sophus::SE3f bestPoses[MAX_FRAMES];
        std::vector<float> best_vertices;

        for(int i = 0; i < MAX_FRAMES; i++)
            bestPoses[i] = frameDataStack[i].pose;
        best_vertices = scene_vertices;

        float last_error = errorStackGPU(lvl) + errorMesh();

        std::cout << "initial error time " << t.toc() << std::endl;
        std::cout << "lvl " << lvl << " initial error " << last_error << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;

        for(int it = 0; it < maxIterations; it++)
        {
            t.tic();

            HJPoseMapStackGPU(lvl);
            HJMesh();

            std::cout << "HJ time " << t.toc() << std::endl;

            int n_try = 0;
            while(true)
            {
                t.tic();

                Eigen::MatrixXf H_joint_lambda = H_joint;

                for(int j = 0; j < H_joint_lambda.rows(); j++)
                {
                    H_joint_lambda(j,j) *= (1.0+lambda);
                }

                //inc_joint = H_joint_lambda.ldlt().solve(J_joint);
                //inc_joint = H_joint_lambda.colPivHouseholderQr().solve(J_joint);


                for(int j = 0; j < int(J_joint.size()); j++)
                {
                    float h = H_joint_lambda(j,j);
                    if(h > 0.0 && abs(J_joint(j)) > 0.0)
                    //if(J_joint(j) > 0.0)
                    {
                        inc_joint(j) = J_joint(j)/h;
                        //inc_joint(j) = (1.0/(1.0+lambda))*J_joint(j)/fabs(J_joint(j));
                        //std::cout << "update" << std::endl;
                    }
                }

                std::cout << "solve time " << t.toc() << std::endl;

                t.tic();

                for(int i = 0; i < MAX_FRAMES; i++)
                {
                    if(frameDataStack[i].init == true)
                    {
                        Eigen::VectorXf inc_pose(6);
                        bool good = true;
                        for(int j = 0; j < 6; j++)
                        {
                            if(std::isnan(inc_joint(i*6+j)) || std::isinf(inc_joint(i*6+j)))
                            {
                                std::cout << "nand in inc_joint pose part" << std::endl;
                                good = false;
                            }
                            inc_pose(j) = inc_joint(i*6+j);
                        }
                        if(good == false)
                        {
                            continue;
                        }
                        frameDataStack[i].pose = bestPoses[i]*Sophus::SE3f::exp(inc_pose).inverse();
                    }
                }

                for(int index=0; index < VERTEX_HEIGH*VERTEX_WIDTH; index++)
                {
                    if(std::isnan(inc_joint(MAX_FRAMES*6+index)))
                    {
                        std::cout << "nand in inc_joint depth part " << std::endl;
                        continue;
                    }
                    scene_vertices[index*3+2] = best_vertices[index*3+2] - inc_joint(MAX_FRAMES*6+index);
                    if(scene_vertices[index*3+2] < min_idepth || scene_vertices[index*3+2] > max_idepth)
                        scene_vertices[index*3+2] = best_vertices[index*3+2];
                }

                glBindVertexArray(scene_VAO);
                glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

                std::cout << "set data time " << t.toc() << std::endl;

                t.tic();

                float error = errorStackGPU(lvl) + errorMesh();

                std::cout << "new error time " << t.toc() << std::endl;
                std::cout << "lvl " << lvl << " new error " << error << std::endl;

                if(error < last_error)
                {
                    //accept update, decrease lambda
                    std::cout << "update accepted " << std::endl;

                    for(int i = 0; i < MAX_FRAMES; i++)
                        bestPoses[i] = frameDataStack[i].pose;
                    best_vertices = scene_vertices;

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

                    for(int i = 0; i < MAX_FRAMES; i++)
                        frameDataStack[i].pose = bestPoses[i];
                    scene_vertices = best_vertices;

                    glBindVertexArray(scene_VAO);
                    glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
                    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

                    //reject update, increase lambda, use un-updated data
                    std::cout << "update rejected " << lambda << " " << inc_joint.dot(inc_joint) << std::endl;

                    if(inc_joint.dot(inc_joint) < 1e-16)
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

void mesh_vo::optMapJoint()
{
    /*
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == false)
            return;
    }
    */

    tic_toc t;
    for(int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();
        std::vector<float> best_vertices;
        best_vertices = scene_vertices;

        float last_error = errorStackGPU(lvl) + errorMesh();

        std::cout << "init error time " << t.toc() << std::endl;
        std::cout << "lvl " << lvl << " initial error " << last_error << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;
        for(int it = 0; it < maxIterations; it++)
        {
            t.tic();

            HJMapStackGPU(lvl);
            HJMesh();

            std::cout << "HJ time " << t.toc() << std::endl;

            int n_try = 0;
            while(true)
            {
                Eigen::SparseMatrix<float> H_depth_lambda = H_depth;

                for(int j = 0; j < H_depth_lambda.rows(); j++)
                {
                    H_depth_lambda.coeffRef(j,j) *= (1.0+lambda);
                }

                t.tic();

                H_depth_lambda.makeCompressed();
                Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
                //Eigen::SparseLU<Eigen::SparseMatrix<float> > solver;
                //Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
                //Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;
                solver.analyzePattern(H_depth_lambda);
                //std::cout << solver.info() << std::endl;
                solver.factorize(H_depth_lambda);
                //std::cout << solver.lastErrorMessage() << std::endl;
                inc_depth = -solver.solve(J_depth);
                //inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                //inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                //inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

/*
                for(int j = 0; j < int(J_depth.size()); j++)
                {
                    float h = acc_H_depth_lambda.coeffRef(j,j);
                    if(h > 0.0)
                    //if(abs(J_depth(j)) > 0.0)
                    {
                        inc_depth(j) = -J_depth(j)/h;
                        //inc_depth(j) = -(1.0/(1.0+lambda))*J_depth(j)/fabs(J_depth(j));
                        //std::cout << "update" << std::endl;
                    }
                }
*/
                std::cout << "solve time " << t.toc() << std::endl;

                t.tic();

                for(int index=0; index < VERTEX_HEIGH*VERTEX_WIDTH; index++)
                {
                    if(inc_depth(index) != inc_depth(index))
                    {
                        std::cout << "some nand in inc_depth " << std::endl;
                        continue;
                    }
                    scene_vertices[index*3+2] = best_vertices[index*3+2] + inc_depth(index);
                    if(scene_vertices[index*3+2] < min_idepth || scene_vertices[index*3+2] > max_idepth)
                        scene_vertices[index*3+2] = best_vertices[index*3+2];
                }

                glBindVertexArray(scene_VAO);
                glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

                std::cout << "set data time " << t.toc() << std::endl;

                t.tic();

                float error = errorStackGPU(lvl) + errorMesh();

                std::cout << "new error time " << t.toc() << std::endl;

                std::cout << "lvl " << lvl << " new error " << error << std::endl;

                if(error < last_error)
                {
                    //accept update, decrease lambda
                    std::cout << "update accepted " << std::endl;

                    best_vertices = scene_vertices;

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

                    scene_vertices = best_vertices;

                    glBindVertexArray(scene_VAO);
                    glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
                    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

                    //reject update, increase lambda, use un-updated data
                    std::cout << "update rejected " << std::endl;


                    if(inc_depth.dot(inc_depth) < 1e-8)
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

/*
void mesh_vo::optMapVertex()
{

    for(int lvl = 0; lvl >= 0; lvl--)
    {
        error_vertex.setZero();
        count_vertex.setZero();
        float last_error = 0.0;
        for(int i = 0; i < MAX_FRAMES; i++)
        {
            //last_error += calcErrorCPU(frameDataStack[i], lvl);
            if(frameDataStack[i].init == true)
                last_error += errorGPU(frameDataStack[i], lvl);
        }
        Eigen::VectorXf last_error_individual = Eigen::VectorXf(vwidth*vheight);
        last_error_individual = error_vertex;

        std::cout << "lvl " << lvl << " initial error " << last_error << std::endl;

        //for(int i = 0; i < vwidth*vheight; i++)
        //    std::cout << i << " error " << error_depth(i) << std::endl;

        Eigen::VectorXf lambda_depth = Eigen::VectorXf(vwidth*vheight);
        lambda_depth.setZero();
        for(int it = 0; it < 10; it++)
        //while(true)
        {
            acc_H_depth.setZero();
            acc_J_depth.setZero();
            inc_depth.setZero();
            count_vertex.setZero();

            for(int i = 0; i < MAX_FRAMES; i++)
            {
                if(frameDataStack[i].init == true)
                    jacobianMapTextureGPU(frameDataStack[i], lvl);
            }

            showGPU(primitiveIdData,lvl);
            cv::waitKey(100.0);

            int n_try = 0;
            //while(true)
            {
                Eigen::SparseMatrix<float> acc_H_depth_lambda = acc_H_depth;

                for(int j = 0; j < acc_H_depth_lambda.rows(); j++)
                {
                    acc_H_depth_lambda.coeffRef(j,j) *= (1.0+lambda_depth(j));
                }

                for(int j = 0; j < int(acc_J_depth.size()); j++)
                {
                    float h = acc_H_depth_lambda.coeffRef(j,j);
                    if(h > 0.0)
                    {
                        inc_depth(j) = -acc_J_depth(j)/h;
                    }
                }

                for(int index=0; index < int(scene_vertices.size()); index++)
                {
                    scene_vertices_updated[index] = scene_vertices[index];// + inc_depth(index);
                }

                for(int index=0; index < int(inc_depth.size()); index++)
                {
                    if(inc_depth(index)!=inc_depth(index))
                    {
                        std::cout << "some nand in inc_depth" << std::endl;
                    }

                    scene_vertices_updated[index*3+2] = scene_vertices[index*3+2] + inc_depth(index);

                    if(scene_vertices_updated[index*3+2] < min_idepth)
                        scene_vertices_updated[index*3+2] = scene_vertices[index*3+2];
                    if(scene_vertices_updated[index*3+2] > max_idepth)
                        scene_vertices_updated[index*3+2] = scene_vertices[index*3+2];
                }

                glBindVertexArray(scene_VAO);
                glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices_updated.size(), scene_vertices_updated.data());

                error_vertex.setZero();
                count_vertex.setZero();
                float error = 0.0;
                for(int i=0; i < MAX_FRAMES; i++)
                {
                    //error += calcErrorCPU(frameDataStack[i], lvl);
                    if(frameDataStack[i].init == true)
                        error += errorGPU(frameDataStack[i],lvl);
                }

                //std::cout << "lvl " << lvl << " new error " << error << std::endl;

                for(int vid = 0; vid < vwidth*vheight; vid++)
                {
                    //std::cout << vid << " last error " << last_error_individual(vid) << std::endl;
                    //std::cout << "new error " << error_depth(vid) << std::endl;
                    if(error_vertex(vid) < last_error_individual(vid))
                    {
                        //accept update, decrease lambda
                        //std::cout << vid << " update accepted " << std::endl;

                        scene_vertices[vid*3+0] = scene_vertices_updated[vid*3+0];
                        scene_vertices[vid*3+1] = scene_vertices_updated[vid*3+1];
                        scene_vertices[vid*3+2] = scene_vertices_updated[vid*3+2];

                        float p = error_vertex(vid) / last_error_individual(vid);

                        if(lambda_depth(vid) < 0.2f)
                            lambda_depth(vid) = 0.0f;
                        else
                            lambda_depth(vid) *= 0.5;

                        last_error = error;

                        if( p > 0.999f)
                        {
                            //std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
                            //if converged, do next level
                            //it = maxIterations;
                        }

                        //if update accepted, do next iteration
                        continue;
                    }
                    else
                    {
                        n_try++;

                        if(lambda_depth(vid) < 0.2f)
                            lambda_depth(vid) = 0.2f;
                        else
                            lambda_depth(vid) *= 2.0;// std::pow(2.0, n_try);

                        //calcIdepthGPU(keyframeData,0);
                        //keyframeData.idepth.gpu_to_cpu(0);
                        //keyframeData.idepth.generateMipmapsCPU(0);

                        //reject update, increase lambda, use un-updated data
                        //std::cout << vid << " update rejected " << lambda_depth(vid) << std::endl;

                        if(inc_depth(vid)*inc_depth(vid) < 1e-16)
                        {
                            //std::cout << "lvl " << lvl << " inc size too small, after " << " itarations with lambda " << lambda << std::endl;
                            //if too small, do next level!
                            //it = maxIterations;
                            //continue;
                        }
                        continue;
                    }
                }

                glBindVertexArray(scene_VAO);
                glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

            }
        }
    }
}
*/

void mesh_vo::HJPoseCPU(frame &_frame, int lvl)
{
    J_pose.setZero();
    H_pose.setZero();

    Sophus::SE3f relativePose = _frame.pose*keyframeData.pose.inverse();
    int count = 0;
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

            Eigen::Matrix<float, 6, 1> J;
            J << d_I_d_tra(0), d_I_d_tra(1), d_I_d_tra(2), d_I_d_rot(0), d_I_d_rot(1), d_I_d_rot(2);

            J_pose += J*residual;
            count++;
            for(int i = 0; i < 6; i++)
            {
                for(int j = 0; j < 6; j++)
                {
                    H_pose(i,j) += J[i]*J[j];
                }
            }
        }

    if(count > 0)
    {
        J_pose /= count;
        H_pose /= count;
    }
}

void mesh_vo::HJPoseGPU(frame &_frame, int lvl)
{
    H_pose.setZero();
    J_pose.setZero();
    jacobianPoseTextureGPU(_frame, lvl);
    reduceHJPoseGPU(lvl);
}

void mesh_vo::jacobianPoseTextureGPU(frame _frame, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, errorData.gpuTexture, lvl);
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
    jacobianPoseShader.use();

    jacobianPoseShader.setMat4("framePose", eigen2glm_mat4((_frame.pose*keyframeData.pose.inverse()).matrix()));
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

void mesh_vo::reduceHJPoseGPU(int lvl)
{
    errorData.gpu_to_cpu(lvl);
    traData.gpu_to_cpu(lvl);
    rotData.gpu_to_cpu(lvl);

    Eigen::Matrix<float, 6, 1> J_pose_n;
    Eigen::Matrix<float, 6, 6> H_pose_n;
    J_pose_n.setZero();
    H_pose_n.setZero();

    int count = 0;
    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            cv::Vec3f d_I_d_tra = traData.cpuTexture[lvl].at<cv::Vec3f>(y,x);
            cv::Vec3f d_I_d_rot = rotData.cpuTexture[lvl].at<cv::Vec3f>(y,x);
            float residual = errorData.cpuTexture[lvl].at<float>(y,x);

            if(residual == 0.0)
                continue;

            Eigen::Matrix<float, 6, 1> J;
            J << d_I_d_tra[0], d_I_d_tra[1], d_I_d_tra[2], d_I_d_rot[0], d_I_d_rot[1], d_I_d_rot[2];

            J_pose_n += J*residual;
            count++;
            for(int i = 0; i < 6; i++)
            {
                for(int j = 0; j < 6; j++)
                {
                    H_pose_n(i,j) += J[i]*J[j];
                }
            }
        }
    if(count > 0)
    {
        J_pose += J_pose_n/count;
        H_pose += H_pose_n/count;
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
            reduceHJMapGPU(lvl);
            count++;

            //showGPU(debugData,lvl);
            //cv::waitKey(1000.0);
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
    reduceHJMapGPU(lvl);
}

void mesh_vo::jacobianMapTextureGPU(frame &_frame, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vertexIdData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, errorData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, d_I_d_p0Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, d_I_d_p1Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, d_I_d_p2Data.gpuTexture, lvl);
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

void mesh_vo::reduceHJMapGPU(int lvl)
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
    errorData.gpu_to_cpu(lvl);
    d_I_d_p0Data.gpu_to_cpu(lvl);
    d_I_d_p1Data.gpu_to_cpu(lvl);
    d_I_d_p2Data.gpu_to_cpu(lvl);

    int count = 0;
    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            //std::cout << "pixel index " << index << std::endl;
            //acumular para cada pixel
            float error = errorData.cpuTexture[lvl].at<float>(y,x);

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
            J[0] = d_I_d_p0Data.cpuTexture[lvl].at<float>(y,x);
            J[1] = d_I_d_p1Data.cpuTexture[lvl].at<float>(y,x);
            J[2] = d_I_d_p2Data.cpuTexture[lvl].at<float>(y,x);

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

                for(int j = 0; j < 3; j++)
                {
                    //acc_H_depth(vertexID[i],vertexID[j]) += J[i]*J[j];
                    H_depth.coeffRef(vertexID[i],vertexID[j]) += J[i]*J[j];
                }
            }
        }
    if(count > 0)
    {
        H_depth /= count;
        J_depth /= count;
    }
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

            //showGPU(debugData,lvl);
            //cv::waitKey(100.0);
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

void mesh_vo::jacobianPoseMapTextureGPU(frame &_frame, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vertexIdData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, errorData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, traData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, rotData.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, d_I_d_p0Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, d_I_d_p1Data.gpuTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, d_I_d_p2Data.gpuTexture, lvl);
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
    errorData.gpu_to_cpu(lvl);
    traData.gpu_to_cpu(lvl);
    rotData.gpu_to_cpu(lvl);
    d_I_d_p0Data.gpu_to_cpu(lvl);
    d_I_d_p1Data.gpu_to_cpu(lvl);
    d_I_d_p2Data.gpu_to_cpu(lvl);

    int count = 0;
    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            cv::Vec3f d_I_d_tra = traData.cpuTexture[lvl].at<cv::Vec3f>(y,x);
            cv::Vec3f d_I_d_rot = rotData.cpuTexture[lvl].at<cv::Vec3f>(y,x);
            float residual = errorData.cpuTexture[lvl].at<float>(y,x);

            float J_depth[3];
            J_depth[0] = d_I_d_p0Data.cpuTexture[lvl].at<float>(y,x);
            J_depth[1] = d_I_d_p1Data.cpuTexture[lvl].at<float>(y,x);
            J_depth[2] = d_I_d_p2Data.cpuTexture[lvl].at<float>(y,x);

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

                for(int j = 0; j < 9; j++)
                {
                    H_joint(paramID[i],paramID[j]) += J[i]*J[j];
                }
            }
        }
    if(count > 0)
    {
        H_joint /= count;
        J_joint /= count;
    }
}

void mesh_vo::visual_odometry(cv::Mat _frame)
{
    glfwMakeContextCurrent(frameWindow);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    Sophus::SE3f viewPose;
    viewPose.translation() = Eigen::Vector3f(0.0,-1.0,0.0);
    view3DTexture(viewPose*frameData.pose, 1);
    showGPU(view3DData,1);

    //calcIdepthGPU(keyframeData,1);
    //keyframeData.idepth.gpu_to_cpu(1);
    //showGPU(keyframeData.idepth,1);

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
    optPose(frameData);//*Sophus::SE3f::exp(inc_pose).inverse());
    glFinish();
    std::cout << "estimated pose " << std::endl;
    std::cout << frameData.pose.matrix() << std::endl;
    std::cout << "clacPose time " << t.toc() << std::endl;

    float occupancy = calcOccupancyCPU(frameData, 1);

    std::cout << "occ " << occupancy << std::endl;

    //showTexture(keyframeIdepthTexture, 0);
    //cv::imshow("idepth", keyframeIdepthMat[0]);
    //cv::waitKey(30);
    //return;

    if(occupancy < 0.8)
    {
        changeKeyframe(frameData,1,0.8);
        optPoseMapJoint();
        //optMapVertex();
        return;
    }

    {
        //std::cout << "sup diff " << diff << " add frame and update map" << std::endl;
        addFrameToStack(frameData);
        optPoseMapJoint();
        //optMapVertex();
    }
}

void mesh_vo::localization(cv::Mat _frame)
{
    glfwMakeContextCurrent(frameWindow);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    //calcIdepthGPU(keyframeData,1);
    //showGPU(keyframeData.idepth,1);

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
    optPose(frameData);//*Sophus::SE3f::exp(inc_pose).inverse());
    glFinish();
    std::cout << "estimated pose " << std::endl;
    std::cout << frameData.pose.matrix() << std::endl;
    std::cout << "clacPose time " << t.toc() << std::endl;
}

void mesh_vo::mapping(cv::Mat _frame, Sophus::SE3f _globalPose)
{
    tic_toc t;

    glfwMakeContextCurrent(frameWindow);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

/*
    Sophus::SE3f viewPose;
    Eigen::Matrix<float, 6, 1> viewPoseExp;
    viewPoseExp << 0.0, 2.0, 2.5, 0.5, 0.0, 0.0;
    view3DTexture(Sophus::SE3f::exp(viewPoseExp)*keyframeData.pose, 1);
    showGPU(view3DData,1);
*/

    calcIdepthGPU(keyframeData,1);
    keyframeData.idepth.gpu_to_cpu(1);
    showGPU(keyframeData.idepth,1);

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

    t.tic();
    float occ = calcOccupancyCPU(frameData,1);
    glFinish();
    std::cout << "occupancy time " << t.toc() << std::endl;

    if(occ < 0.8)
    {
        t.tic();
        changeKeyframe(frameData,1,0.8);
        //optMapVertex();
        optMapJoint();
        //optPoseMapJoint();

        glFinish();
        std::cout << "change keyframe time " << t.toc() << std::endl;

        return;
    }

    t.tic();
    addFrameToStack(frameData);
    glFinish();
    std::cout << "add frame stack time " << t.toc() << std::endl;

    t.tic();
    //optMapVertex();
    optMapJoint();
    //optPoseMapJoint();
    glFinish();
    std::cout << "update map time " << t.toc() << std::endl;

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

float mesh_vo::goodVertexViewPercentage(frame &_frame)
{
    int count = 0;
    for(int i = 0; i < VERTEX_HEIGH*VERTEX_WIDTH; i++)
    {
        if(_frame.vertexViewCount(i) > 0)
            count++;
    }
    return float(count)/(VERTEX_HEIGH*VERTEX_WIDTH);
}

void mesh_vo::vertexViewCountGPU(frame &_frame, int lvl)
{
    _frame.vertexViewCount.setZero();
    vertexViewCountTextureGPU(_frame, lvl);
    reduceVertexViewCountGPU(_frame,lvl);
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

void mesh_vo::reduceVertexViewCountGPU(frame &_frame,int lvl)
{
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






