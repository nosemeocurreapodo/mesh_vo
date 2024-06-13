#include "mesh_vo.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>
#include <Eigen/IterativeLinearSolvers>

#include <map>



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
    lastframeData = frame(height[0], width[0]);
    for(int i = 0; i < MAX_FRAMES; i++)
        frameDataStack[i] = frame(height[0], width[0]);

    vertexIdData = data(height[0], width[0], 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    debugData = data(height[0], width[0], 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    view3DData = data(height[0], width[0], 4, GL_UNSIGNED_BYTE, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);



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

        calcIdepthGPU(keyframeData, lvl);
        keyframeData.idepth.gpu_to_cpu(lvl);
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
                //if(((x % 2 == 0) && (y % 2 == 0)) || ((x % 2 != 0) && (y % 2 != 0)))
                if(rand() > 0.5*RAND_MAX)
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

    float frameError = errorGPU(&_frame, 1);
    float minError = 100000000000.0;
    int minErrorIndex = -1;
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == true)
        {
            float error = errorGPU(&frameDataStack[i],1);
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


    if(minErrorIndex >= 0)// && frameError > minErrorIndex)
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


    /*
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

    */
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
        for(int j = 0; j < 3; j++)
        {
            vertexIndex[j] = scene_indices.at(i+j);
            idepth[j] = scene_vertices.at(vertexIndex[j]*3+2);
        }
        float diff1 = idepth[0]-idepth[1];
        float diff2 = idepth[0]-idepth[2];
        float diff3 = idepth[1]-idepth[2];

        float J1[3] = {1.0,-1.0,0.0};
        float J2[3] = {1.0,0.0,-1.0};
        float J3[3] = {0.0,1.0,-1.0};

        for(int j = 0; j < 3; j++)
        {
            J_depth(vertexIndex[j]) += (MESH_REGU/(VERTEX_HEIGH*VERTEX_WIDTH))*(diff1*J1[j]+diff2*J2[j]+diff3*J3[j]);
            J_joint(MAX_FRAMES*6+vertexIndex[j]) += (MESH_REGU/(VERTEX_HEIGH*VERTEX_WIDTH))*(diff1*J1[j]+diff2*J2[j]+diff3*J3[j]);

            for(int k = 0; k < 3; k++)
            {
                H_depth.coeffRef(vertexIndex[j],vertexIndex[k]) += (MESH_REGU/(VERTEX_WIDTH*VERTEX_HEIGH))*(J1[j]*J1[k]+J2[j]*J2[k]+J3[j]*J3[k]);
                H_joint(MAX_FRAMES*6+vertexIndex[j],MAX_FRAMES*6+vertexIndex[k]) += (MESH_REGU/(VERTEX_WIDTH*VERTEX_HEIGH))*(J1[j]*J1[k]+J2[j]*J2[k]+J3[j]*J3[k]);
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


float mesh_vo::errorStackCPU(int lvl)
{
    float error = 0.0;
    int count = 0;
    for(int i = 0; i < MAX_FRAMES; i++)
    {
        if(frameDataStack[i].init == true)
        {
            //frameData = frameDataStack[i];
            //error += errorCPU(lvl);
            count++;
        }
    }
    if(count > 0)
        error /= count;
    return error;
}

float mesh_vo::errorGPU(frame *_frame, int lvl)
{
    errorTextureGPU(_frame, lvl);
    //float error =  reduceErrorGPU(lvl);
    float error = reduceErrorComputeGPU(_frame, lvl);
    return error;
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

void mesh_vo::optPose(frame &_frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    Sophus::SE3f bestPose = _frame.pose;

    tic_toc t;

    for(int lvl=4; lvl >= 1; lvl--)
    {
        std::cout << "*************************lvl " << lvl << std::endl;
        t.tic();
        float last_error = 0.0;//errorCPU(lvl);
        //float last_error = errorGPU(_frame, lvl);
        //std::cout << "init error " << last_error << " time " << t.toc() << std::endl;

        for(int it = 0; it < maxIterations[lvl]; it++)
        {
            t.tic();
            HJPose _hjpose = HJPoseCPU(&_frame, lvl);
            //std::cout << "J_pose cpu: " << acc_J_pose << std::endl;
            //HJPoseGPU(_frame, lvl);
            //std::cout << "J_pose gpu: " << acc_J_pose << std::endl;
            std::cout << "HJPose time " << t.toc() << std::endl;

            float lambda = 0.0;
            int n_try = 0;
            while(true)
            {
                Eigen::Matrix<float, 6, 6> acc_H_pose_lambda;
                acc_H_pose_lambda = _hjpose.H_pose;

                for(int j = 0; j < 6; j++)
                    acc_H_pose_lambda(j,j) *= 1.0 + lambda;

                Eigen::Matrix<float, 6, 1> inc_pose = acc_H_pose_lambda.ldlt().solve(_hjpose.J_pose);

                //std::cout << "acc_J_pose " << acc_J_pose << std::endl;

                //Sophus::SE3f new_pose = framePose.inverse()*Sophus::SE3f::exp(inc).inverse();
                //Sophus::SE3f new_pose = Sophus::SE3f::exp(inc) * framePose;
                //Sophus::SE3f new_pose = Sophus::SE3f::exp(inc).inverse() * framePose;
                //Sophus::SE3f new_pose = framePose*Sophus::SE3f::exp(inc);

                _frame.pose = ((bestPose*keyframeData.pose.inverse())*Sophus::SE3f::exp(inc_pose).inverse())*keyframeData.pose;
                //_frame.pose = bestPose*Sophus::SE3f::exp(inc_pose).inverse();

                //std::cout << "new_pose " << new_pose.matrix() << std::endl;

                t.tic();
                float error = 0.0;//errorCPU(lvl);
                //float error = errorGPU(_frame, lvl);
                std::cout << "new error time " << t.toc() << std::endl;

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


void mesh_vo::optPose2(frame &_frame)
{
    addFrameToStack(_frame);

    int tlvl = 0;

    tic_toc t;
    t.tic();
    HJPose CPUHJPose;
    for(int i = 0; i < 100; i++)
    {
        //for(int index = 0; index < MAX_FRAMES; index++)
        {
            //if(!frameDataStack[index].init)
            //    continue;

            CPUHJPose =  HJPoseCPU(&_frame, tlvl);
            //CPUHJPose.error = errorCPU(&_frame, tlvl);
        }
    }
    std::cout << "cpu error: " << CPUHJPose.error << " cpu time " << t.toc() << std::endl;

    t.tic();
    HJPose GPUHJPose;
    for(int i = 0; i < 100; i++)
    {
        //for(int index = 0; index < MAX_FRAMES; index++)
        {
            //if(!frameDataStack[index].init)
            //    continue;

            //GPUHJPose.error = errorGPU(&_frame,tlvl);
            GPUHJPose = HJPoseGPU(&_frame, tlvl);
        }
    }
    std::cout << "gpu error " << GPUHJPose.error << " gpu time " << t.toc() << std::endl;

    t.tic();
    HJPose GPUHJPose2;
    for(int i = 0; i < 100; i++)
    {
        //GPUHJPose2.error = errorGPU_v2(&_frame, tlvl);
        GPUHJPose2 = HJPoseGPU_v3(&_frame, tlvl);

    }
    std::cout << "gpu error2 " << GPUHJPose2.error << " gpu time " << t.toc() << std::endl;

    showCPU(_frame.error, tlvl);
    showGPU(_frame.error,tlvl);

    return;

    t.tic();

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeData.image.gpuTexture);

    glBindVertexArray(scene_VAO);

    glViewport(0,0,width[tlvl],height[tlvl]);

    errorShader.setMat4("opencv2opengl", opencv2opengl);
    errorShader.setMat4("projection", projMat[tlvl]);
    errorShader.setFloat("fx", fx[tlvl]);
    errorShader.setFloat("fy", fy[tlvl]);
    errorShader.setFloat("cx", cx[tlvl]);
    errorShader.setFloat("cy", cy[tlvl]);
    errorShader.setFloat("fxinv", fxinv[tlvl]);
    errorShader.setFloat("fyinv", fyinv[tlvl]);
    errorShader.setFloat("cxinv", cxinv[tlvl]);
    errorShader.setFloat("cyinv", cyinv[tlvl]);
    errorShader.setFloat("dx", dx[tlvl]);
    errorShader.setFloat("dy", dy[tlvl]);

    float gpuError2 = 0;
    for(int i = 0; i < 100; i++)
    {
        errorShader.use();
        for(int index = 0; index < MAX_FRAMES; index++)
        {
            if(!frameDataStack[index].init)
                continue;

            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, frameDataStack[index].error.gpuTexture, tlvl);
            unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
            glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, frameDataStack[index].image.gpuTexture);
            errorShader.setMat4("framePose", eigen2glm_mat4((frameDataStack[index].pose*keyframeData.pose.inverse()).matrix()));
            glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);
            //gpuError2 = reduceErrorGPU(0);
        }

        int srclvl = tlvl;
        int dstlvl = tlvl + 3;

        reduceErrorShader.use();
        for(int index = 0; index < MAX_FRAMES; index++)
        {
            if(!frameDataStack[index].init)
                continue;

            glActiveTexture(GL_TEXTURE0);
            glBindImageTexture( 0, frameDataStack[index].error.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_R32F );

            glActiveTexture(GL_TEXTURE1);
            glBindImageTexture( 1, frameDataStack[index].error.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );

            glActiveTexture(GL_TEXTURE2);
            glBindImageTexture( 2, frameDataStack[index].count.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );

            glDispatchCompute(width[dstlvl], height[dstlvl], 1 );
            glMemoryBarrier( GL_ALL_BARRIER_BITS );
        }

        gpuError2 = 0.0;
        for(int index = 0; index < MAX_FRAMES; index++)
        {
            gpuError2 += reduceErrorGPU(&frameDataStack[index], dstlvl);
        }
    }
    std::cout << "gpu2 error " << gpuError2 << " gpu2 time " << t.toc() << std::endl;



    t.tic();

    float gpuError3 = 0.0;

    int srclvl = tlvl;
    int dstlvl = tlvl + 3;
    for(int i = 0; i < 100; i++)
    {
        for(int index = 0; index < MAX_FRAMES; index++)
        {
            if(!frameDataStack[index].init)
                continue;

            glActiveTexture(GL_TEXTURE0);
            glBindImageTexture( 0, keyframeData.image.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_R8UI);

            glActiveTexture(GL_TEXTURE1);
            glBindImageTexture( 1, frameDataStack[index].image.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_R8UI);

            glActiveTexture(GL_TEXTURE2);
            glBindImageTexture( 2, keyframeData.idepth.gpuTexture, srclvl, GL_FALSE, 0, GL_READ_ONLY, GL_R32F );

            glActiveTexture(GL_TEXTURE3);
            glBindImageTexture( 3, frameDataStack[index].error.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );

            glActiveTexture(GL_TEXTURE4);
            glBindImageTexture( 4, frameDataStack[index].count.gpuTexture, dstlvl, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );

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
            computeErrorAndReduceCShader.setMat4("framePose", eigen2glm_mat4((frameDataStack[index].pose*keyframeData.pose.inverse()).matrix()));

            glDispatchCompute(width[dstlvl], height[dstlvl], 1 );
            glMemoryBarrier( GL_ALL_BARRIER_BITS );
        }

        gpuError3 = 0.0;
        for(int index = 0; index < MAX_FRAMES; index++)
        {
            if(!frameDataStack[index].init)
                continue;
            gpuError3 += reduceErrorGPU(&frameDataStack[index], dstlvl);
        }
    }
    //gpuError3 = reduceErrorGPU(dstlvl, true);
    std::cout << "gpu3 error " << gpuError3 << " gpu3 time " << t.toc() << std::endl;

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

    for(int lvl = 0; lvl >= 0; lvl--)
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
                        frameDataStack[i].pose = ((bestPoses[i]*keyframeData.pose.inverse())*Sophus::SE3f::exp(inc_pose).inverse())*keyframeData.pose;
                        //frameDataStack[i].pose = bestPoses[i]*Sophus::SE3f::exp(inc_pose).inverse();
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
                //Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
                Eigen::SparseLU<Eigen::SparseMatrix<float> > solver;
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







void mesh_vo::visual_odometry(cv::Mat _frame)
{
    glfwMakeContextCurrent(frameWindow);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    Sophus::SE3f viewPose;
    viewPose.translation() = Eigen::Vector3f(0.0,-1.0,0.0);
    view3DTexture(viewPose*lastframeData.pose, 1);
    showGPU(view3DData,1);

    //calcIdepthGPU(keyframeData,1);
    //keyframeData.idepth.gpu_to_cpu(1);
    //showGPU(keyframeData.idepth,1);

    //save frame in gpu memory, calc derivavites y mipmaps
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_frame,lastframeData.image.cpuTexture[lvl],cv::Size(width[lvl], height[lvl]), cv::INTER_LANCZOS4);
        lastframeData.image.cpu_to_gpu(lvl);

        calcDerivativeCPU(lastframeData,lvl);
        lastframeData.der.cpu_to_gpu(lvl);
    }

    tic_toc t;
    t.tic();
    //frameData.pose = _globalPose*keyframeData.pose.inverse();
    optPose(lastframeData);//*Sophus::SE3f::exp(inc_pose).inverse());
    glFinish();
    std::cout << "estimated pose " << std::endl;
    std::cout << lastframeData.pose.matrix() << std::endl;
    std::cout << "clacPose time " << t.toc() << std::endl;

    float occupancy = calcOccupancyCPU(lastframeData, 1);

    std::cout << "occ " << occupancy << std::endl;

    //showTexture(keyframeIdepthTexture, 0);
    //cv::imshow("idepth", keyframeIdepthMat[0]);
    //cv::waitKey(30);
    //return;

    if(occupancy < 0.8)
    {
        changeKeyframe(lastframeData,1,0.8);
        optPoseMapJoint();
        //optMapVertex();
        return;
    }

    {
        //std::cout << "sup diff " << diff << " add frame and update map" << std::endl;
        addFrameToStack(lastframeData);
        optPoseMapJoint();
        //optMapVertex();
    }
}

void mesh_vo::localization(cv::Mat _frame)
{
    //calcIdepthGPU(keyframeData,1);
    //showGPU(keyframeData.idepth,1);

    //save frame in gpu memory, calc derivavites y mipmaps
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_frame,lastframeData.image.cpuTexture[lvl],cv::Size(width[lvl], height[lvl]), cv::INTER_LANCZOS4);
        lastframeData.image.cpu_to_gpu(lvl);

        calcDerivativeCPU(lastframeData,lvl);
        lastframeData.der.cpu_to_gpu(lvl);
    }

    //frameData.pose = _globalPose*keyframeData.pose.inverse();
    optPose2(lastframeData);//*Sophus::SE3f::exp(inc_pose).inverse());
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
        cv::resize(_frame,lastframeData.image.cpuTexture[lvl],cv::Size(width[lvl], height[lvl]), cv::INTER_LANCZOS4);
        lastframeData.image.cpu_to_gpu(lvl);

        calcDerivativeCPU(lastframeData,lvl);
        lastframeData.der.cpu_to_gpu(lvl);
    }

    glFinish();
    std::cout << "save frame time " << t.toc() << std::endl;

    lastframeData.pose = _globalPose;//*keyframeData.pose.inverse();

    t.tic();
    float occ = calcOccupancyCPU(lastframeData,1);
    glFinish();
    std::cout << "occupancy time " << t.toc() << std::endl;

    if(occ < 0.8)
    {
        t.tic();
        changeKeyframe(lastframeData,1,0.8);
        //optMapVertex();
        //optMapJoint();
        optPoseMapJoint();

        glFinish();
        std::cout << "change keyframe time " << t.toc() << std::endl;

        return;
    }

    t.tic();
    addFrameToStack(lastframeData);
    glFinish();
    std::cout << "add frame stack time " << t.toc() << std::endl;

    t.tic();
    //optMapVertex();
    //optMapJoint();
    optPoseMapJoint();
    glFinish();
    std::cout << "update map time " << t.toc() << std::endl;

}



float mesh_vo::goodVertexViewPercentage(frame &_frame)
{
    /*
    int count = 0;
    for(int i = 0; i < VERTEX_HEIGH*VERTEX_WIDTH; i++)
    {
        if(_frame.vertexViewCount(i) > 0)
            count++;
    }
    return float(count)/(VERTEX_HEIGH*VERTEX_WIDTH);
    */
    return 1.0;
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






















