#include "mesh_vo.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

mesh_vo::mesh_vo(float fx, float fy, float cx, float cy, int _width, int _height)
{
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        float scale = std::pow(2.0f, float(lvl));

        width[lvl] = int(_width/scale);
        height[lvl] = int(_height/scale);

        K[lvl] = Eigen::Matrix3f::Zero();
        K[lvl](0,0) = fx/scale;
        K[lvl](1,1) = fy/scale;
        K[lvl](2,2) = 1.0f;
        K[lvl](0,2) = cx/scale;
        K[lvl](1,2) = cy/scale;

        KInv[lvl] = K[lvl].inverse();

        /*
        width[lvl] = int(_width);
        height[lvl] = int(_height);

        K[lvl] = Eigen::Matrix3f::Zero();
        K[lvl](0,0) = fx;
        K[lvl](1,1) = fy;
        K[lvl](2,2) = 1.0f;
        K[lvl](0,2) = cx;
        K[lvl](1,2) = cy;

        KInv[lvl] = K[lvl].inverse();
        */
    }


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
    vwidth = 32;
    vheight = 32;

    //prealocate
    for(int y=0;y<vheight;y++)
    {
        for(int x=0;x<vwidth;x++)
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

    //feedback buffer
    // Create transform feedback buffer
    glGenBuffers(1, &feedbackrbo);
    glBindBuffer(GL_ARRAY_BUFFER, feedbackrbo);
    glBufferData(GL_ARRAY_BUFFER, scene_vertices.size() * sizeof(float), nullptr, GL_STATIC_READ);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, feedbackrbo);


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
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width[0], height[0]); // use a single renderbuffer object for both a depth AND stencil buffer.
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

    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    float borderColor[] = {-1.0, -1.0, -1.0, -1.0};
    int borderID[] = {-1, -1, -1, -1};

    glGenTextures(1, &keyframeTexture);
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //GL_NEAREST
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //nearest porque sino interpola en el borde
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);//entre el ultimo pixel y borderColor
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//GL_MIRRORED_REPEAT
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);

    glGenTextures(1, &keyframeDerivativeTexture);
    glBindTexture(GL_TEXTURE_2D, keyframeDerivativeTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);

    glGenTextures(1, &currentframeTexture);
    glBindTexture(GL_TEXTURE_2D, currentframeTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);

    glGenTextures(1, &currentframeDerivativeTexture);
    glBindTexture(GL_TEXTURE_2D, currentframeDerivativeTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);

}

void mesh_vo::setKeyframeRandomIdepth(cv::Mat _keyFrame)
{

   //const float* maxGradients = new_frame->maxGradients();

    scene_vertices.clear();
    scene_indices.clear();

    for(int y=0;y<vheight;y++) //los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for(int x=0;x<vwidth;x++)
        {
            float idepth = 0.1 + 0.5f * float(y)/vheight;
            //float idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
            //float idepth = -1.0;

            float xi = (float(x)/float(vwidth-1))*float(width[0]);
            float yi = (float(y)/float(vheight-1))*float(height[0]);
//            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
//            Eigen::Vector3f p = KInv*(u/idepth);

//            scene_vertices.push_back(p(0));
//            scene_vertices.push_back(p(1));
//            scene_vertices.push_back(p(2));

            scene_vertices.push_back(xi);
            scene_vertices.push_back(yi);
            scene_vertices.push_back(idepth);

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

    glBindTexture(GL_TEXTURE_2D, keyframeTexture);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, new_frame->image());
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width[0], height[0], GL_RED, GL_FLOAT, _keyFrame.data);
    glGenerateMipmap(GL_TEXTURE_2D);

    frameDerivative(keyframeTexture, keyframeDerivativeTexture);

}

void mesh_vo::setKeyframeWithIdepth(cv::Mat _keyFrame, cv::Mat _idepth)
{    
    scene_vertices.clear();
    scene_indices.clear();

    for(int y=0;y<vheight;y++) //los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for(int x=0;x<vwidth;x++)
        {
            //float idepth = 0.1 + 0.5f * float(y)/vheight;
            //float idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
            //float idepth = -1.0;

            float xi = (float(x)/float(vwidth-1))*float(width[0]);
            float yi = (float(y)/float(vheight-1))*float(height[0]);

            float idepth = _idepth.at<float>(yi,xi);

//            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
//            Eigen::Vector3f p = KInv*(u/idepth);

//            scene_vertices.push_back(p(0));
//            scene_vertices.push_back(p(1));
//            scene_vertices.push_back(p(2));

            scene_vertices.push_back(xi);
            scene_vertices.push_back(yi);
            scene_vertices.push_back(idepth);

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

    glBindTexture(GL_TEXTURE_2D, keyframeTexture);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, new_frame->image());
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width[0], height[0], GL_RED, GL_FLOAT, _keyFrame.data);
    glGenerateMipmap(GL_TEXTURE_2D);

    frameDerivative(keyframeTexture, keyframeDerivativeTexture);
}

float mesh_vo::calcResidual(unsigned int frame, Sophus::SE3f framePose, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    float residual = 0;
    int num = 0;

    cv::Mat debug(height[lvl], width[lvl], CV_32FC1, 0.0);

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            //std::cout << "pixel: " << y << " " << x << std::endl;

            uchar vkf = keyFrame[lvl].at<uchar>(y,x);
            float keyframeId = idepth[lvl].at<float>(y,x);

            //std::cout << "vkf " << vkf << " id " << id << std::endl;

            Eigen::Vector3f poinKeyframe = (KInv[lvl]*Eigen::Vector3f(x,y,1.0))/keyframeId;
            Eigen::Vector3f pointFrame = framePose*poinKeyframe;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = (K[lvl]*pointFrame)/pointFrame(2);

            if(pixelFrame(0) < 0.0 || pixelFrame(0) > width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) > height[lvl])
                continue;

            uchar vf = frame.at<uchar>(pixelFrame(1), pixelFrame(0));

            float res = (vkf-vf);

            //std::cout << "pixel " << " " << float(vkf) << " " << float(vf) << " res " << res << std::endl;

            residual += res*res;
            num++;

            //std::cout << "accres " << residual << std::endl;

            debug.at<float>(y,x) = abs(res)*0.01;
        }

    cv::namedWindow("calcResidual debug", cv::WINDOW_NORMAL);
    cv::imshow("calcResidual debug", debug);
    cv::waitKey(30);

    return residual/num;
}

Sophus::SE3f mesh_vo::updatePose(cv::Mat _frame)
{
    std::cout << "entrando updatePose\n";

    //Sophus::SE3f newPose;
    //framePose = newPose;

    cv::Mat frame[MAX_LEVELS];
    cv::Mat frameDer[MAX_LEVELS];

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {

        //cv::GaussianBlur(_frame, frame[lvl], cv::Size(lvl*2+1,lvl*2+1),0);
        //cv::blur(_frame, frame[lvl], cv::Size(lvl*2+1,lvl*2+1));
        cv::resize(_frame, frame[lvl], cv::Size(width[lvl],height[lvl]),0,0,cv::INTER_AREA);
        //if(lvl == 0)
            frameDer[lvl] = frameDerivative(frame[lvl], lvl);
        //else
            //cv::resize(frameDer[0], frameDer[lvl], cv::Size(width[lvl],height[lvl]),0,0,cv::INTER_LINEAR);
            //cv::GaussianBlur(frameDer[0], frameDer[lvl], cv::Size(lvl*2+1,lvl*2+1),0);

    }

    //calcResidual(frame[0],framePose,0);
    //return framePose;

    //std::cout << "se setearon frames y frameder\n";

    int maxIterations[10] = {5, 20, 100, 100, 100, 100, 100, 100, 100, 100};

                    Eigen::Matrix<float, 6, 1> inc;
    //int lvl = 3;
    for(int lvl=MAX_LEVELS-1; lvl >= 0; lvl--)
    {
        //std::cin.get();

        float last_error = calcResidual(frame[lvl],framePose,lvl);

        //std::cout << "pose for lvl " << lvl << std::endl;
        //std::cout << framePose.matrix() << std::endl;
        //std::cout << "lvl " << lvl << " init error " << last_error << std::endl;

        //for(int i = 0; i < maxIterations[lvl]; i++)
        for(int it = 0; it < maxIterations[lvl]; it++)
        {
            //calcHJ(frame[lvl], frameDer[lvl], framePose ,lvl);
            //calcHJ_2(frame[lvl], frameDer[lvl], framePose ,lvl);
            calcHJ_3(frame[lvl], frameDer[lvl], framePose ,lvl);

            float lambda = 0.0;
            int n_try = 0;
            while(true)
            {
                Eigen::Matrix<float, 6, 6> acc_H_pose_lambda;
                acc_H_pose_lambda = acc_H_pose;

                for(int j = 0; j < 6; j++)
                    acc_H_pose_lambda(j,j) *= 1.0 + lambda;

                //std::cout << "H " << acc_H_pose << std::endl;
                //std::cout << "J " << acc_J_pose << std::endl;

                inc = acc_H_pose_lambda.ldlt().solve(acc_J_pose);
                //inc = acc_H_pose_lambda.colPivHouseholderQr().solve(acc_J_pose);
                //inc = 0.001*acc_J_pose/acc_J_pose.norm();

                //Sophus::SE3f new_pose = framePose.inverse()*Sophus::SE3f::exp(inc).inverse();

                //Sophus::SE3f new_pose = Sophus::SE3f::exp(inc) * framePose;
                Sophus::SE3f new_pose = Sophus::SE3f::exp(inc).inverse() * framePose;
                //Sophus::SE3f new_pose = framePose*Sophus::SE3f::exp(inc);
                //Sophus::SE3f new_pose = framePose*Sophus::SE3f::exp(inc).inverse();

                float error = calcResidual(frame[lvl],new_pose,lvl);

                std::cout << "lvl " << lvl << " it " << it << " try " << n_try << " lambda " << lambda << " error " << error << std::endl;

                //if(true)
                if(error < last_error)
                {
                    std::cout << "update accepted " << std::endl;
                    //std::cout << "lambda " << poseLambda << std::endl;

                    //std::cout << "p " << p << std::endl;

                    /*
                    std::cout << "inc " << std::endl;
                    std::cout << inc << std::endl;
                    std::cout << "inc SE3 " << std::endl;
                    std::cout << Sophus::SE3f::exp(inc).matrix() << std::endl;
                    std::cout << "last pose " << std::endl;
                    std::cout << framePose.matrix() << std::endl;
                    std::cout << "pose " << std::endl;
                    std::cout << new_pose.matrix() << std::endl;

                    std::cin.get();
                    */

                    //accept update, decrease lambda
                    framePose = new_pose;

                    float p = error / last_error;


                    if(lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

                    last_error = error;

                    if( p >  0.999f)
                    {
                        std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
                        //if converged, do next level
                        it = maxIterations[lvl];
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
                        lambda *= std::pow(2.0, n_try);

                    //reject update, increase lambda, use un-updated data
                    //std::cout << "update rejected " << std::endl;

                    if(!(inc.dot(inc) > 1e-8))
                    {
                        std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations with lambda " << lambda << std::endl;
                        //if too small, do next level!
                        it = maxIterations[lvl];
                        break;
                    }
                }
            }
        }
    }

    return framePose;
}
/*
void mesh_vo::calcHJ(cv::Mat frame, cv::Mat frameDer, Sophus::SE3f framePose, int lvl)
{
    acc_J_pose.setZero();
    acc_H_pose.setZero();

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            uchar vkf = keyFrame[lvl].at<uchar>(y,x);
            float id = idepth[lvl].at<float>(y,x);

            Eigen::Vector3f rayKeyframe = KInv[lvl]*Eigen::Vector3f(x,y,1.0);
            Eigen::Vector3f poinKeyframe = rayKeyframe/id;
            Eigen::Vector3f pointFrame = framePose*poinKeyframe;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f rayFrame = pointFrame/pointFrame(2);
            Eigen::Vector3f pixelFrame = K[lvl]*rayFrame;

            if(pixelFrame(0) < 0.0 || pixelFrame(0) >= width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= height[lvl])
                continue;

            uchar vf = frame.at<uchar>(pixelFrame(1), pixelFrame(0));

            //std::cout << "pixel " << " " << float(vkf) << " " << float(vf) << std::endl;

            Eigen::Matrix3f d_uf_d_pf;
            d_uf_d_pf(0,0) = K[lvl](0,0);
            d_uf_d_pf(0,1) = 0.0;
            d_uf_d_pf(0,2) = 0.0;

            d_uf_d_pf(1,0) = 0.0;
            d_uf_d_pf(1,1) = K[lvl](1,1);
            d_uf_d_pf(1,2) = 0.0;

            d_uf_d_pf(2,0) = -rayFrame(0)*K[lvl](0,0);
            d_uf_d_pf(2,1) = -rayFrame(1)*K[lvl](1,1);
            d_uf_d_pf(2,2) = 0.0;

            d_uf_d_pf = d_uf_d_pf/pointFrame(2);

            Eigen::Matrix3f d_pf_d_tra;

            d_pf_d_tra(0,0) = 1.0;
            d_pf_d_tra(0,1) = 0.0;
            d_pf_d_tra(0,2) = 0.0;

            d_pf_d_tra(1,0) = 0.0;
            d_pf_d_tra(1,1) = 1.0;
            d_pf_d_tra(1,2) = 0.0;

            d_pf_d_tra(2,0) = 0.0;
            d_pf_d_tra(2,1) = 0.0;
            d_pf_d_tra(2,2) = 1.0;

            Eigen::Matrix3f d_pf_d_rot;
            d_pf_d_rot(0,0) = 0.0;
            d_pf_d_rot(0,1) = -pointFrame(2);
            d_pf_d_rot(0,2) = pointFrame(1);

            d_pf_d_rot(1,0) = pointFrame(2);
            d_pf_d_rot(1,1) = 0.0;
            d_pf_d_rot(1,2) = -pointFrame(0);

            d_pf_d_rot(2,0) = -pointFrame(1);
            d_pf_d_rot(2,1) = pointFrame(0);
            d_pf_d_rot(2,2) = 0.0;

            cv::Vec2f der = frameDer.at<cv::Vec2f>(pixelFrame(1),pixelFrame(0));
            Eigen::Vector2f d_f_d_uf(der.val[0],der.val[1]);
            //vec2 d_f_d_uf = textureLod(frameDerivative, ufTexCoord, lvl).rg;

            //std::cout << "derivative " << d_f_d_uf << std::endl;

            Eigen::Vector3f d_I_d_pf;
            d_I_d_pf(0) = d_f_d_uf(0)*d_uf_d_pf(0,0);
            d_I_d_pf(1) = d_f_d_uf(1)*d_uf_d_pf(1,1);
            d_I_d_pf(2) = d_f_d_uf(0)*d_uf_d_pf(2,0)+d_f_d_uf(1)*d_uf_d_pf(2,1);

            //vec3 d_I_d_pf = vec3(dot(d_f_d_uf,d_uf_d_pf[0].xy), dot(d_f_d_uf,d_uf_d_pf[1].xy), dot(d_f_d_uf,d_uf_d_pf[2].xy));
            Eigen::Vector3f d_I_d_rot = d_I_d_pf.transpose()*d_pf_d_rot;
            Eigen::Vector3f d_I_d_tra = d_I_d_pf.transpose()*d_pf_d_tra;

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

void mesh_vo::calcHJ_2(cv::Mat frame, cv::Mat frameDer, Sophus::SE3f framePose, int lvl)
{
    acc_J_pose.setZero();
    acc_H_pose.setZero();

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            uchar vkf = keyFrame[lvl].at<uchar>(y,x);
            float keyframeId = idepth[lvl].at<float>(y,x);

            Eigen::Vector3f rayKeyframe = KInv[lvl]*Eigen::Vector3f(x,y,1.0);
            Eigen::Vector3f poinKeyframe = rayKeyframe/keyframeId;
            Eigen::Vector3f pointFrame = framePose*poinKeyframe;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f rayFrame = pointFrame/pointFrame(2);
            Eigen::Vector3f pixelFrame = K[lvl]*rayFrame;

            if(pixelFrame(0) < 0.0 || pixelFrame(0) >= width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= height[lvl])
                continue;

            uchar vf = frame.at<uchar>(pixelFrame(1), pixelFrame(0));

            //std::cout << "pixel " << " " << float(vkf) << " " << float(vf) << std::endl;

            cv::Vec2f der = frameDer.at<cv::Vec2f>(pixelFrame(1),pixelFrame(0));
            Eigen::Vector2f d_f_d_uf(der.val[0],der.val[1]);

            float id = 1.0/pointFrame(2);
            float id2 = 1.0/(pointFrame(2)*pointFrame(2));

            Eigen::Vector2f g = Eigen::Vector2f(d_f_d_uf(0)*K[lvl](0,0), d_f_d_uf(1)*K[lvl](1,1));

            Eigen::Vector3f d_I_d_tra = Eigen::Vector3f(g(0)*id, g(1)*id, -(g(0)*pointFrame(0)*id2 + g(1)*pointFrame(1)*id2));
            Eigen::Vector3f d_I_d_rot = Eigen::Vector3f(-(g(0)*pointFrame(0)*pointFrame(1)*id2 + g(1) + g(1)*pointFrame(1)*pointFrame(1)*id2), g(0) + g(0)*pointFrame(0)*pointFrame(0)*id2 + g(1)*pointFrame(0)*pointFrame(1)*id2, -g(0)*pointFrame(1)*id + g(1)*pointFrame(0)*id);

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

void mesh_vo::calcHJ_3(cv::Mat frame, cv::Mat frameDer, Sophus::SE3f framePose, int lvl)
{
    acc_J_pose.setZero();
    acc_H_pose.setZero();

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            //std::cout << 1 << std::endl;
            uchar vkf = keyFrame[lvl].at<uchar>(y,x);
            float keyframeId = idepth[lvl].at<float>(y,x);
//std::cout << 2 << std::endl;
            Eigen::Vector3f poinKeyframe = (KInv[lvl]*Eigen::Vector3f(x,y,1.0))/keyframeId;
            Eigen::Vector3f pointFrame = framePose*poinKeyframe;
//std::cout << 3 << std::endl;
            if(pointFrame(2) <= 0.0)
                continue;
//std::cout << 4 << std::endl;
            Eigen::Vector3f pixelFrame = (K[lvl]*pointFrame)/pointFrame(2);
//std::cout << 5 << std::endl;
            if(pixelFrame(0) < 0.0 || pixelFrame(0) >= width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= height[lvl])
                continue;
//std::cout << 6 << std::endl;
            uchar vf = frame.at<uchar>(pixelFrame(1), pixelFrame(0));
//std::cout << 7 << std::endl;
            //std::cout << "pixel " << " " << float(vkf) << " " << float(vf) << std::endl;

            cv::Vec2f der = frameDer.at<cv::Vec2f>(pixelFrame(1),pixelFrame(0));
            Eigen::Vector2f d_f_d_uf(der.val[0],der.val[1]);
//std::cout << 8 << std::endl;
            float id = 1.0/pointFrame(2);

            float v0 = d_f_d_uf(0) * K[lvl](0,0) * id;
            float v1 = d_f_d_uf(1) * K[lvl](1,1) * id;
            float v2 = -(v0 * pointFrame(0) + v1 * pointFrame(1)) * id;
//std::cout << 9 << std::endl;
            Eigen::Vector3f d_I_d_tra = Eigen::Vector3f(v0, v1, v2);
            Eigen::Vector3f d_I_d_rot = Eigen::Vector3f( -pointFrame(2) * v1 + pointFrame(1) * v2, pointFrame(2) * v0 - pointFrame(0) * v2, -pointFrame(1) * v0 + pointFrame(0) * v1);

            float residual = (vf - vkf);

            Eigen::Matrix<float, 6, 1> J_pose;
            J_pose << d_I_d_tra(0), d_I_d_tra(1), d_I_d_tra(2), d_I_d_rot(0), d_I_d_rot(1), d_I_d_rot(2);

            acc_J_pose += J_pose*residual;
//std::cout << 10 << std::endl;
            for(int i = 0; i < 6; i++)
            {
                for(int j = 0; j < 6; j++)
                {
                    acc_H_pose(i,j) += J_pose[i]*J_pose[j];
                }
            }
        }
}
*/
void mesh_vo::frameDerivative(unsigned int frame, unsigned int frameDerivative)
{
    //calculate frame derivative
    //glfwMakeContextCurrent(frameWindow);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameDerivative, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, frameDerivative, 0);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! d_C_d_z" << std::endl;


    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! frameDerivative" << std::endl;
    //    auto fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    //    if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
    //        std::cout << "Framebuffer not complete: " << fboStatus << std::endl;

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! incomplete attachment" << std::endl;
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! missing attachment" << std::endl;
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! incomplete draw buffer" << std::endl;
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! incomplete read buffer" << std::endl;
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_UNSUPPORTED)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! unsupported" << std::endl;

    glDisable(GL_DEPTH_TEST);
    glClearColor(-1.0f, -1.0f, -1.0f, -1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    // bind textures on corresponding texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, frame);
    // activate shader
    frameDerivativeShader.use();
    // render frame
    glBindVertexArray(frame_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glGenerateMipmap(GL_TEXTURE_2D);

        glBindVertexArray(0);
}
