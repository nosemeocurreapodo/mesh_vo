#include "mesh_vo.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

        projMat[lvl] = create_glm_prj_matrix(fx/scale, fy/scale, cx/scale, cy/scale, _width/scale, _height/scale, 0.0001, 10000.0);
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
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST); //GL_NEAREST
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_NEAREST); //nearest porque sino interpola en el borde
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);//entre el ultimo pixel y borderColor
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//GL_MIRRORED_REPEAT
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &keyframeDerivativeTexture);
    glBindTexture(GL_TEXTURE_2D, keyframeDerivativeTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &frameTexture);
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 6);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &frameDerivativeTexture);
    glBindTexture(GL_TEXTURE_2D, frameDerivativeTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &idepthTexture);
    glBindTexture(GL_TEXTURE_2D, idepthTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &residualTexture);
    glBindTexture(GL_TEXTURE_2D, residualTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &traTexture);
    glBindTexture(GL_TEXTURE_2D, traTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &rotTexture);
    glBindTexture(GL_TEXTURE_2D, rotTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    keyframe_cpu_data = new GLfloat[width[0]*height[0]];
    frame_cpu_data = new GLfloat[width[0]*height[0]];
    frameDer_cpu_data = new GLfloat[width[0]*height[0]*2];
    idepth_cpu_data = new GLfloat[width[0]*height[0]];

    residual_cpu_data = new GLfloat[width[0]*height[0]];
    tra_cpu_data = new GLfloat[width[0]*height[0]*3];
    rot_cpu_data = new GLfloat[width[0]*height[0]*3];

    frameDerivativeShader.init("frameDerivative.vs", "frameDerivative.fs");
    frameDerivativeShader.use();
    frameDerivativeShader.setInt("frame", 0);

    residualShader.init("residual.vs", "residual.fs");
    residualShader.use();
    residualShader.setInt("keyframe", 0);
    residualShader.setInt("frame", 1);

    calcHJShader.init("calcHJ.vs", "calcHJ.gs", "calcHJ.fs");
    calcHJShader.use();
    calcHJShader.setInt("keyframe", 0);
    calcHJShader.setInt("keyframeDer", 1);
    calcHJShader.setInt("frame", 2);
    calcHJShader.setInt("frameDer", 3);

    idepthShader.init("idepth.vs", "idepth.fs");

    showTextureShader.init("showTexture.vs", "showTexture.fs");
    showTextureShader.use();
    showTextureShader.setInt("frame", 0);

    debugShader.init("debug.vs", "debug.gs", "debug.fs");
    debugShader.use();
    debugShader.setInt("keyframe", 0);
    debugShader.setInt("frame", 1);
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

            float xi = (float(x)/float(vwidth-1));
            float yi = (float(y)/float(vheight-1));
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

            float xi = (float(x)/float(vwidth-1));
            float yi = (float(y)/float(vheight-1));

            float idepth = _idepth.at<float>(yi*(height[0]-1),xi*(width[0]-1));

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

float mesh_vo::calcResidual(unsigned int frame, Sophus::SE3f framePose, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, residualTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcResidual" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(-1.0f, -1.0f, -1.0f, -1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, frame);


    // activate shader
    residualShader.use();

    residualShader.setMat4("cameraPose", eigen2glm_mat4(framePose.matrix()));
    residualShader.setMat4("opencv2opengl", opencv2opengl);
    residualShader.setMat4("projection", projMat[lvl]);
    residualShader.setMat3("K", eigen2glm_mat3(K[lvl]));
    residualShader.setMat3("invK", eigen2glm_mat3(KInv[lvl]));
    residualShader.setFloat("width", width[lvl]);
    residualShader.setFloat("height", height[lvl]);
    residualShader.setInt("lvl", lvl);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    //glActiveTexture(GL_TEXTURE0);
    //glBindTexture(GL_TEXTURE_2D, residualTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    //glGenerateMipmap(GL_TEXTURE_2D);

    float residual = 0.0;
    int rlvl = lvl;
    //for(int rlvl = lvl; rlvl < 6; rlvl++)
    {
        residual = reduce_residual_CPU(residualTexture, rlvl);
        //std::cout << "reduce residual from lvl " << lvl << " to " << rlvl << " residual " << residual << std::endl;
    }

    return residual;
}

float mesh_vo::calcResidual_CPU(unsigned int frame, Sophus::SE3f framePose, int lvl)
{
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, keyframe_cpu_data);

    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, frame_cpu_data);

    glBindTexture(GL_TEXTURE_2D, idepthTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, idepth_cpu_data);

    //std::cout << "entrando calcResidual" << std::endl;

     float residual = 0;
     int num = 0;

     cv::Mat debug(height[lvl], width[lvl], CV_32FC1, 0.0);

     for(int y = 0; y < height[lvl]; y++)
         for(int x = 0; x < width[lvl]; x++)
         {
             int index = (height[lvl]-y)*width[lvl] + x;

             float vkf = keyframe_cpu_data[index];
             float keyframeId = idepth_cpu_data[index];

             if(keyframeId <= 0.0)
                 continue;

             Eigen::Vector3f poinKeyframe = (KInv[lvl]*Eigen::Vector3f(x,y,1.0))/keyframeId;
             Eigen::Vector3f pointFrame = framePose*poinKeyframe;

             if(pointFrame(2) <= 0.0)
                 continue;

             Eigen::Vector3f pixelFrame = K[lvl]*(pointFrame/pointFrame(2));

             if(pixelFrame(0) < 0.0 || pixelFrame(0) >= width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) >= height[lvl])
                 continue;

             int nindex = (height[lvl]-int(pixelFrame(1)))*width[lvl] + int(pixelFrame(0));

             float vf = frame_cpu_data[nindex];

             float res = (vkf-vf);

             //std::cout << "pixel " << " " << float(vkf) << " " << float(vf) << " res " << res << std::endl;

             residual += res*res;
             num++;

             //std::cout << "accres " << residual << std::endl;

             debug.at<float>(y,x) = res*0.01;
         }

     cv::namedWindow("calcResidual debug", cv::WINDOW_NORMAL);
     cv::imshow("calcResidual debug", debug);
     cv::waitKey(30);

     return residual/num;
}

float mesh_vo::reduce_residual_CPU(unsigned int residualTexture, int lvl)
{
    glBindTexture(GL_TEXTURE_2D, residualTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, residual_cpu_data);

    float error = 0.0;
    int count = 0;

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            int index = (height[lvl] - y)*width[lvl] + x;
            float res = residual_cpu_data[index];
            if(res < 0)
                continue;

            count++;
            error += res;
        }

    if(count > 0)
        error /= count;
    else
        error = 1000000000000000000000000000.0f;

    return error;
}

Sophus::SE3f mesh_vo::updatePose(cv::Mat _frame)
{
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width[0], height[0], GL_RED, GL_FLOAT, _frame.data);
    glGenerateMipmap(GL_TEXTURE_2D);

    frameDerivative(frameTexture, frameDerivativeTexture);


    int lvl = 0;
    //for(int lvl=MAX_LEVELS-1; lvl >= 0; lvl--)
    {
        //float last_error = calcResidual(frameTexture,framePose,lvl);
        //std::cout << "lvl " << lvl << " error " << last_error << std::endl;
        //calcHJ(frameTexture, frameDerivativeTexture, framePose ,lvl);
        //Sophus::SE3f keyframePose;
        //calcIdepth(keyframePose, lvl);
        float residual1 = calcResidual(frameTexture, framePose, lvl);
        showTexture(residualTexture, lvl);
        Sophus::SE3f keyframePose;
        calcIdepth(keyframePose, lvl);
        float residual2 = calcResidual_CPU(frameTexture, framePose, lvl);

        std::cout << "residual " << residual1 << " " << residual2 << std::endl;
        //showDebug(frameTexture, framePose, lvl2);
        cv::waitKey(30);
    }
    return framePose;


    int maxIterations[10] = {5, 20, 100, 100, 100, 100, 100, 100, 100, 100};

    Eigen::Matrix<float, 6, 1> inc;
    //int lvl = 4;
    for(int lvl=MAX_LEVELS-1; lvl >= 0; lvl--)
    {
        Sophus::SE3f keyframePose;
        calcIdepth(keyframePose, lvl);

        float last_error = calcResidual_CPU(frameTexture,framePose,lvl);
        //float last_error = calcResidual(frameTexture,framePose,lvl);

        //showTexture(residualTexture, lvl);
        //showDebug(frameTexture, framePose, 0);
        //std::cout << "pose for lvl " << lvl << std::endl;
        //std::cout << framePose.matrix() << std::endl;
        std::cout << "lvl " << lvl << " init error " << last_error << std::endl;

        for(int it = 0; it < maxIterations[lvl]; it++)
        {
            //calcHJ_CPU(frameTexture, frameDerivativeTexture, framePose ,lvl);
            calcHJ(frameTexture, frameDerivativeTexture, framePose ,lvl);
            showTexture(residualTexture, lvl);

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

                //Sophus::SE3f new_pose = framePose.inverse()*Sophus::SE3f::exp(inc).inverse();
                //Sophus::SE3f new_pose = Sophus::SE3f::exp(inc) * framePose;
                Sophus::SE3f new_pose = Sophus::SE3f::exp(inc).inverse() * framePose;
                //Sophus::SE3f new_pose = framePose*Sophus::SE3f::exp(inc);
                //Sophus::SE3f new_pose = framePose*Sophus::SE3f::exp(inc).inverse();

                //std::cout << "new_pose " << new_pose.matrix() << std::endl;

                float error = calcResidual_CPU(frameTexture,new_pose,lvl);
                float error2 = calcResidual(frameTexture,new_pose,lvl);
                std::cout << "error2 " << error2 << std::endl;
                std::cout << "lvl " << lvl << " it " << it << " try " << n_try << " lambda " << lambda << " error " << error << std::endl;

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
                    std::cout << "update rejected " << std::endl;

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


void mesh_vo::calcHJ(unsigned int frame, unsigned int frameDer, Sophus::SE3f framePose, int lvl)
{   
    glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, residualTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, traTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, rotTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0,
                                GL_COLOR_ATTACHMENT1,
                                GL_COLOR_ATTACHMENT2};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcResidual" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, keyframeDerivativeTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, frame);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, frameDer);

    // activate shader
    calcHJShader.use();

    calcHJShader.setMat4("cameraPose", eigen2glm_mat4(framePose.matrix()));
    calcHJShader.setMat4("opencv2opengl", opencv2opengl);
    calcHJShader.setMat4("projection", projMat[lvl]);
    calcHJShader.setMat3("K", eigen2glm_mat3(K[lvl]));
    calcHJShader.setMat3("invK", eigen2glm_mat3(KInv[lvl]));
    calcHJShader.setFloat("width", width[lvl]);
    calcHJShader.setFloat("height", height[lvl]);
    calcHJShader.setInt("lvl", lvl);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);

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


    glBindTexture(GL_TEXTURE_2D, residualTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, residual_cpu_data);

    glBindTexture(GL_TEXTURE_2D, traTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RGB, GL_FLOAT, tra_cpu_data);

    glBindTexture(GL_TEXTURE_2D, rotTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RGB, GL_FLOAT, rot_cpu_data);


    acc_J_pose.setZero();
    acc_H_pose.setZero();

    for(int index = 0; index < width[lvl]*height[lvl]; index++)
    {
        Eigen::Matrix<float, 6, 1> J_pose;
        J_pose << tra_cpu_data[index*3], tra_cpu_data[index*3+1], tra_cpu_data[index*3+2], rot_cpu_data[index*3], rot_cpu_data[index*3+1], rot_cpu_data[index*3+2];

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
            for(int j = 0; j < 6; j++)
            {
                acc_H_pose(i,j) += J_pose[i]*J_pose[j];
            }
        }

        //std::cout << "acc_J_pose " << acc_J_pose << std::endl;
        //std::cout << "acc_J_pose " << acc_H_pose << std::endl;

    }
}

void mesh_vo::calcHJ_CPU(unsigned int frame, unsigned int frameDer, Sophus::SE3f framePose, int lvl)
{
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, keyframe_cpu_data);

    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, frame_cpu_data);

    glBindTexture(GL_TEXTURE_2D, frameDerivativeTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RG, GL_FLOAT, frameDer_cpu_data);

    glBindTexture(GL_TEXTURE_2D, idepthTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, idepth_cpu_data);

    acc_J_pose.setZero();
    acc_H_pose.setZero();

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            int index = (height[lvl] - y)*width[lvl] + x;

            float keyframeId = idepth_cpu_data[index];

            if(keyframeId <= 0.0)
                continue;

            float vkf = keyframe_cpu_data[index];

            Eigen::Vector3f poinKeyframe = (KInv[lvl]*Eigen::Vector3f(x,y,1.0))/keyframeId;
            Eigen::Vector3f pointFrame = framePose*poinKeyframe;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = (K[lvl]*pointFrame)/pointFrame(2);

            if(pixelFrame(0) < 0.0 || pixelFrame(0) >= width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= height[lvl])
                continue;

            int nindex = (height[lvl] - int(pixelFrame(1)))*width[lvl] + int(pixelFrame(0));
            float vf = frame_cpu_data[nindex];

            Eigen::Vector2f d_f_d_uf;
            d_f_d_uf(0) = frameDer_cpu_data[nindex*2];
            d_f_d_uf(1) = frameDer_cpu_data[nindex*2+1];

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

void mesh_vo::frameDerivative(unsigned int frame, unsigned int frameDerivative)
{
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        //calculate frame derivative
        glfwMakeContextCurrent(frameWindow);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameDerivative, 0);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, frameDerivative, lvl);

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
        glDisable(GL_CULL_FACE);
        glClearColor(-1.0f, -1.0f, -1.0f, -1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, frame);
        // activate shader
        frameDerivativeShader.use();
        frameDerivativeShader.setInt("lvl", lvl);
        // render frame
        glBindVertexArray(frame_VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

    }

    //glActiveTexture(GL_TEXTURE0);
    //glBindTexture(GL_TEXTURE_2D, frameDerivative);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glGenerateMipmap(GL_TEXTURE_2D);

        //glBindVertexArray(0);
}

void mesh_vo::showTexture(unsigned int texture, int lvl)
{
    glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[0],height[0]);

    // visualize
    // now bind back to default framebuffer and draw a quad plane with the attached framebuffer color texture
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.
    glDisable(GL_CULL_FACE);
    // clear all relevant buffers
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessery actually, since we won't be able to see behind the quad anyways)
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    showTextureShader.use();
    showTextureShader.setInt("lvl", lvl);

    glBindVertexArray(frame_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glfwSwapBuffers(frameWindow);
}

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
    debugShader.setMat3("K", eigen2glm_mat3(K[lvl]));
    debugShader.setMat3("invK", eigen2glm_mat3(KInv[lvl]));
    debugShader.setFloat("width", width[lvl]);
    debugShader.setFloat("height", height[lvl]);
    debugShader.setInt("lvl", lvl);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(frameWindow);
}

void mesh_vo::calcIdepth(Sophus::SE3f framePose, int lvl)
{
    glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, idepthTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcResidual" << std::endl;

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // activate shader
    idepthShader.use();

    idepthShader.setMat4("cameraPose", eigen2glm_mat4(framePose.matrix()));
    idepthShader.setMat4("opencv2opengl", opencv2opengl);
    idepthShader.setMat4("projection", projMat[lvl]);
    idepthShader.setMat3("invK", eigen2glm_mat3(KInv[lvl]));
    idepthShader.setFloat("width", width[lvl]);
    idepthShader.setFloat("height", height[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);
}

