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

mesh_vo::mesh_vo(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
{
    float xp = float(MAX_WIDTH)/_width;
    float yp = float(MAX_HEIGHT)/_height;

    float out_fx, out_fy, out_cx, out_cy;
    out_fx = _fx*xp; out_fy = _fy*yp; out_cx = _cx*xp; out_cy = _cy*yp;

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

        projMat[lvl] = create_glm_prj_matrix(out_fx/scale, out_fy/scale, out_cx/scale, out_cy/scale, MAX_WIDTH/scale, MAX_HEIGHT/scale, 0.0001, 10000.0);
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST); //GL_NEAREST
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST); //nearest porque sino interpola en el borde
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);//entre el ultimo pixel y borderColor
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);//GL_MIRRORED_REPEAT
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &frameTexture);
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 6);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    //glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &frameDerivativeTexture);
    glBindTexture(GL_TEXTURE_2D, frameDerivativeTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);//border los de afuera son erroneos
    //glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width[0], height[0], 0, GL_RG, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    for(int i = 0; i < MAX_FRAMES; i++)
    {
        glGenTextures(1, &frameTextureStack[i]);
        glBindTexture(GL_TEXTURE_2D, frameTextureStack[i]);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 6);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
        //glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
        glGenerateMipmap(GL_TEXTURE_2D);

        glGenTextures(1, &frameDerivativeTextureStack[i]);
        glBindTexture(GL_TEXTURE_2D, frameDerivativeTextureStack[i]);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);//border los de afuera son erroneos
        //glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width[0], height[0], 0, GL_RG, GL_UNSIGNED_BYTE, NULL);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    lastFrameAdded = -MAX_FRAMES;

    glGenTextures(1, &idepthTexture);
    glBindTexture(GL_TEXTURE_2D, idepthTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &residualTexture);
    glBindTexture(GL_TEXTURE_2D, residualTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &errorTexture);
    glBindTexture(GL_TEXTURE_2D, errorTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &reduceFloatTexture);
    glBindTexture(GL_TEXTURE_2D, reduceFloatTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &reduceVec4Texture);
    glBindTexture(GL_TEXTURE_2D, reduceVec4Texture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &traTexture);
    glBindTexture(GL_TEXTURE_2D, traTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width[0], height[0], 0, GL_RGB, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &rotTexture);
    glBindTexture(GL_TEXTURE_2D, rotTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width[0], height[0], 0, GL_RGB, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);


    glGenTextures(1, &JposeTexture1);
    glBindTexture(GL_TEXTURE_2D, JposeTexture1);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width[0], height[0], 0, GL_RGBA, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &JposeTexture2);
    glBindTexture(GL_TEXTURE_2D, JposeTexture2);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width[0], height[0], 0, GL_RGBA, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &JposeTexture3);
    glBindTexture(GL_TEXTURE_2D, JposeTexture3);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width[0], height[0], 0, GL_RGBA, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &JposeTexture4);
    glBindTexture(GL_TEXTURE_2D, JposeTexture4);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width[0], height[0], 0, GL_RGBA, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &JposeTexture5);
    glBindTexture(GL_TEXTURE_2D, JposeTexture5);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width[0], height[0], 0, GL_RGBA, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &JposeTexture6);
    glBindTexture(GL_TEXTURE_2D, JposeTexture6);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width[0], height[0], 0, GL_RGBA, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &JposeTexture7);
    glBindTexture(GL_TEXTURE_2D, JposeTexture7);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width[0], height[0], 0, GL_RGBA, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);


    glGenTextures(1, &vertexID_Texture);
    glBindTexture(GL_TEXTURE_2D, vertexID_Texture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width[0], height[0], 0, GL_RGB, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &primitiveID_Texture);
    glBindTexture(GL_TEXTURE_2D, primitiveID_Texture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &d_I_d_p0_Texture);
    glBindTexture(GL_TEXTURE_2D, d_I_d_p0_Texture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &d_I_d_p1_Texture);
    glBindTexture(GL_TEXTURE_2D, d_I_d_p1_Texture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glGenTextures(1, &d_I_d_p2_Texture);
    glBindTexture(GL_TEXTURE_2D, d_I_d_p2_Texture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//border los de afuera son erroneos
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width[0], height[0], 0, GL_RED, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    error_cpu_data = new GLfloat[width[0]*height[0]];

    residual_cpu_data = new GLfloat[width[0]*height[0]];
    tra_cpu_data = new GLfloat[width[0]*height[0]*3];
    rot_cpu_data = new GLfloat[width[0]*height[0]*3];

    j_pose_data1 = new GLfloat[width[0]*height[0]*4];
    j_pose_data2 = new GLfloat[width[0]*height[0]*4];
    j_pose_data3 = new GLfloat[width[0]*height[0]*4];
    j_pose_data4 = new GLfloat[width[0]*height[0]*4];
    j_pose_data5 = new GLfloat[width[0]*height[0]*4];
    j_pose_data6 = new GLfloat[width[0]*height[0]*4];
    j_pose_data7 = new GLfloat[width[0]*height[0]*4];

    vertexID_cpu_data = new GLfloat[width[0]*height[0]*3];
    primitiveID_cpu_data = new GLfloat[width[0]*height[0]];

    d_I_d_p0_cpu_data = new GLfloat[width[0]*height[0]];
    d_I_d_p1_cpu_data = new GLfloat[width[0]*height[0]];
    d_I_d_p2_cpu_data = new GLfloat[width[0]*height[0]];

    frameDerivativeShader.init("frameDerivative.vs", "frameDerivative.fs");
    frameDerivativeShader.use();
    frameDerivativeShader.setInt("frame", 0);

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

    acc_H_map = Eigen::MatrixXf::Zero(vwidth*vheight*3, vwidth*vheight*3);
    acc_J_map = Eigen::VectorXf::Zero(vwidth*vheight*3);
    inc_map = Eigen::VectorXf(vwidth*vheight*3);

    occupancy = 1.0;

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
            float idepth = 0.1 + 0.5f * float(y)/vheight;
            //float idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
            //float idepth = -1.0;

            float xi = (float(x)/float(vwidth-1))*width[0];
            float yi = (float(y)/float(vheight-1))*height[0];
            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
            Eigen::Vector3f r = Eigen::Vector3f(fxinv[0]*u(0) + cxinv[0], fyinv[0]*u(1) + cyinv[0], 1.0);
            Eigen::Vector3f p = r/idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(log(p(2)));

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
    cv::Mat frameResized;//dst image
    cv::Mat frameFlipped;
    cv::resize(_keyFrame,frameResized,cv::Size(width[0], height[0]));//resize image

    keyframePose = _pose;

    glBindTexture(GL_TEXTURE_2D, keyframeTexture);
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(frameResized, keyframeMat[lvl], cv::Size(width[lvl],height[lvl]),0,0,cv::INTER_AREA);

        cv::flip(keyframeMat[lvl], frameFlipped, 0);
        glTexSubImage2D(GL_TEXTURE_2D, lvl, 0, 0, width[lvl], height[lvl], GL_RED, GL_UNSIGNED_BYTE, frameFlipped.data);
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
            if(idepth <= 0.0)
                idepth = 0.1 + 0.5f * float(y)/vheight;

            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
            Eigen::Vector3f r = Eigen::Vector3f(fxinv[0]*u(0) + cxinv[0], fyinv[0]*u(1) + cyinv[0], 1.0);
            Eigen::Vector3f p = r/idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(log(p(2)));

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
    cv::Mat frameResized;//dst image
    cv::Mat frameFlipped;
    cv::resize(_keyFrame,frameResized,cv::Size(width[0], height[0]));//resize image

    keyframePose = _pose;

    glBindTexture(GL_TEXTURE_2D, keyframeTexture);
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(frameResized, keyframeMat[lvl], cv::Size(width[lvl],height[lvl]),0,0,cv::INTER_AREA);
        cv::flip(keyframeMat[lvl], frameFlipped, 0);
        glTexSubImage2D(GL_TEXTURE_2D, lvl, 0, 0, width[lvl], height[lvl], GL_RED, GL_UNSIGNED_BYTE, frameFlipped.data);
    }
}

void mesh_vo::changeKeyframe(unsigned int _keyframeTexture, Sophus::SE3f _keyframePose)
{
    int lvl = 2;

    calcIdepth(_keyframePose, lvl);

    glBindTexture(GL_TEXTURE_2D, idepthTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, idepth_cpu_data);

    scene_vertices.clear();
    scene_indices.clear();

    for(int y=0;y<vheight;y++) //los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for(int x=0;x<vwidth;x++)
        {
            float xi = (float(x)/float(vwidth-1))*width[lvl];
            float yi = (float(y)/float(vheight-1))*height[lvl];

            int index = (height[lvl] - int(yi))*width[lvl] + int(xi);

            float idepth = idepth_cpu_data[index];

            if(idepth <= 0.0)
                idepth = 0.1 + 0.5f * float(y)/vheight;

            Eigen::Vector3f u = Eigen::Vector3f(xi,yi,1.0);
            Eigen::Vector3f r = Eigen::Vector3f(fxinv[lvl]*u(0) + cxinv[lvl], fyinv[lvl]*u(1) + cyinv[lvl], 1.0);
            Eigen::Vector3f p = r/idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(log(p(2)));

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

    //copy texture to keyframetexture, and generate its mipmaps
    copyTexture(_keyframeTexture, keyframeTexture, 0);
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);
    glGenerateMipmap(GL_TEXTURE_2D);

}

void mesh_vo::addFrameToStack(unsigned int _frameTexture, unsigned int _frameDerTexture, Sophus::SE3f _framePose)
{
    lastFrameAdded++;
    if(lastFrameAdded >= MAX_FRAMES)
        lastFrameAdded = 0;

    //copy texture and generate its mipmaps
    copyTexture(_frameTexture, frameTextureStack[abs(lastFrameAdded)], 0);
    glBindTexture(GL_TEXTURE_2D, frameTextureStack[abs(lastFrameAdded)]);
    glGenerateMipmap(GL_TEXTURE_2D);

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
        copyTexture(_frameDerTexture, frameDerivativeTextureStack[abs(lastFrameAdded)], lvl);

    framePoseStack[abs(lastFrameAdded)] = _framePose;
}

float mesh_vo::calcError(unsigned int _frameTexture, Sophus::SE3f _framePose, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    //return calcResidual_CPU(frame, framePose, lvl);

    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, errorTexture, lvl);

    unsigned int drawbuffers[]={GL_COLOR_ATTACHMENT0};
    glDrawBuffers(sizeof(drawbuffers)/sizeof(unsigned int), drawbuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcResidual" << std::endl;

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frameTexture);


    // activate shader
    errorShader.use();

    errorShader.setMat4("framePose", eigen2glm_mat4(_framePose.matrix()));
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


    int new_lvl[MAX_LEVELS] = {5,6,7,8,9};

    reduceFloat(errorTexture, lvl, new_lvl[lvl]);

    glBindTexture(GL_TEXTURE_2D, errorTexture);
    glGetTexImage(GL_TEXTURE_2D, new_lvl[lvl], GL_RED, GL_FLOAT, error_cpu_data);
    float error = 0.0;
    int count = 0;
    for(int index = 0; index < width[new_lvl[lvl]]*height[new_lvl[lvl]]; index+=1)
    {
        float res = error_cpu_data[index];
        if(res <= 0)
            continue;
        count++;
        error += res;
    }

    if(count > height[new_lvl[lvl]]*width[new_lvl[lvl]]*0.5)
        error /= count;
    else
        error = 1230000000000000000000000000.0f;

    //std::cout << "lvl " << lvl << " new_lvl " << new_lvl << " error " << error << std::endl;

    return error;
}

float mesh_vo::calcError_CPU(cv::Mat frame, Sophus::SE3f framePose, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    float residual = 0;
    int num = 0;

    //cv::Mat debug(height[lvl], width[lvl], CV_32FC1, 0.0);

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            //std::cout << "pixel: " << y << " " << x << std::endl;

            uchar vkf = keyframeMat[lvl].at<uchar>(y,x);
            float keyframeId = idepthMat[lvl].at<float>(y,x);

            //std::cout << "vkf " << vkf << " id " << id << std::endl;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(fxinv[lvl]*x + cxinv[lvl], fyinv[lvl]*y + cyinv[lvl],1.0)/keyframeId;
            Eigen::Vector3f pointFrame = framePose*poinKeyframe;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = Eigen::Vector3f((fx[lvl]*pointFrame(0) + cx[lvl])/pointFrame(2), (fy[lvl]*pointFrame(1) + cy[lvl])/pointFrame(2), 1.0);

            if(pixelFrame(0) < 0.0 || pixelFrame(0) > width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) > height[lvl])
                continue;

            uchar vf = frame.at<uchar>(pixelFrame(1), pixelFrame(0));

            float res = (vkf-vf);

            //std::cout << "pixel " << " " << float(vkf) << " " << float(vf) << " res " << res << std::endl;

            residual += res*res;
            num++;

            //std::cout << "accres " << residual << std::endl;

            //debug.at<float>(y,x) = abs(res)*0.01;
        }

    //cv::namedWindow("calcResidual debug", cv::WINDOW_NORMAL);
    //cv::imshow("calcResidual debug", debug);
    //cv::waitKey(30);

    return residual/num;
}

Sophus::SE3f mesh_vo::calcPose(unsigned int _frameTexture, unsigned int _frameDerivativeTexture, Sophus::SE3f initialGuessPose)
{

    /*
    Sophus::SE3f pose;
    calcIdepth(pose, 0);
    glBindTexture(GL_TEXTURE_2D, idepthTexture);
    glGenerateMipmap(GL_TEXTURE_2D);

    int lvl = 4;

    glBindTexture(GL_TEXTURE_2D, keyframeTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, keyframe_cpu_data);

    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, frame_cpu_data);

    glBindTexture(GL_TEXTURE_2D, idepthTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, idepth_cpu_data);

    float residual;
    acc_H_pose.setZero();
    acc_J_pose.setZero();

    for(int i = 0; i < 100; i++)
    {
        residual = calcError(keyframeTexture, frameTexture, initialGuessPose, lvl);
        //residual = calcError_CPU(frameTexture, initialGuessPose, lvl);
        //calcHJPose(keyframeTexture,keyframeDerivativeTexture,frameTexture,frameDerivativeTexture,initialGuessPose, lvl);
        //calcHJPose2(keyframeTexture,keyframeDerivativeTexture,frameTexture,frameDerivativeTexture,initialGuessPose, lvl);
    }
     std::cout << "residual " << residual << std::endl;
     std::cout << "J " << acc_J_pose.matrix() << std::endl;
     std::cout << "H " << acc_H_pose.matrix() << std::endl;

    return initialGuessPose;
    */


    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    Sophus::SE3f framePose = initialGuessPose;

    //std::cout << "initial guess " << std::endl;
    //std::cout << framePose.matrix() << std::endl;

    for(int lvl=4; lvl >= 1; lvl--)
    {
        //Sophus::SE3f keyframePose;
        //calcIdepth(keyframePose, lvl);

        float last_error = calcError_CPU(frameMat[lvl],framePose,lvl);
        //float last_error = calcError(_frameTexture, framePose, lvl);

        //showTexture(residualTexture, lvl);
        //showDebug(frameTexture, framePose, 0);
        //std::cout << "pose for lvl " << lvl << std::endl;
        //std::cout << framePose.matrix() << std::endl;
        //std::cout << "lvl " << lvl << " init error " << last_error << std::endl;

        int t_try = 0;
        for(int it = 0; it < maxIterations[lvl]; it++)
        {
            acc_J_pose.setZero();
            acc_H_pose.setZero();

            calcHJPose_CPU(frameMat[lvl], frameDerivativeMat[lvl], framePose ,lvl);
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

                //std::cout << "H " << acc_H_pose << std::endl;
                //std::cout << "J " << acc_J_pose << std::endl;

                inc_pose = acc_H_pose_lambda.ldlt().solve(acc_J_pose);

                //Sophus::SE3f new_pose = framePose.inverse()*Sophus::SE3f::exp(inc).inverse();
                //Sophus::SE3f new_pose = Sophus::SE3f::exp(inc) * framePose;
                //Sophus::SE3f new_pose = Sophus::SE3f::exp(inc).inverse() * framePose;
                //Sophus::SE3f new_pose = framePose*Sophus::SE3f::exp(inc);
                Sophus::SE3f new_pose = framePose*Sophus::SE3f::exp(inc_pose).inverse();

                //std::cout << "new_pose " << new_pose.matrix() << std::endl;

                //float error = calcError(_frameTexture, new_pose, lvl);
                float error = calcError_CPU(frameMat[lvl],new_pose,lvl);
                //std::cout << "error2 " << error2 << std::endl;
                //std::cout << "lvl " << lvl << " it " << it << " try " << n_try << " lambda " << lambda << " error " << error << std::endl;

                if(error < last_error)
                {
                    //std::cout << "update accepted " << std::endl;
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

                    //std::cout << "good update " << std::endl;
                    //std::cout << framePose.matrix() << std::endl;

                    float p = error / last_error;


                    if(lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

                    last_error = error;

                    if( p >  0.999f)
                    //if(p > 0.9f)
                    {
                        std::cout << "lvl " << lvl << " converged after " << it << " itarations and " << t_try << " total tries, and lambda " << lambda << std::endl;
                        //if converged, do next level
                        it = maxIterations[lvl];
                    }

                    //if update accepted, do next iteration
                    break;
                }
                else
                {
                    n_try++;
                    t_try++;

                    if(lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= std::pow(2.0, n_try);

                    //reject update, increase lambda, use un-updated data
                    //std::cout << "update rejected " << std::endl;

                    if(!(inc_pose.dot(inc_pose) > 1e-8))
                    //if(!(inc.dot(inc) > 1e-6))
                    {
                        std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
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

void mesh_vo::updateMap()
{
    int lvl = 0;

    //std::cout << "lastFrameAdded " << lastFrameAdded << std::endl;

    if(lastFrameAdded < 0)
        return;

    /*
    //float last_error = calcResidual(frameTexture,framePose,lvl);
    //std::cout << "lvl " << lvl << " error " << last_error << std::endl;
    calcHJMap(keyframeTexture, keyframeDerivativeTexture, keyframePose, frameTexture, frameDerivativeTexture, _framePose ,lvl);
    //Sophus::SE3f keyframePose;
    //calcIdepth(_framePose, lvl);
    //float residual1 = calcResidual(frameTexture, framePose, lvl);
    //Sophus::SE3f keyframePose;
    //calcIdepth(framePose, lvl);
    showTexture(d_I_d_p0_Texture, lvl);
    //float residual2 = calcResidual_CPU(frameTexture, framePose, lvl);

    //std::cout << "residual " << residual1 << " " << residual2 << std::endl;
    //showDebug(frameTexture, framePose, lvl2);
    //cv::waitKey(30);

    return;
    */

    float last_error = 0.0;
    for(int i=0; i < MAX_FRAMES; i++)
    {
        last_error += calcError(frameTextureStack[i], framePoseStack[i], lvl);
    }
    //showTexture(residualTexture, lvl);

    std::cout << "lvl " << lvl << " init error " << last_error << std::endl;

    int maxIterations = 10;

    for(int it = 0; it < maxIterations; it++)
    {
        acc_H_map.setZero();
        acc_J_map.setZero();
        inc_map.setZero();

        for(int i = 0; i < MAX_FRAMES; i++)
            calcHJMap(frameTextureStack[i], frameDerivativeTextureStack[i], framePoseStack[i], lvl);
        //showTexture(d_I_d_p0_Texture, lvl);

        float lambda = 0.0;
        int n_try = 0;
        while(true)
        {
            Eigen::MatrixXf acc_H_map_lambda = acc_H_map;

            for(int j = 0; j < acc_H_map_lambda.rows(); j++)
                if(acc_H_map_lambda(j,j) == 0.0)
                    acc_H_map_lambda(j,j) = 1.0;
                else
                    acc_H_map_lambda(j,j) *= 1.0+lambda;

            //inc = -acc_H_map_lambda.ldlt().solve(acc_J_map);
            inc_map = -acc_J_map/(1.0*(1.0+lambda));

            for(int index=0; index < int(scene_vertices.size()); index++)
            {
                scene_vertices_updated[index] = scene_vertices[index] + inc_map(index);
            }

            glBindVertexArray(scene_VAO);
            glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices_updated.size(), scene_vertices_updated.data());

            float error = 0.0;
            for(int i=0; i < MAX_FRAMES; i++)
            {
                error += calcError(frameTextureStack[i], framePoseStack[i], lvl);
            }

            std::cout << "lvl " << lvl << " it " << it << " try " << n_try << " lambda " << lambda << " error " << error << std::endl;

            if(error < last_error)
            {
                std::cout << "update accepted " << std::endl;

                //accept update, decrease lambda
                scene_vertices = scene_vertices_updated;

                float p = error / last_error;

                if(lambda < 0.2f)
                    lambda = 0.0f;
                else
                    lambda *= 0.5;

                last_error = error;

                if( p >  0.999f)
                {
                    //std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
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
                    lambda *= std::pow(2.0, n_try);

                glBindVertexArray(scene_VAO);
                glBindBuffer(GL_ARRAY_BUFFER, scene_VBO);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*scene_vertices.size(), scene_vertices.data());

                //reject update, increase lambda, use un-updated data
                std::cout << "update rejected " << std::endl;

                if(!(inc_map.dot(inc_map) > 1e-16))
                {
                    //std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations with lambda " << lambda << std::endl;
                    //if too small, do next level!
                    it = maxIterations;
                    break;
                }
            }
        }
    }
}


void mesh_vo::calcHJPose(unsigned int _frameTexture, unsigned int _frameDerTexture, Sophus::SE3f framePose, int lvl)
{   
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, residualTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, traTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, rotTexture, lvl);

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
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frameTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _frameDerTexture);

    // activate shader
    calcJShader.use();

    calcJShader.setMat4("framePose", eigen2glm_mat4(framePose.matrix()));
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
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, JposeTexture1, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, JposeTexture2, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, JposeTexture3, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, JposeTexture4, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, JposeTexture5, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, JposeTexture6, new_lvl[lvl]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, JposeTexture7, new_lvl[lvl]);

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
    glBindTexture(GL_TEXTURE_2D, residualTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, traTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, rotTexture);

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

    reduceVec4(JposeTexture1, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, j_pose_data1);
    reduceVec4(JposeTexture2, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, j_pose_data2);
    reduceVec4(JposeTexture3, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, j_pose_data3);
    reduceVec4(JposeTexture4, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, j_pose_data4);
    reduceVec4(JposeTexture5, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, j_pose_data5);
    reduceVec4(JposeTexture6, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, j_pose_data6);
    reduceVec4(JposeTexture7, new_lvl[lvl], new_lvl2[lvl]);
    glGetTexImage(GL_TEXTURE_2D, new_lvl2[lvl], GL_RGBA, GL_FLOAT, j_pose_data7);


    int count = 0;
    for(int index = 0; index < width[new_lvl2[lvl]]*height[new_lvl2[lvl]]; index+=1)
    {
        /*
            if(residual == 0.0f)
                continue;

            if(J_pose.norm() == 0.0f)
                continue;
            */

        Eigen::Matrix<float, 6, 1> J_pose;
        J_pose << j_pose_data1[index*4], j_pose_data1[index*4+1], j_pose_data1[index*4+2], j_pose_data1[index*4+3], j_pose_data2[index*4], j_pose_data2[index*4+1];

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
void mesh_vo::reduceFloat(unsigned int texture, int src_lvl, int dst_lvl)
{
    /*
    //reduce by mipmap generation
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, src_lvl);
    glGenerateMipmap(GL_TEXTURE_2D);
    */


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

void mesh_vo::reduceVec4(unsigned int texture, int src_lvl, int dst_lvl)
{

    //reduce by mipmap generation
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, src_lvl);
    glGenerateMipmap(GL_TEXTURE_2D);


    /*
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
    */
}

void mesh_vo::copyTexture(unsigned int srcTexture, unsigned int dstTexture, int lvl)
{
    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, dstTexture, lvl);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! calcResidual" << std::endl;

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, srcTexture);

    copyShader.use();

    glBindVertexArray(frame_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void mesh_vo::calcHJPose2(unsigned int _frameTexture, unsigned int _frameDerTexture, Sophus::SE3f framePose, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    int new_lvl[MAX_LEVELS] = {9,9,9,9,9};

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, JposeTexture1, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, JposeTexture2, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, JposeTexture3, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, JposeTexture4, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, JposeTexture5, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, JposeTexture6, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, JposeTexture7, lvl);

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
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frameTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _frameDerTexture);

    // activate shader
    calcHJShader2.use();

    calcHJShader2.setMat4("framePose", eigen2glm_mat4(framePose.matrix()));
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


    glBindTexture(GL_TEXTURE_2D, JposeTexture1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGetTexImage(GL_TEXTURE_2D, new_lvl[lvl], GL_RGBA, GL_FLOAT, j_pose_data1);

    glBindTexture(GL_TEXTURE_2D, JposeTexture2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGetTexImage(GL_TEXTURE_2D, new_lvl[lvl], GL_RGBA, GL_FLOAT, j_pose_data2);

    glBindTexture(GL_TEXTURE_2D, JposeTexture3);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGetTexImage(GL_TEXTURE_2D, new_lvl[lvl], GL_RGBA, GL_FLOAT, j_pose_data3);

    glBindTexture(GL_TEXTURE_2D, JposeTexture4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGetTexImage(GL_TEXTURE_2D, new_lvl[lvl], GL_RGBA, GL_FLOAT, j_pose_data4);

    glBindTexture(GL_TEXTURE_2D, JposeTexture5);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGetTexImage(GL_TEXTURE_2D, new_lvl[lvl], GL_RGBA, GL_FLOAT, j_pose_data5);

    glBindTexture(GL_TEXTURE_2D, JposeTexture6);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGetTexImage(GL_TEXTURE_2D, new_lvl[lvl], GL_RGBA, GL_FLOAT, j_pose_data6);

    glBindTexture(GL_TEXTURE_2D, JposeTexture7);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, lvl);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGetTexImage(GL_TEXTURE_2D, new_lvl[lvl], GL_RGBA, GL_FLOAT, j_pose_data7);

    for(int index = 0; index < width[new_lvl[lvl]]*height[new_lvl[lvl]]; index+=1)//2*MAX_LEVELS/(lvl+1))
    {
        /*
        if(residual == 0.0f)
            continue;

        if(J_pose.norm() == 0.0f)
            continue;
        */

        Eigen::Matrix<float, 6, 1> J_pose;
        J_pose << j_pose_data1[index*4], j_pose_data1[index*4+1], j_pose_data1[index*4+2], j_pose_data1[index*4+3], j_pose_data2[index*4], j_pose_data2[index*4+1];

        if(J_pose.norm() == 0.0f)
            continue;

        acc_J_pose += J_pose;

        /*
        acc_J_pose(0) += j_pose_data1[index*4];
        acc_J_pose(1) += j_pose_data1[index*4+1];
        acc_J_pose(2) += j_pose_data1[index*4+2];
        acc_J_pose(3) += j_pose_data1[index*4+3];
        acc_J_pose(4) += j_pose_data2[index*4];
        acc_J_pose(5) += j_pose_data2[index*4+1];
        */

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

void mesh_vo::calcHJPose_CPU(cv::Mat frame, cv::Mat frameDer, Sophus::SE3f framePose, int lvl)
{
    acc_J_pose.setZero();
    acc_H_pose.setZero();

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            //std::cout << 1 << std::endl;
            uchar vkf = keyframeMat[lvl].at<uchar>(y,x);
            float keyframeId = idepthMat[lvl].at<float>(y,x);
//std::cout << 2 << std::endl;
            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(fxinv[lvl]*x + cxinv[lvl],fyinv[lvl]*y + cyinv[lvl],1.0)/keyframeId;
            Eigen::Vector3f pointFrame = framePose*poinKeyframe;
//std::cout << 3 << std::endl;
            if(pointFrame(2) <= 0.0)
                continue;
//std::cout << 4 << std::endl;
            Eigen::Vector3f pixelFrame = Eigen::Vector3f((fx[lvl]*pointFrame(0) + cx[lvl])/pointFrame(2), (fy[lvl]*pointFrame(1) + cy[lvl])/pointFrame(2), 1.0);
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

            float v0 = d_f_d_uf(0) * fx[lvl] * id;
            float v1 = d_f_d_uf(1) * fy[lvl] * id;
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

void mesh_vo::calcHJMap(unsigned int _frameTexture, unsigned int _frameDerTexture, Sophus::SE3f framePose, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vertexID_Texture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, primitiveID_Texture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, residualTexture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, d_I_d_p0_Texture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, d_I_d_p1_Texture, lvl);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, d_I_d_p2_Texture, lvl);

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
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, keyframeTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frameTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _frameDerTexture);

    // activate shader
    calcHJMapShader.use();

    calcHJMapShader.setMat4("framePose", eigen2glm_mat4(framePose.matrix()));
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

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, 0, 0);

    glBindTexture(GL_TEXTURE_2D, vertexID_Texture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RGB, GL_FLOAT, vertexID_cpu_data);

    glBindTexture(GL_TEXTURE_2D, primitiveID_Texture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, primitiveID_cpu_data);

    glBindTexture(GL_TEXTURE_2D, residualTexture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, residual_cpu_data);

    glBindTexture(GL_TEXTURE_2D, d_I_d_p0_Texture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, d_I_d_p0_cpu_data);

    glBindTexture(GL_TEXTURE_2D, d_I_d_p1_Texture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, d_I_d_p1_cpu_data);

    glBindTexture(GL_TEXTURE_2D, d_I_d_p2_Texture);
    glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, d_I_d_p2_cpu_data);

    for(int index = 0; index < height[lvl]*width[lvl]; index+=10)
    {
        //std::cout << "pixel index " << index << std::endl;
        //acumular para cada pixel
        float error = residual_cpu_data[index];

        if(error != error)
        {
            //std::cout << "nand in error " << error << std::endl;
            continue;
        }

        //cada pixel aporta 9 parametros
        //81 coeficientes en H! (pero es simetrico, osea son: 9 diagonales y 8+7+6+5+4+3+2+1(36) no diagonales)
        int vertexID[3];
        vertexID[0] = int(vertexID_cpu_data[index*3]);
        vertexID[1] = int(vertexID_cpu_data[index*3+1]);
        vertexID[2] = int(vertexID_cpu_data[index*3+2]);
        if(vertexID[0] < 0 || vertexID[1] < 0 || vertexID[2] < 0)
            continue;

        //std::cout << "updating vertex " << vertexID[0] << " " << vertexID[1] << " " << vertexID[2] << std::endl;
        //std::cout << "vertices " << vertexID[0] << " " << vertexID[1] << " " << vertexID[2] << std::endl;
        //std::cout << "error " << error << std::endl;

        //std::cout << "incides " << std::endl;
        //for(int i = 0; i < 9; i++)
        //    std::cout << indices[i] << std::endl;

        float J[3];
        J[0] = d_I_d_p0_cpu_data[index];
        J[1] = d_I_d_p1_cpu_data[index];
        J[2] = d_I_d_p2_cpu_data[index];

        for(int i = 0; i < 3; i++)
        {
            if(J[i]!=J[i])
            {
               //printf("J nand %d %d %d %d %d\n", i, index, vertexID[0], vertexID[1], vertexID[2]);
               continue;
            }
        }

        //ahora si, actualizo las matrices usando los indices de cada vertice
        for(int i = 0; i < 3; i++)
        {
            acc_J_map(vertexID[i]*3+2) += J[i]*error;
            /*
            for(int j = 0; j < 3; j++)
            {
                acc_H_map(vertexID[i]*3+2, vertexID[j]*3+2) += J[i]*J[j];
            }
            */
        }
    }
}

void mesh_vo::frameDerivative(unsigned int frame, unsigned int frameDerivative, int lvl)
{
    //calculate frame derivative
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
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

    /*
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
        */
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    //glClearColor(-1.0f, -1.0f, -1.0f, -1.0f);
    //glClear(GL_COLOR_BUFFER_BIT);
    // bind textures on corresponding texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, frame);
    // activate shader
    frameDerivativeShader.use();
    frameDerivativeShader.setFloat("dx", dx[lvl]);
    frameDerivativeShader.setFloat("dy", dy[lvl]);
    // render frame
    glBindVertexArray(frame_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
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

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
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

void mesh_vo::calcIdepth(Sophus::SE3f framePose, int lvl)
{
    //glfwMakeContextCurrent(frameWindow);

    glViewport(0,0,width[lvl],height[lvl]);

    //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, idepthTexture, lvl);

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
    idepthShader.setMat4("framePose", eigen2glm_mat4(framePose.matrix()));
    idepthShader.setMat4("opencv2opengl", opencv2opengl);
    idepthShader.setMat4("projection", projMat[lvl]);

    glBindVertexArray(scene_VAO);
    glDrawElements(GL_TRIANGLES, scene_indices.size(), GL_UNSIGNED_INT, 0);
}

/*
float mesh_vo::calcOccupancy(Sophus::SE3f framePose, int lvl)
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

float mesh_vo::calcOccupancy_CPU(Sophus::SE3f framePose, int lvl)
{
    int count = 0;

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            float keyframeId = idepthMat[lvl].at<float>(y,x);
            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(fxinv[lvl]*x + cxinv[lvl],fyinv[lvl]*y + cyinv[lvl],1.0)/keyframeId;
            Eigen::Vector3f pointFrame = framePose*poinKeyframe;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = Eigen::Vector3f((fx[lvl]*pointFrame(0) + cx[lvl])/pointFrame(2), (fy[lvl]*pointFrame(1) + cy[lvl])/pointFrame(2), 1.0);

            if(pixelFrame(0) < 0.0 || pixelFrame(0) >= width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= height[lvl])
                continue;
            count++;
        }

    return float(count)/(width[lvl]*height[lvl]);
}


void mesh_vo::visual_odometry(cv::Mat _frame)
{
    glfwMakeContextCurrent(frameWindow);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    //save frame in gpu memory, calc derivavites y mipmaps  
    cv::Mat frameResized;
    cv::Mat frameFlipped;

    cv::resize(_frame,frameResized,cv::Size(width[0], height[0]));//resize image

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(frameResized, frameMat[lvl], cv::Size(width[lvl],height[lvl]),0,0,cv::INTER_AREA);

        cv::flip(frameMat[lvl], frameFlipped, 0);
        glBindTexture(GL_TEXTURE_2D, frameTexture);
        glTexSubImage2D(GL_TEXTURE_2D, lvl, 0, 0, width[lvl], height[lvl], GL_RED, GL_UNSIGNED_BYTE, frameFlipped.data);

        frameDerivative_CPU(frameMat[lvl], frameDerivativeMat[lvl], lvl);

        cv::flip(frameDerivativeMat[lvl], frameFlipped, 0);
        glBindTexture(GL_TEXTURE_2D, frameDerivativeTexture);
        glTexSubImage2D(GL_TEXTURE_2D, lvl, 0, 0, width[lvl], height[lvl], GL_RED, GL_UNSIGNED_BYTE, frameFlipped.data);
    }



    //copyTexture(frameTexture, keyframeTexture, 0);
    //changeKeyframe(frameTexture, trackedPose);
    //glBindTexture(GL_TEXTURE_2D, keyframeTexture);
    //glGenerateMipmap(GL_TEXTURE_2D);
    //showTexture(keyframeTexture, 0);
    //return;

    tic_toc t;
    t.tic();
    trackedPose = calcPose(frameTexture, frameDerivativeTexture, trackedPose*Sophus::SE3f::exp(inc_pose).inverse());
    glFinish();
    if(calcPoseTime == 0.0)
        calcPoseTime = t.toc();
    else
        calcPoseTime = calcPoseTime*0.9 + t.toc()*0.1;
    std::cout << "clacPose time " << calcPoseTime << std::endl;

    float new_occupancy = calcOccupancy_CPU(trackedPose, 1);

    std::cout << "occ " << new_occupancy << std::endl;

    float diff = occupancy - new_occupancy;
    occupancy = new_occupancy;

    if(diff > 0.001)
    {
        std::cout << "sup diff " << diff << " add frame and update map" << std::endl;
        addFrameToStack(frameTexture, frameDerivativeTexture, trackedPose);
        updateMap();
        for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
          calcIdepth(Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()), lvl);
          glBindTexture(GL_TEXTURE_2D, idepthTexture);
          glGetTexImage(GL_TEXTURE_2D, lvl, GL_RED, GL_FLOAT, idepthMat[lvl].data);
          cv::flip(idepthMat[lvl],idepthMat[lvl],0);
        }
    }

    if(occupancy < 0.95)
    {
        std::cout << "change keyframe " << std::endl;
        changeKeyframe(frameTexture, trackedPose);

        keyframePose = trackedPose*keyframePose;

        //calcIdepth(Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()),2);
        //showTexture(idepthTexture, 2);

        for(int i = 0; i < MAX_FRAMES; i++)
            framePoseStack[i] = framePoseStack[i]*trackedPose.inverse();

        trackedPose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());
        occupancy = 1.0;
    }
}


void mesh_vo::frameDerivative_CPU(cv::Mat frame, cv::Mat &frameDer, int lvl)
{
    frameDer.setTo(0.0);

    for(int y = 1; y < height[lvl]-1; y++)
        for(int x = 1; x < width[lvl]-1; x++)
        {
            cv::Vec2f d;
            d.val[0] = (frame.at<uchar>(y,x+1) - frame.at<uchar>(y,x-1))/2.0;
            d.val[1] = (frame.at<uchar>(y+1,x) - frame.at<uchar>(y-1,x))/2.0;

            frameDer.at<cv::Vec2f>(y,x) = d;
        }
}

