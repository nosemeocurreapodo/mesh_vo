#include <pangolin/pangolin.h>

#include "common/types.h"
#include "common/camera.h"

const char *tplotter_vertex_shader = R"Shader(
    #version 330 core
    layout (location = 0) in vec3 a_position;
    uniform mat4 MVP;

    void main() {
        gl_Position = MVP * vec4(a_position, 1.0);
    }
    )Shader";

const char *tplotter_fragment_shader = R"Shader(
    #version 330 core
    layout(location = 0) out vec4 f_color;
    
    void main() {
        float color = 1.0;
        f_color = vec4(color, color, color, 1.0);
    }
    )Shader";

class trayectoryPlotter
{
public:
    trayectoryPlotter()
    {
        indicesSize = 0;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glGenTextures(1, &textureID);
    }

    void setBuffers(std::vector<SE3f> globalPoses, cameraType cam)
    {
        // 5 points for each pose, to draw the frustrum of the camera
        int vertices_size = globalPoses.size() * 5 * 3;
        float vertex_buffer[vertices_size];
        // for each camera, draw 8 lines
        int indices_size = globalPoses.size() * 8 * 2;
        unsigned int indices_buffer[indices_size];

        float distance = 0.1;

        for (int i = 0; i < globalPoses.size(); i++)
        {
            //pose goes from world to local 
            SE3f pose = globalPoses[i].inverse();

            vec3f ray1 = pose * (cam.pixToRay(vec2f(0.0, 0.0)) * distance);
            vec3f ray2 = pose * (cam.pixToRay(vec2f(1.0, 0.0)) * distance);
            vec3f ray3 = pose * (cam.pixToRay(vec2f(1.0, 1.0)) * distance);
            vec3f ray4 = pose * (cam.pixToRay(vec2f(0.0, 1.0)) * distance);

            vertex_buffer[i * 5 * 3 + 0 * 3 + 0] = pose.translation()(0);
            vertex_buffer[i * 5 * 3 + 0 * 3 + 1] = pose.translation()(1);
            vertex_buffer[i * 5 * 3 + 0 * 3 + 2] = pose.translation()(2);
            vertex_buffer[i * 5 * 3 + 1 * 3 + 0] = ray1(0);
            vertex_buffer[i * 5 * 3 + 1 * 3 + 1] = ray1(1);
            vertex_buffer[i * 5 * 3 + 1 * 3 + 2] = ray1(2);
            vertex_buffer[i * 5 * 3 + 2 * 3 + 0] = ray2(0);
            vertex_buffer[i * 5 * 3 + 2 * 3 + 1] = ray2(1);
            vertex_buffer[i * 5 * 3 + 2 * 3 + 2] = ray2(2);
            vertex_buffer[i * 5 * 3 + 3 * 3 + 0] = ray3(0);
            vertex_buffer[i * 5 * 3 + 3 * 3 + 1] = ray3(1);
            vertex_buffer[i * 5 * 3 + 3 * 3 + 2] = ray3(2);
            vertex_buffer[i * 5 * 3 + 4 * 3 + 0] = ray4(0);
            vertex_buffer[i * 5 * 3 + 4 * 3 + 1] = ray4(1);
            vertex_buffer[i * 5 * 3 + 4 * 3 + 2] = ray4(2);

            indices_buffer[i * 8 * 2 + 0] = i * 5 * 3 + 0;
            indices_buffer[i * 8 * 2 + 1] = i * 5 * 3 + 1;
            indices_buffer[i * 8 * 2 + 2] = i * 5 * 3 + 0;
            indices_buffer[i * 8 * 2 + 3] = i * 5 * 3 + 2;
            indices_buffer[i * 8 * 2 + 4] = i * 5 * 3 + 0;
            indices_buffer[i * 8 * 2 + 5] = i * 5 * 3 + 3;
            indices_buffer[i * 8 * 2 + 6] = i * 5 * 3 + 0;
            indices_buffer[i * 8 * 2 + 7] = i * 5 * 3 + 4;

            indices_buffer[i * 8 * 2 + 8] = i * 5 * 3 + 1;
            indices_buffer[i * 8 * 2 + 9] = i * 5 * 3 + 2;
            indices_buffer[i * 8 * 2 + 10] = i * 5 * 3 + 2;
            indices_buffer[i * 8 * 2 + 11] = i * 5 * 3 + 3;
            indices_buffer[i * 8 * 2 + 12] = i * 5 * 3 + 3;
            indices_buffer[i * 8 * 2 + 13] = i * 5 * 3 + 4;
            indices_buffer[i * 8 * 2 + 14] = i * 5 * 3 + 4;
            indices_buffer[i * 8 * 2 + 15] = i * 5 * 3 + 1;
        }

        // Bind the VAO so that our buffer and attribute settings are stored in it.
        glBindVertexArray(VAO);

        // VBO: Upload the vertex data to the GPU.
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices_size * sizeof(float),
                     vertex_buffer, GL_STATIC_DRAW);

        // EBO: Upload the index data to the GPU.
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size * sizeof(unsigned int),
                     indices_buffer, GL_STATIC_DRAW);

        // Vertex Attribute: We assume each vertex only has a 3D position.
        // We tell OpenGL that the attribute at location 0 (in our shader) should
        // get its data from the currently bound GL_ARRAY_BUFFER (VBO).
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);

        // Unbind the VAO (optional for clarity, as we'll bind it later)
        glBindVertexArray(0);

        indicesSize = indices_size;
    }

    void compileShaders()
    {
        // Build and compile our shader program
        // ------------------------------------

        // Create a vertex shader GL object
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);

        // Tell GL the source code to use
        glShaderSource(vertexShader, 1, &tplotter_vertex_shader, NULL);

        // Actually compile the program
        glCompileShader(vertexShader);

        // Check if the compilation was successfull, and print anyn errors
        int success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n"
                      << infoLog << std::endl;
        }

        // Repeat for the Fragment shader
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &tplotter_fragment_shader, NULL);
        glCompileShader(fragmentShader);
        // check for shader compile errors
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n"
                      << infoLog << std::endl;
        }

        // Link the vertex and fragment shaders into one complete program
        shaderProgram = glCreateProgram();

        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        // Check for linking errors
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
                      << infoLog << std::endl;
        }

        // Delete the now unused shader objects
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        mvpLoc = glGetUniformLocation(shaderProgram, "MVP");
        timeLoc = glGetUniformLocation(shaderProgram, "u_time");
        imageLoc = glGetUniformLocation(shaderProgram, "image");
    }

    void draw(pangolin::OpenGlMatrix mvp)
    {
        glUseProgram(shaderProgram);
        GLfloat mvp_float[16];
        for (int i = 0; i < 16; ++i)
        {
            mvp_float[i] = static_cast<GLfloat>(mvp.m[i]);
        }
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, mvp_float);

        glBindVertexArray(VAO);
        glDrawElements(GL_LINES, static_cast<GLsizei>(indicesSize), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        glUseProgram(0);
    }

private:
    GLuint VAO, VBO, EBO;
    GLuint textureID;
    unsigned int shaderProgram;
    GLint mvpLoc;
    GLint timeLoc;
    GLint imageLoc;
    int indicesSize;
};