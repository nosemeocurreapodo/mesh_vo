#include <pangolin/pangolin.h>

class BaseRendererGPU
{
public:
    BaseRendererGPU()
    {
    }

protected:
    void compileShaders(const char *vertex_shader, const char *fragment_shader)
    {
        // Build and compile our shader program
        // ------------------------------------

        // Create a vertex shader GL object
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);

        // Tell GL the source code to use
        glShaderSource(vertexShader, 1, &vertex_shader, NULL);

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
        glShaderSource(fragmentShader, 1, &fragment_shader, NULL);
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
    }

    unsigned int shaderProgram;
    GLint mvpLoc;

    const char *vertex_shader;
    const char *fragment_shader;
};

class DepthRendererGPU : public BaseRendererGPU
{
public:
    DepthRendererGPU()
    {
        vertex_shader = R"Shader(
            #version 330 core
            layout (location = 0) in vec3 a_position;
            out float depth;
            uniform mat4 MVP;
        
            void main() {
                gl_Position = MVP * vec4(a_position, 1.0);
                depth = a_position.z;
            }
            )Shader";

        fragment_shader = R"Shader(
            #version 330 core
            layout(location = 0) out float f_color;
            in vec2 v_texcoord;
            in float depth;
            
            void main() {
                f_color = depth;
            }
            )Shader";

        compileShaders(vertex_shader, fragment_shader);
        mvpLoc = glGetUniformLocation(shaderProgram, "MVP");
    }

    void render(keyFrameCPU &kframe, SE3f localPose, dataMipMapCPU<float> &buffer, cameraType cam, int lvl)
    {
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, buffer.get(lvl), lvl);

        mat4f viewMatrix = cam.getProjectiveMatrix() * localPose;

        glUseProgram(shaderProgram);
        GLfloat mvp_float[16];
        for (int i = 0; i < 16; ++i)
        {
            mvp_float[i] = static_cast<GLfloat>(viewMatrix[i]);
        }
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, mvp_float);

        // glBindVertexArray(VAO);
        glBindVertexArray(kframe.geometry.VAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(kframe.geometry.size()), GL_UNSIGNED_INT, 0);
        // glDrawElements(GL_LINE_STRIP, static_cast<GLsizei>(indicesSize), GL_UNSIGNED_INT, 0);
        // glDrawArrays(GL_LINE_STRIP, 0, verticesSize);
        glBindVertexArray(0);
        glUseProgram(0);
    }

private:
};