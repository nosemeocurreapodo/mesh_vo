#include <pangolin/pangolin.h>

const char *iplotter_vertex_shader = R"Shader(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 v_texcoords;

    void main()
    {
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
        v_texcoords = aTexCoords;
    }
    )Shader";

const char *iplotter_fragment_shader = R"Shader(
    #version 330 core
    layout(location = 0) out vec4 f_color;
    in vec2 v_texcoords;
    uniform sampler2D image;
    
    void main() {
        float color = texture(image, v_texcoords).x;
        //color /= 255.0;
        f_color = vec4(color, color, color, 1.0);
    }
    )Shader";

class imagePlotter
{
public:
    imagePlotter(int x, int y)
    {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        // configure frame vertex
        glBindVertexArray(VAO);

        float x_size = 0.5;
        float y_size = 0.5;

        float left = -1.0 + x * x_size; //-1.0;
        float right = left + x_size;    // 1.0;
        float bottom = -1.0 + y * y_size;
        float top = bottom + y_size; // 1.0;

        std::vector<float> frame_vertices;
        frame_vertices.push_back(left);
        frame_vertices.push_back(top);
        frame_vertices.push_back(0.0);
        frame_vertices.push_back(0.0);

        frame_vertices.push_back(left);
        frame_vertices.push_back(bottom);
        frame_vertices.push_back(0.0);
        frame_vertices.push_back(1.0);

        frame_vertices.push_back(right);
        frame_vertices.push_back(bottom);
        frame_vertices.push_back(1.0);
        frame_vertices.push_back(1.0);

        frame_vertices.push_back(left);
        frame_vertices.push_back(top);
        frame_vertices.push_back(0.0);
        frame_vertices.push_back(0.0);

        frame_vertices.push_back(right);
        frame_vertices.push_back(bottom);
        frame_vertices.push_back(1.0);
        frame_vertices.push_back(1.0);

        frame_vertices.push_back(right);
        frame_vertices.push_back(top);
        frame_vertices.push_back(1.0);
        frame_vertices.push_back(0.0);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, frame_vertices.size() * sizeof(float), frame_vertices.data(), GL_STATIC_DRAW);

        // position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);
        // texture coord attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glGenTextures(1, &textureID);
    }

    void setBuffers(dataCPU<float> &image)
    {
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        GLint internalFormat = GL_R32F;
        GLenum openglImageType = GL_FLOAT;
        /*
        if(std::is_same<imageType, float>::value)
        {
            internalFormat = GL_R32F;
            openglImageType = GL_FLOAT;
        }
        else if(std::is_same<imageType, unsigned char>::value)
        {
            internalFormat = GL_R8;
            openglImageType = GL_UNSIGNED_BYTE;
        }
        else if(std::is_same<imageType, int>::value)
        {
            internalFormat = GL_R32I;
            openglImageType = GL_INT;
        }
        */

        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, image.width, image.height, 0, GL_RED, openglImageType, image.get());
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    void compileShaders()
    {
        // Build and compile our shader program
        // ------------------------------------

        // Create a vertex shader GL object
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);

        // Tell GL the source code to use
        glShaderSource(vertexShader, 1, &iplotter_vertex_shader, NULL);

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
        glShaderSource(fragmentShader, 1, &iplotter_fragment_shader, NULL);
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

        imageLoc = glGetUniformLocation(shaderProgram, "image");
    }

    void draw()
    {
        glUseProgram(shaderProgram);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(imageLoc, 0);

        glBindVertexArray(VAO);
        // glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indicesSize), GL_UNSIGNED_INT, 0);
        // glDrawElements(GL_LINE_STRIP, static_cast<GLsizei>(indicesSize), GL_UNSIGNED_INT, 0);
        glDrawArrays(GL_TRIANGLES, 0, 4 * 6);
        glBindVertexArray(0);
        glUseProgram(0);
    }

private:
    GLuint VAO, VBO;
    GLuint textureID;
    unsigned int shaderProgram;
    GLint imageLoc;
};