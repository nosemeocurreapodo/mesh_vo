#include <pangolin/pangolin.h>

const char *gplotter_vertex_shader = R"Shader(
    #version 330 core
    layout (location = 0) in vec3 a_position;
    layout (location = 1) in vec2 a_texcoord;
    out vec2 v_texcoord;
    out float depth;
    uniform mat4 MVP;

    void main() {
        gl_Position = MVP * vec4(a_position, 1.0);
        v_texcoord = a_texcoord;
        depth = a_position.z;
    }
    )Shader";

const char *gplotter_fragment_shader = R"Shader(
    #version 330 core
    layout(location = 0) out vec4 f_color;
    in vec2 v_texcoord;
    in float depth;
    uniform sampler2D image;
    
    void main() {
        //float pattern = sin(10*v_pos.y + u_time) * sin(10*v_pos.x + u_time) * 0.5 + 0.5;
        //vec3 color = mix(colorA, colorB, pattern);
        //float depth = gl_FragCoord.z;
        //vec3 color = vec3(1.0/depth, 1.0/depth, 1.0/depth);
        float color = texture(image, v_texcoord).x;
        color /= 255.0;
        f_color = vec4(color, color, color, 1.0);
    }
    )Shader";


class geometryPlotter
{
public:
    geometryPlotter()
    {
        indicesSize = 0;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glGenTextures(1, &textureID);
    }

    void setBuffers(dataCPU<float> &image, geometryType &geometry)
    {
        int vertices_size = geometry.getVerticesIds().size() * 5; // 3 for the position, 2 for the texcoord
        float vertex_buffer[vertices_size];
        for (int i = 0; i < geometry.getVerticesIds().size(); i++)
        {
            vec3f ver = geometry.getVertex(i).ver;
            vec2f pix = geometry.getVertex(i).pix;

            vertex_buffer[i * 5 + 0] = ver(0);
            vertex_buffer[i * 5 + 1] = ver(1);
            vertex_buffer[i * 5 + 2] = ver(2);

            vertex_buffer[i * 5 + 3] = pix(0);
            vertex_buffer[i * 5 + 4] = pix(1);
        }

        int indices_size = geometry.getShapesIds().size() * 3;
        unsigned int indices_buffer[indices_size];
        for (int i = 0; i < geometry.getShapesIds().size(); i++)
        {
            vec3i ids = geometry.getShape(i).getParamIds();
            indices_buffer[i * 3 + 0] = ids(0);
            indices_buffer[i * 3 + 1] = ids(1);
            indices_buffer[i * 3 + 2] = ids(2);
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
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Unbind the VAO (optional for clarity, as we'll bind it later)
        glBindVertexArray(0);

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, image.width, image.height, 0, GL_RED, GL_FLOAT, image.get());
        glGenerateMipmap(GL_TEXTURE_2D);

        verticesSize = vertices_size;
        indicesSize = indices_size;
    }

    void compileShaders()
    {
        // Build and compile our shader program
        // ------------------------------------

        // Create a vertex shader GL object
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);

        // Tell GL the source code to use
        glShaderSource(vertexShader, 1, &gplotter_vertex_shader, NULL);

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
        glShaderSource(fragmentShader, 1, &gplotter_fragment_shader, NULL);
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

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(glGetUniformLocation(shaderProgram, "image"), 0);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indicesSize), GL_UNSIGNED_INT, 0);
        //glDrawElements(GL_LINE_STRIP, static_cast<GLsizei>(indicesSize), GL_UNSIGNED_INT, 0);
        //glDrawArrays(GL_LINE_STRIP, 0, verticesSize);
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
    int verticesSize;
    int indicesSize;
};