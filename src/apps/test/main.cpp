#include <iostream>

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pangolin/pangolin.h>

#include "utils/convertAhandaPovRayToStandard.h"
#include "common/camera.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "visualOdometry.h"

inline bool fileExist(const std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

std::mutex pose_mutex;
std::mutex map_mutex;
geometryType mapGeometry;
bool geometryUpdated = false;

inline void glDrawColouredCube(GLfloat axis_min = -0.5f, GLfloat axis_max = +0.5f)
{
    const GLfloat l = axis_min;
    const GLfloat h = axis_max;

    const GLfloat verts[] = {
        l, l, h, h, l, h, l, h, h, h, h, h, // FRONT
        l, l, l, l, h, l, h, l, l, h, h, l, // BACK
        l, l, h, l, h, h, l, l, l, l, h, l, // LEFT
        h, l, l, h, h, l, h, l, h, h, h, h, // RIGHT
        l, h, h, h, h, h, l, h, l, h, h, l, // TOP
        l, l, h, l, l, l, h, l, h, h, l, l  // BOTTOM
    };

    glVertexPointer(3, GL_FLOAT, 0, verts);
    glEnableClientState(GL_VERTEX_ARRAY);

    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDrawArrays(GL_TRIANGLE_STRIP, 4, 4);

    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, 8, 4);
    glDrawArrays(GL_TRIANGLE_STRIP, 12, 4);

    glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, 16, 4);
    glDrawArrays(GL_TRIANGLE_STRIP, 20, 4);

    glDisableClientState(GL_VERTEX_ARRAY);
}

class geometryPlotter
{
public:
    geometryPlotter()
    {
        indicesSize = 0;

        // ----------------------------
        // Set Up OpenGL Buffers (VAO/VBO/EBO)
        // ----------------------------
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
    }

    void setBuffers(geometryType &geometry)
    {
        int vertices_size = geometry.getVerticesIds().size();
        float vertices_buffer[vertices_size * 3];
        for (int i = 0; i < vertices_size; i++)
        {
            vec3f ver = geometry.getVertex(i).ver;
            vertices_buffer[i * 3 + 0] = ver(0);
            vertices_buffer[i * 3 + 1] = ver(1);
            vertices_buffer[i * 3 + 2] = ver(2);
        }

        int indices_size = geometry.getShapesIds().size();
        int indices_buffer[indices_size * 3];
        for (int i = 0; i < indices_size; i++)
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
                     vertices_buffer, GL_STATIC_DRAW);

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
    };

    void draw()
    {
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indicesSize), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

private:
    GLuint VAO, VBO, EBO;
    int indicesSize;
};

// Visualization thread using Pangolin
int visualizationThread()
{
    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    geometryPlotter geomPlotter;

    while (!pangolin::ShouldQuit())
    {
        {
            std::lock_guard<std::mutex> lock(map_mutex);
            if (geometryUpdated)
            {
                geomPlotter.setBuffers(mapGeometry);
                geometryUpdated = false;
            }
        }

        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // Render OpenGL Cube
        // pangolin::glDrawColouredCube();
        geomPlotter.draw();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "usage: " << argv[0] << " /path/to/dataset" << std::endl;
        return 0;
    }

    std::thread visThread(visualizationThread);

    std::string dataset_path = argv[1];

    int frameNumber = 0;
    int frameCounterDirection = 1;

    int framesTracked = 0;

    cv::Mat imageMat = cv::imread(dataset_path + "images/scene_000.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat idepthMat = cv::imread(dataset_path + "depths/scene_000.png", cv::IMREAD_GRAYSCALE);
    Sophus::SE3f initPose = readPose(dataset_path + "poses/scene_000.txt");

    imageMat.convertTo(imageMat, CV_32FC1);

    int width = imageMat.cols;
    int height = imageMat.rows;

    float fx = height;     // 481.20;
    float fy = height;     // 480.0;
    float cx = width / 2;  // 319.5;
    float cy = height / 2; // 239.5;

    cameraType cam(fx, fy, cx, cy, width, height);

    // to avoid idepth = 0 in the data
    // idepthMat = idepthMat + 1.0;
    // idepthMat.convertTo(idepthMat, CV_32FC1);
    // cv::resize(idepthMat, idepthMat, cv::Size(cam.width, cam.height), cv::INTER_AREA);

    dataCPU<float> image(width, height, -1.0);
    // dataCPU<float> idepth(IMAGE_WIDTH, IMAGE_HEIGHT, -1.0);

    image.set((float *)imageMat.data);
    // idepth.set((float*)idepthMat.data);

    /*
    //get corner from image
    int maxCorners = MESH_WIDTH*MESH_HEIGHT;
    std::vector<cv::Point2f> corners;
    double qualityLevel = 0.000000001;
    double minDistance = 20;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    cv::goodFeaturesToTrack( idepthMat,
                         corners,
                         maxCorners,
                         qualityLevel,
                         minDistance,
                         cv::Mat(),
                         blockSize,
                         gradientSize,
                         useHarrisDetector,
                         k );

    std::vector<vec2<float>> pixels;
    std::vector<float> idepths;

    for(auto corner : corners)
    {
        vec2<float> pixel(corner.x, corner.y);
        float id = idepth.get(pixel(1), pixel(0), 0);

        pixels.push_back(pixel);
        idepths.push_back(id);
    }
    */

    visualOdometry odometry(cam, width, height);

    // odometry.initScene(image, pixels, idepths, Sophus::SE3f());
    // odometry.init(image, idepth, Sophus::SE3f());
    odometry.init(image, SE3f());

    while (1)
    {
        framesTracked++;
        frameNumber += frameCounterDirection;
        if (frameNumber > 598)
            frameCounterDirection = -1;
        if (frameNumber < 2)
            frameCounterDirection = 1;
        // if(frameNumber > 100)
        //     return 1;

        char image_filename[500];
        char RT_filename[500];

        // file name
        sprintf(image_filename, "images/scene_%03d.png", frameNumber);
        sprintf(RT_filename, "poses/scene_%03d.txt", frameNumber);

        std::string image_path = dataset_path + image_filename;
        std::string pose_path = dataset_path + RT_filename;

        cv::Mat imageMat = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        SE3f realPose = readPose(pose_path) * initPose.inverse();
        // scale the translation, so that the initial map has a mean depth of 1.0
        realPose.translation() = realPose.translation() * 0.25;

        imageMat.convertTo(imageMat, CV_32FC1);
        // cv::resize(imageMat, imageMat, cv::Size(mesh_vo::image_width, mesh_vo::image_height), cv::INTER_AREA);

        image.set((float *)imageMat.data);
        if (odometry.locAndMap(image))
        {
            std::lock_guard<std::mutex> lock(map_mutex);
            mapGeometry = odometry.getKeyframe().getGeometry();
            geometryUpdated = true;
        }
    }

    return 1;
}
