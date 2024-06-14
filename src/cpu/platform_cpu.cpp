#include "platform_cpu.h"

void platformCpu::computeFrameDerivative(frameCpu &frame, camera &cam, int lvl)
{
    for (int y = 1; y < cam.height[lvl] - 1; y++)
        for (int x = 1; x < cam.width[lvl] - 1; x++)
        {
            cv::Vec2f d;
            d.val[0] = (frame.image.texture[lvl].at<uchar>(y, x + 1) - frame.image.texture[lvl].at<uchar>(y, x - 1)) / 2.0;
            d.val[1] = (frame.image.texture[lvl].at<uchar>(y + 1, x) - frame.image.texture[lvl].at<uchar>(y - 1, x)) / 2.0;

            frame.der.texture[lvl].at<cv::Vec2f>(y, x) = d;
        }
}

void platformCpu::computeFrameIdepth(frameCpu &frame, camera &cam, sceneMesh &scene, int lvl)
{
    // for each triangle
    for (std::size_t index = 0; index < scene.scene_indices.size(); index += 3)
    {
        // get its vertices
        // Eigen::Vector3f world_vertex[3];
        Eigen::Vector3f keyframe_vertex[3];
        for (int vertex = 0; vertex < 3; vertex++)
        {
            int vertex_index = scene.scene_indices[index + vertex];

            // scene vertices are vertex in keyframe coordinates
            // keyframe_vertex[vertex](0) = scene.scene_vertices[vertex_index * 3];
            // keyframe_vertex[vertex](1) = scene.scene_vertices[vertex_index * 3 + 1];
            // keyframe_vertex[vertex](2) = scene.scene_vertices[vertex_index * 3 + 2];

            // scene vertices are ray + idepth in keyframe coordinates
            Eigen::Vector3f keyframe_ray;
            keyframe_ray(0) = scene.scene_vertices[vertex_index * 3];
            keyframe_ray(1) = scene.scene_vertices[vertex_index * 3 + 1];
            keyframe_ray(2) = 1.0;
            float keyframe_idepth = scene.scene_vertices[vertex_index * 3 + 2];

            keyframe_vertex[vertex] = keyframe_ray / keyframe_idepth;
        }

        Eigen::Vector3f frame_vertex[3];

        // vertex from world reference to camera reference system
        frame_vertex[0] = frame.pose * keyframe_vertex[0];
        frame_vertex[1] = frame.pose * keyframe_vertex[1];
        frame_vertex[2] = frame.pose * keyframe_vertex[2];

        Eigen::Vector3f frame_normal = (frame_vertex[0] - frame_vertex[2]).cross(frame_vertex[0] - frame_vertex[1]);

        // back-face culling
        float point_dot_normal = frame_vertex[0].dot(frame_normal);
        if (point_dot_normal <= 0.0)
            continue;

        Eigen::Vector2f pixel[3];
        pixel[0] = Eigen::Vector2f(cam.fx[lvl] * frame_vertex[0](0) / frame_vertex[0](2) + cam.cx[lvl], cam.fy[lvl] * frame_vertex[0](1) / frame_vertex[0](2) + cam.cy[lvl]);
        pixel[1] = Eigen::Vector2f(cam.fx[lvl] * frame_vertex[1](0) / frame_vertex[1](2) + cam.cx[lvl], cam.fy[lvl] * frame_vertex[1](1) / frame_vertex[1](2) + cam.cy[lvl]);
        pixel[2] = Eigen::Vector2f(cam.fx[lvl] * frame_vertex[2](0) / frame_vertex[2](2) + cam.cx[lvl], cam.fy[lvl] * frame_vertex[2](1) / frame_vertex[2](2) + cam.cy[lvl]);

        int min_x = std::min(std::min(pixel[0](0), pixel[1](0)), pixel[2](0));
        int max_x = std::max(std::max(pixel[0](0), pixel[1](0)), pixel[2](0));
        int min_y = std::min(std::min(pixel[0](1), pixel[1](1)), pixel[2](1));
        int max_y = std::max(std::max(pixel[0](1), pixel[1](1)), pixel[2](1));

        // triangle outside of frame
        if (min_x >= cam.width[lvl] || max_x < 0.0)
            continue;
        if (min_y >= cam.height[lvl] || max_y < 0.0)
            continue;

        Eigen::Matrix2f T;
        T(0, 0) = pixel[0](0) - pixel[2](0);
        T(0, 1) = pixel[1](0) - pixel[2](0);
        T(1, 0) = pixel[0](1) - pixel[2](1);
        T(1, 1) = pixel[1](1) - pixel[2](1);
        Eigen::Matrix2f T_inv;
        T_inv = T.inverse();

        for (int y = min_y; y <= max_y; y++)
        {
            for (int x = min_x; x <= max_x; x++)
            {
                /*
                Eigen::Vector3f ray = Eigen::Vector3f(cam.fxinv[lvl] * (x + 0.5) + cam.cxinv[lvl], cam.fyinv[lvl] * (y + 0.5) + cam.cyinv[lvl], 1.0);
                float ray_dot_normal = ray.dot(frame_normal);
                if (ray_dot_normal <= 0.0) // osea, este punto no se ve desde la camara...
                    continue;
                float depth = point_dot_normal / ray_dot_normal; // ya estoy seguro que es positivo

                frame.idepth.texture[lvl].at<float>(y, x) = 1.0 / depth;
                */

                Eigen::Vector2f lambda = T_inv * (Eigen::Vector2f(x, y) - pixel[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                float depth = lambda(0) * frame_vertex[0](2) + lambda(1) * frame_vertex[1](2) + (1 - lambda(0) - lambda(1)) * frame_vertex[2](2);

                frame.idepth.texture[lvl].at<float>(y, x) = 1.0 / depth;
            }
        }
    }
}

float platformCpu::computeError(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl)
{
    HGPose _hjpose = errorPerIndex(frame, keyframe, cam, lvl, 0, cam.height[lvl]);
    // float error = treadReducer.reduce(std::bind(&mesh_vo::errorCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    return _hjpose.error / _hjpose.count;
}

HGPose platformCpu::errorPerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl, int ymin, int ymax)
{
    HGPose hgpose;

    Sophus::SE3f relativePose = frame.pose; // * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            uchar vkf = keyframe.image.texture[lvl].at<uchar>(y, x);
            float keyframeId = keyframe.idepth.texture[lvl].at<float>(y, x);

            if (keyframeId <= 0.0)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(cam.fxinv[lvl] * (x + 0.5) + cam.cxinv[lvl], cam.fyinv[lvl] * (y + 0.5) + cam.cyinv[lvl], 1.0) / keyframeId;
            Eigen::Vector3f pointFrame = relativePose * poinKeyframe;

            // std::cout << "pointKeyframe " << poinKeyframe << std::endl;
            // std::cout << "pointFrame " << pointFrame << std::endl;

            if (pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector2f pixelFrame = Eigen::Vector2f(cam.fx[lvl] * pointFrame(0) / pointFrame(2) + cam.cx[lvl], cam.fy[lvl] * pointFrame(1) / pointFrame(2) + cam.cy[lvl]);

            if (pixelFrame(0) < 0.0 || pixelFrame(0) > cam.width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) > cam.height[lvl])
                continue;

            // std::cout << "pixelFrame " << pixelFrame << std::endl;
            uchar vf = frame.image.texture[lvl].at<uchar>(pixelFrame(1), pixelFrame(0));

            float residual = float(vf) - float(vkf);
            float error = residual * residual;

            frame.error.texture[lvl].at<float>(y, x) = error;

            hgpose.error += error;
            hgpose.count++;
        }

    return hgpose;
}

HGPose platformCpu::computeHGPose(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl)
{
    HGPose hgpose = HGPosePerIndex(frame, keyframe, cam, lvl, 0, cam.height[lvl]);
    // HJPose _hjpose = treadReducer.reduce(std::bind(&mesh_vo::HJPoseCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    hgpose.H_pose /= hgpose.count;
    hgpose.G_pose /= hgpose.count;
    hgpose.error /= hgpose.count;

    return hgpose;
}

HGPose platformCpu::HGPosePerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl, int ymin, int ymax)
{
    HGPose hgpose;

    Sophus::SE3f relativePose = frame.pose; // * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            uchar vkf = keyframe.image.texture[lvl].at<uchar>(y, x);
            float keyframeId = keyframe.idepth.texture[lvl].at<float>(y, x);

            // std::cout << "keyframeId " << keyframeId << std::endl;

            if (keyframeId <= 0.0)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(cam.fxinv[lvl] * x + cam.cxinv[lvl], cam.fyinv[lvl] * y + cam.cyinv[lvl], 1.0) / keyframeId;
            Eigen::Vector3f pointFrame = relativePose * poinKeyframe;

            // std::cout << "pointFrame " << pointFrame << std::endl;

            if (pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector2f pixelFrame = Eigen::Vector2f(cam.fx[lvl] * pointFrame(0) / pointFrame(2) + cam.cx[lvl], cam.fy[lvl] * pointFrame(1) / pointFrame(2) + cam.cy[lvl]);

            // std::cout << "pixelFrame " << std::endl;

            if (pixelFrame(0) < 0.0 || pixelFrame(0) >= cam.width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= cam.height[lvl])
                continue;

            uchar vf = frame.image.texture[lvl].at<uchar>(pixelFrame(1), pixelFrame(0));
            cv::Vec2f der = frame.der.texture[lvl].at<cv::Vec2f>(pixelFrame(1), pixelFrame(0));

            Eigen::Vector2f d_f_d_uf(der.val[0], der.val[1]);

            // std::cout << "vf " << vf << " der " << der << std::endl;

            float id = 1.0 / pointFrame(2);

            float v0 = d_f_d_uf(0) * cam.fx[lvl] * id;
            float v1 = d_f_d_uf(1) * cam.fy[lvl] * id;
            float v2 = -(v0 * pointFrame(0) + v1 * pointFrame(1)) * id;

            Eigen::Vector3f d_I_d_tra = Eigen::Vector3f(v0, v1, v2);
            Eigen::Vector3f d_I_d_rot = Eigen::Vector3f(-pointFrame(2) * v1 + pointFrame(1) * v2, pointFrame(2) * v0 - pointFrame(0) * v2, -pointFrame(1) * v0 + pointFrame(0) * v1);

            float residual = (vf - vkf);
            hgpose.error += residual * residual;

            Eigen::Matrix<float, 6, 1> J;
            J << d_I_d_tra(0), d_I_d_tra(1), d_I_d_tra(2), d_I_d_rot(0), d_I_d_rot(1), d_I_d_rot(2);

            hgpose.count++;
            for (int i = 0; i < 6; i++)
            {
                hgpose.G_pose(i) += J[i] * residual;
                for (int j = i; j < 6; j++)
                {
                    float jj = J[i] * J[j];
                    hgpose.H_pose(i, j) += jj;
                    hgpose.H_pose(j, i) += jj;
                }
            }
        }

    return hgpose;
}

#version 330 core
layout(location = 0) out vec3 f_vertexID;
layout(location = 1) out float f_error;
layout(location = 2) out float f_d_I_d_z0;
layout(location = 3) out float f_d_I_d_z1;
layout(location = 4) out float f_d_I_d_z2;
layout(location = 5) out float f_debug;

in vec3 g_pframe;
in vec3 g_pkeyframe;
in vec2 g_u;

flat in float g_idepth[3];
flat in int g_vertexID[3];

flat in vec3 g_N_p0;
flat in vec3 g_d_N_d_z0;
flat in float g_N_p0_dot_point;
flat in vec3 g_pr_p0;

flat in vec3 g_N_p1;
flat in vec3 g_d_N_d_z1;
flat in float g_N_p1_dot_point;
flat in vec3 g_pr_p1;

flat in vec3 g_N_p2;
flat in vec3 g_d_N_d_z2;
flat in float g_N_p2_dot_point;
flat in vec3 g_pr_p2;

uniform float dx;
uniform float dy;

uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;

uniform float fxinv;
uniform float fyinv;
uniform float cxinv;
uniform float cyinv;

uniform mat4 framePose;

// texture samplers
uniform sampler2D keyframe;
uniform sampler2D frame;
uniform sampler2D frameDer;

HGMap platformCpu::HGMapPerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl, int ymin, int ymax)
{
    Sophus::SE3f relativePose = frame.pose; // * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            // if(g_u.x < 0.1/dx || g_u.x > 0.9/dx || g_u.y < 0.1/dy || g_u.y > 0.9/dy)
            //     discard;

            // from frame perspective
            vec2 ukeyframe = vec2(g_u.x, 1.0 / dy - g_u.y);
            vec3 pkeyframe = g_pkeyframe;
            float keyframeDepth = g_pkeyframe.z;
            vec2 uframe = vec2(gl_FragCoord.x, gl_FragCoord.y);
            vec3 pframe = (framePose * vec4(pkeyframe, 1.0)).xyz;
            float frameDepth = gl_FragCoord.z / gl_FragCoord.w;

            // from keyframe perspective
            // vec2 uframe  = vec2(g_u.x,1.0/dy-g_u.y);
            // vec2 ukeyframe = vec2(gl_FragCoord.x,gl_FragCoord.y);
            // float keyframeDepth = gl_FragCoord.z/gl_FragCoord.w;

            vec2 ukeyframeTexCoord = vec2(ukeyframe.x * dx, ukeyframe.y * dy);
            vec2 uframeTexCoord = vec2(uframe.x * dx, uframe.y * dy);

            float ikeyframe = texture(keyframe, ukeyframeTexCoord).x * 255.0;
            // float ikeyframe = textureLod(keyframe,ukeyframeTexCoord,srclvl).x*255.0;
            // vec2 dkeyframe = texture(keyframeDer,ukeyframeTexCoord).xy;

            float iframe = texture(frame, uframeTexCoord).x * 255.0;
            // float iframe = textureLod(frame,uframeTexCoord,srclvl).x*255.0;
            vec2 dframe = texture(frameDer, uframeTexCoord).xy;
            // vec2 dframe = textureLod(frameDer,uframeTexCoord,srclvl).xy;

            float cosangle = abs(dot(normalize(pkeyframe), normalize(framePose[3].xyz)));
            if (cosangle >= 0.8)
                discard;

            vec3 d_I_d_pframe = vec3(0);
            d_I_d_pframe.x = dframe.x * fx / frameDepth;
            d_I_d_pframe.y = dframe.y * fy / frameDepth;
            d_I_d_pframe.z = -(d_I_d_pframe.x * pframe.x / frameDepth + d_I_d_pframe.y * pframe.y / frameDepth);

            vec3 raykeyframe = pkeyframe / keyframeDepth;

            vec3 d_pframe_d_z = mat3(framePose) * raykeyframe;

            float d_I_d_z = dot(d_I_d_pframe, d_pframe_d_z);

            float N_p0_dot_ray = dot(g_N_p0, raykeyframe);
            float N_p1_dot_ray = dot(g_N_p1, raykeyframe);
            float N_p2_dot_ray = dot(g_N_p2, raykeyframe);

            //    if(abs(N_p0_dot_ray) <= 0.001 || abs(N_p1_dot_ray) <= 0.001 || abs(N_p2_dot_ray) <= 0.001)
            //        discard;

            //    if(abs(N_p0_dot_ray)/(length(g_N_p0)*length(raykeyframe)) <= 0.0)
            //        discard;
            /*
                vec3 N_frame = normalize(mat3(framePose)*g_N_p0);
                vec3 rayframe = normalize(pframe/frameDepth);
                float N_frame_dot_rayframe = dot(N_frame,rayframe);
                if(abs(N_frame_dot_rayframe) <= 0.02)
                    discard;
            */
            float d_z_d_z0 = dot(g_d_N_d_z0, g_pr_p0) / N_p0_dot_ray - g_N_p0_dot_point * dot(g_d_N_d_z0, raykeyframe) / (N_p0_dot_ray * N_p0_dot_ray);
            float d_z_d_z1 = dot(g_d_N_d_z1, g_pr_p1) / N_p1_dot_ray - g_N_p1_dot_point * dot(g_d_N_d_z1, raykeyframe) / (N_p1_dot_ray * N_p1_dot_ray);
            float d_z_d_z2 = dot(g_d_N_d_z2, g_pr_p2) / N_p2_dot_ray - g_N_p2_dot_point * dot(g_d_N_d_z2, raykeyframe) / (N_p2_dot_ray * N_p2_dot_ray);

            float d_z0_d_iz0 = -1.0 / (g_idepth[0] * g_idepth[0]);
            float d_z1_d_iz1 = -1.0 / (g_idepth[1] * g_idepth[1]);
            float d_z2_d_iz2 = -1.0 / (g_idepth[2] * g_idepth[2]);

            float d_I_d_z0 = d_I_d_z * d_z_d_z0 * d_z0_d_iz0;
            float d_I_d_z1 = d_I_d_z * d_z_d_z1 * d_z1_d_iz1;
            float d_I_d_z2 = d_I_d_z * d_z_d_z2 * d_z2_d_iz2;

            float error = iframe - ikeyframe;

            f_vertexID = vec3(g_vertexID[0], g_vertexID[1], g_vertexID[2]);
            f_error = error;
            f_d_I_d_z0 = d_I_d_z0;
            f_d_I_d_z1 = d_I_d_z1;
            f_d_I_d_z2 = d_I_d_z2;

            /*
            float color = 0.0;
            if(isinf(d_I_d_z0) || isnan(d_I_d_z0) || isinf(d_I_d_z1) || isnan(d_I_d_z1) || isinf(d_I_d_z2) || isnan(d_I_d_z2))
                color = 1.0;
            f_debug = color;
            */

            f_debug = 1.0 / frameDepth + abs(error) / 127.0;
        }

    /*
    vertexIdData.generateMipmapsGPU(lvl);
    primitiveIdData.generateMipmapsGPU(lvl);
    residualData.generateMipmapsGPU(lvl);
    d_I_d_p0Data.generateMipmapsGPU(lvl);
    d_I_d_p1Data.generateMipmapsGPU(lvl);
    d_I_d_p2Data.generateMipmapsGPU(lvl);
*/
    vertexIdData.gpu_to_cpu(lvl);
    _frame.error.gpu_to_cpu(lvl);
    _frame.jp0.gpu_to_cpu(lvl);
    _frame.jp1.gpu_to_cpu(lvl);
    _frame.jp2.gpu_to_cpu(lvl);

    int count = 0;
    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            //std::cout << "pixel index " << index << std::endl;
            //acumular para cada pixel
            float error = _frame.error.cpuTexture[lvl].at<float>(y,x);

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
            J[0] = _frame.jp0.cpuTexture[lvl].at<float>(y,x);
            J[1] = _frame.jp1.cpuTexture[lvl].at<float>(y,x);
            J[2] = _frame.jp2.cpuTexture[lvl].at<float>(y,x);

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

                for(int j = i; j < 3; j++)
                {
                    //acc_H_depth(vertexID[i],vertexID[j]) += J[i]*J[j];
                    float jj = J[i]*J[j];
                    H_depth.coeffRef(vertexID[i],vertexID[j]) += jj;
                    H_depth.coeffRef(vertexID[j],vertexID[i]) += jj;
                }
            }
        }
    if(count > 0)
    {
        H_depth /= count;
        J_depth /= count;
    }
}