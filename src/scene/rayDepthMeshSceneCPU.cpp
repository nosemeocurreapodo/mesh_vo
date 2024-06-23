#include "scene/rayDepthMeshSceneCPU.h"
#include "common/Triangle.h"
#include "utils/tictoc.h"

rayDepthMeshSceneCPU::rayDepthMeshSceneCPU(float fx, float fy, float cx, float cy, int width, int height)
    : cam(fx, fy, cx, cy, width, height),
      z_buffer(-1.0)
{
    multiThreading = false;
    meshRegularization = 100.0;

    sceneMesh.init();

}

void rayDepthMeshSceneCPU::init(frameCPU &frame, dataCPU<float> &idepth)
{
    frame.copyTo(keyframe);
    setFromIdepth(idepth);
}

void rayDepthMeshSceneCPU::setFromIdepth(dataCPU<float> id)
{
    sceneMesh.vertices.clear();

    int lvl = 0;

    for (int y = 0; y < MESH_HEIGHT; y++)
    {
        for (int x = 0; x < MESH_WIDTH; x++)
        {
            float xi = (float(x) / float(VERTEX_WIDTH - 1)) * cam.width[lvl];
            float yi = (float(y) / float(VERTEX_HEIGHT - 1)) * cam.height[lvl];

            float idepth = id.get(yi, xi, lvl);
            /*
            if(idepth <= min_idepth)
                idepth = min_idepth;
            if(idepth > max_idepth)
                idepth = max_idepth;
                */
            if (idepth != idepth || idepth < 0.1 || idepth > 1.0)
                idepth = 0.1 + (1.0 - 0.1) * float(y) / VERTEX_HEIGHT;

            Eigen::Vector3f u = Eigen::Vector3f(xi, yi, 1.0);
            Eigen::Vector3f r = Eigen::Vector3f(cam.fxinv[lvl] * u(0) + cam.cxinv[lvl], cam.fyinv[lvl] * u(1) + cam.cyinv[lvl], 1.0);
            // Eigen::Vector3f p = r / idepth;

            std::array<float, 3> vertex;
            vertex[0] = r(0);
            vertex[1] = r(1);
            vertex[2] = idepth;

            sceneMesh.vertices.push_back(vertex);
            // vertices.push_back(r(0));
            // vertices.push_back(r(1));
            // vertices.push_back(idepth);
        }
    }
}

dataCPU<float> rayDepthMeshSceneCPU::computeFrameIdepth(frameCPU &frame, int lvl)
{
    dataCPU<float> idepth(-1.0);

    Mesh frameMesh = sceneMesh.getCopy();
    frameMesh.transform(frame.pose);
    frameMesh.project(cam, lvl);

    // for each triangle
    for (std::size_t t_id = 0; t_id < frameMesh.triangles.size(); t_id++)
    {
        Triangle kf_tri = sceneMesh.triangles[t_id];
        if(kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(kf_tri.isBackFace())
            continue;
        Triangle f_tri = frameMesh.triangle[t_id];
        if(f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(f_tri.isBackFace())
            continue;

        f_tri.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri.getMinMax();

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix, lvl))
                    continue;

                tri.computeBarycentric(f_pix);
                if (!tri.isBarycentricOk())
                    continue;

                Eigen::Vector3f f_ver = tri.interpolate(f_tri.vertices[0]->position, f_tri.vertices[1]->position, f_tri.vertices[2]->position);

                if (f_ver(2) <= 0.0)
                    continue;

                // float l_idepth = idepth.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != idepth.nodata)
                //     continue;

                idepth.set(1.0 / f_ver(2), f_pix(1), f_pix(0), lvl);
            }
        }
    }
    return idepth;
}

Error rayDepthMeshSceneCPU::computeError(frameCPU &frame, int lvl)
{
    Error e;

    if (multiThreading)
    {
        errorTreadReduce.reduce(boost::bind(&rayDepthMeshSceneCPU::errorPerIndex, this, frame, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, sceneMesh.triangle_index_to_vertices_indeces.size(), 0);
        e = errorTreadReduce.stats;
    }
    else
    {
        errorPerIndex(frame, lvl, 0, sceneMesh.triangle_index_to_vertices_indeces.size(), &e, 0);
    }

    // if (e.count > 0)
    //     e.error /= e.count;

    // return e.error;
    return e;
}

void rayDepthMeshSceneCPU::errorPerIndex(frameCPU &frame, int lvl, int tmin, int tmax, Error *e, int tid)
{
    // z_buffer.reset(lvl);

    Mesh frameMesh = sceneMesh.getCopy();

    frameMesh.transform(frame.pose);
    frameMesh.project(cam, lvl);

    // for each triangle
    for (std::size_t t_id = tmin; t_id < tmax; t_id++)
    {
        Triangle kf_tri = sceneMesh.triangles[t_id];
        if(kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(kf_tri.isBackFace())
            continue;
        Triangle f_tri = frameMesh.triangle[t_id];
        if(f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(f_tri.isBackFace())
            continue;

        f_tri.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri.getMinMax();

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix, lvl))
                    continue;

                tri.computeBarycentric(f_pix);
                if (!tri.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = tri.interpolate(kf_tri.vertices[0]->texcoord, kf_tri.vertices[1]->texcoord, kf_tri.vertices[2]->texcoord);
                if (!cam.isPixVisible(kf_pix, lvl))
                    continue;

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame.image.get(f_pix(1), f_pix(0), lvl));
                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata)
                    continue;

                float residual = f_i - kf_i;
                float residual_2 = residual * residual;

                // z buffer
                // float f_idepth = 1.0 / f_ver(2);
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                (*e).error += residual_2;
                (*e).count++;
            }
        }
    }
}

dataCPU<float> rayDepthMeshSceneCPU::computeErrorImage(frameCPU &frame, int lvl)
{
    dataCPU<float> error(-1.0);

    // for each triangle
    for (std::size_t t_id = 0; t_id < sceneMesh.triangle_index_to_vertices_indeces.size(); t_id++)
    {
        Triangle kf_tri = sceneMesh.triangles[t_id];
        if(kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(kf_tri.isBackFace())
            continue;
        Triangle f_tri = frameMesh.triangle[t_id];
        if(f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(f_tri.isBackFace())
            continue;

        f_tri.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri.getMinMax();

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix, lvl))
                    continue;
                tri.computeBarycentric(f_pix);
                if (!tri.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = tri.interpolate(kf_tri.vertices[0]->texcoord, kf_tri.vertices[1]->texcoord, kf_tri.vertices[2]->texcoord);
                if (!cam.isPixVisible(kf_pix, lvl))
                    continue;

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame.image.get(f_pix(1), f_pix(0), lvl));
                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata)
                    continue;

                float residual = f_i - kf_i;
                float residual_2 = residual * residual;

                // z buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                error.set(residual_2, y, x, lvl);
            }
        }
    }
    return error;
}

dataCPU<float> rayDepthMeshSceneCPU::computeSceneImage(frameCPU &frame, int lvl)
{
    dataCPU<float> image(-1.0);

    // for each triangle
    for (std::size_t index = 0; index < sceneMesh.triangle_index_to_vertices_indeces.size(); index++)
    {
        Triangle kf_tri = sceneMesh.triangles[t_id];
        if(kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(kf_tri.isBackFace())
            continue;
        Triangle f_tri = frameMesh.triangle[t_id];
        if(f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(f_tri.isBackFace())
            continue;

        f_tri.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri.getMinMax();

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix, lvl))
                    continue;

                tri.computeBarycentric(f_pix);
                if (!tri.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = tri.interpolate(kf_tri.vertices[0]->texcoord, kf_tri.vertices[1]->texcoord, kf_tri.vertices[2]->texcoord);
                if (!cam.isPixVisible(kf_pix, lvl))
                    continue;

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                if (kf_i == keyframe.image.nodata)
                    continue;

                // float f_idepth = 1.0 / f_ver(2);

                // z buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                image.set(kf_i, y, x, lvl);
            }
        }
    }
    return image;
}

HGPose rayDepthMeshSceneCPU::computeHGPose(frameCPU &frame, int lvl)
{
    HGPose hg;

    if (multiThreading)
    {
        hgPoseTreadReduce.reduce(boost::bind(&rayDepthMeshSceneCPU::HGPosePerIndex, this, frame, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, sceneMesh.triangle_index_to_vertices_indeces.size(), 0);
        hg = hgPoseTreadReduce.stats;
    }
    else
    {
        HGPosePerIndex(frame, lvl, 0, sceneMesh.triangle_index_to_vertices_indeces.size(), &hg, 0);
    }

    /*
    if(hg.count > 0)
    {
         hg.H /= hg.count;
         hg.G /= hg.count;
    }
    */

    return hg;
}

void rayDepthMeshSceneCPU::HGPosePerIndex(frameCPU &frame, int lvl, int tmin, int tmax, HGPose *hg, int tid)
{
    // z_buffer.reset(lvl);

    Mesh frameMesh = sceneMesh.getCopy();
    frameMesh.transform(frame.pose);
    frameMesh.project(cam, lvl);

    // for each triangle
    for (int index = tmin; index < tmax; index++)
    {
        Triangle kf_tri = sceneMesh.triangles[t_id];
        if(kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(kf_tri.isBackFace())
            continue;
        Triangle f_tri = frameMesh.triangle[t_id];
        if(f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(f_tri.isBackFace())
            continue;

        f_tri.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri.getMinMax();

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix, lvl))
                    continue;

                tri.computeBarycentric(f_pix);

                if (!tri.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = tri.interpolate(kf_tri.vertices[0]->texcoord, kf_tri.vertices[1]->texcoord, kf_tri.vertices[2]->texcoord);
                if (!cam.isPixVisible(kf_pix, lvl))
                    continue;

                Eigen::Vector3f f_ver = tri.interpolate(f_tri.vertices[0]->position, f_tri.vertices[1]->position, f_tri.vertices[2]->position);

                if (f_ver(2) <= 0.0)
                    continue;

                // z-buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                float v0 = d_f_i_d_pix(0) * cam.fx[lvl] / f_ver(2);
                float v1 = d_f_i_d_pix(1) * cam.fy[lvl] / f_ver(2);
                float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_i_d_tra = Eigen::Vector3f(v0, v1, v2);
                Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);

                float residual = (f_i - kf_i);
                // float residual_2 = residual * residual;

                Eigen::Matrix<float, 6, 1> J;
                J << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                (*hg).count += 1;
                for (int i = 0; i < 6; i++)
                {
                    (*hg).G(i) += J[i] * residual;
                    for (int j = i; j < 6; j++)
                    {
                        float jj = J[i] * J[j];
                        (*hg).H(i, j) += jj;
                        (*hg).H(j, i) += jj;
                    }
                }
            }
        }
    }
}

HGPoseMapMesh rayDepthMeshSceneCPU::computeHGMap(frameCPU &frame, int lvl)
{
    HGPoseMapMesh hg;

    if (multiThreading)
    {
        hgPoseMapTreadReduce.reduce(boost::bind(&rayDepthMeshSceneCPU::HGMapPerIndex, this, frame, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, sceneMesh.triangle_index_to_vertices_indeces.size(), 0);
        hg = hgPoseMapTreadReduce.stats;
    }
    else
    {
        HGMapPerIndex(frame, lvl, 0, sceneMesh.triangle_index_to_vertices_indeces.size(), &hg, 0);
    }

    // if(hg.count > 0)
    //{
    //      hg.H /= hg.count;
    //      hg.G /= hg.count;
    // }

    return hg;
}

void rayDepthMeshSceneCPU::HGMapPerIndex(frameCPU &frame, int lvl, int tmin, int tmax, HGPoseMapMesh *hg, int tid)
{
    // z_buffer.reset(lvl);

    Mesh frameMesh = sceneMesh.getCopy();
    frameMesh.transform(frame.pose);
    frameMesh.project(cam, lvl);

    // for each triangle
    for (std::size_t index = tmin; index < tmax; index++)
    {
        Triangle kf_tri = sceneMesh.triangles[t_id];
        if(kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(kf_tri.isBackFace())
            continue;
        Triangle f_tri = frameMesh.triangle[t_id];
        if(f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
            continue;
        if(f_tri.isBackFace())
            continue;

        f_tri.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri.getMinMax();

        Eigen::Vector3f n_p[3];
        n_p[0] = (kf_tri.vertices[0]->position - kf_tri.vertices[1]->position).cross(kf_tri.vertices[2]->position - kf_tri.vertices[1]->position);
        n_p[1] = (kf_tri.vertices[1]->position - kf_tri.vertices[0]->position).cross(kf_tri.vertices[2]->position - kf_tri.vertices[0]->position);
        n_p[2] = (kf_tri.vertices[2]->position - kf_tri.vertices[1]->position).cross(kf_tri.vertices[0]->position - kf_tri.vertices[1]->position);

        Eigen::Vector3f pw2mpw1[3];
        pw2mpw1[0] = (kf_tri.vertices[2]->position - kf_tri.vertices[1]->position);
        pw2mpw1[1] = (kf_tri.vertices[2]->position - kf_tri.vertices[0]->position);
        pw2mpw1[2] = (kf_tri.vertices[0]->position - kf_tri.vertices[1]->position);

        float n_p_dot_point[3];
        n_p_dot_point[0] = n_p[0].dot(kf_tri.vertices[1]->position);
        n_p_dot_point[1] = n_p[1].dot(kf_tri.vertices[0]->position);
        n_p_dot_point[2] = n_p[2].dot(kf_tri.vertices[1]->position);

        Eigen::Vector3f pr_p[3];
        pr_p[0] = kf_tri.vertices[1]->position;
        pr_p[1] = kf_tri.vertices[0]->position;
        pr_p[2] = kf_tri.vertices[1]->position;

        Eigen::Vector3f d_n_d_z[3];
        float d_z_d_iz[3];
        for (int i = 0; i < 3; i++)
        {
            d_n_d_z[i] = kf_tri.vertices[i]->position.cross(pw2mpw1[i]);
            // d_z_d_iz[i] = -1.0 / (kf_tri_idepth[i] * kf_tri_idepth[i]);
            d_z_d_iz[i] = -(kf_tri.vertices[i]->position(2) * kf_tri.vertices[i]->position(2));
        }

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix, lvl))
                    continue;

                tri.computeBarycentric(f_pix);
                if (!tri.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = tri.interpolate(kf_tri.vertices[0]->texcoord, kf_tri.vertices[1]->texcoord, kf_tri.vertices[2]->texcoord);
                if (!cam.isPixVisible(kf_pix, lvl))
                    continue;

                Eigen::Vector3f kf_ver = tri.interpolate(kf_tri.vertices[0]->position, kf_tri.vertices[1]->position, kf_tri.vertices[2]->position);
                Eigen::Vector3f f_ver = tri.interpolate(f_tri.vertices[0]->position, f_tri.vertices[1]->position, f_tri.vertices[2]->position);

                // z-buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                Eigen::Vector3f d_f_i_d_f_ver;
                d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx[lvl] / f_ver(2);
                d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy[lvl] / f_ver(2);
                // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) / f_ver(2) + d_f_i_d_f_ver(1) * f_ver(1) / f_ver(2));
                d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_ver_d_kf_depth = frame.pose.rotationMatrix() * kf_ver / kf_ver(2);

                float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                float error = f_i - kf_i;

                float J[3];
                for (int i = 0; i < 3; i++)
                {
                    float n_p_dot_ray = n_p[i].dot(kf_ver / kf_ver(2));
                    float d_kf_depth_d_z = d_n_d_z[i].dot(pr_p[i]) / n_p_dot_ray - n_p_dot_point[i] * d_n_d_z[i].dot(kf_ver / kf_ver(2)) / (n_p_dot_ray * n_p_dot_ray);
                    float d_f_i_d_z = d_f_i_d_kf_depth * d_kf_depth_d_z * d_z_d_iz[i];
                    J[i] = d_f_i_d_z;
                }

                (*hg).count += 1;
                for (int i = 0; i < 3; i++)
                {
                    // if the jacobian is 0
                    // we really cannot say anything about the depth
                    // can make the hessian non-singular
                    if (J[i] == 0)
                        continue;
                    (*hg).G(NUM_FRAMES * 6 + vertex_id[i]) += J[i] * error; // / (cam.width[lvl]*cam.height[lvl]);
                    (*hg).G_count(NUM_FRAMES * 6 + vertex_id[i]) += 1;
                    for (int j = i; j < 3; j++)
                    {
                        float jj = J[i] * J[j]; // / (cam.width[lvl]*cam.height[lvl]);
                        (*hg).H.coeffRef(NUM_FRAMES * 6 + vertex_id[i], NUM_FRAMES * 6 + vertex_id[j]) += jj;
                        (*hg).H.coeffRef(NUM_FRAMES * 6 + vertex_id[j], NUM_FRAMES * 6 + vertex_id[i]) += jj;
                    }
                }
            }
        }
    }
}

void rayDepthMeshSceneCPU::computeHGPoseMap(frameCPU &frame, HGPoseMapMesh &hg, int frame_index, int lvl)
{
    // z_buffer.reset(lvl);

    // for each triangle
    for (std::size_t t_id = 0; t_id < sceneMesh.triangle_index_to_vertices_indeces.size(); t_id++)
    {
        // get its vertices
        // Eigen::Vector3f world_vertex[3];

        int vertex_id[3];

        Eigen::Vector3f kf_tri_ver[3];
        Eigen::Vector3f kf_tri_ray[3];
        Eigen::Vector2f kf_tri_pix[3];
        float kf_tri_idepth[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            int vertex_i = sceneMesh.triangle_index_to_vertices_indeces[t_id][vertex];

            vertex_id[vertex] = vertex_i;

            // scene vertices are ray + idepth in keyframe coordinates
            kf_tri_ray[vertex](0) = sceneMesh.vertices[vertex_i][0];
            kf_tri_ray[vertex](1) = sceneMesh.vertices[vertex_i][1];
            kf_tri_ray[vertex](2) = 1.0;
            kf_tri_idepth[vertex] = sceneMesh.vertices[vertex_i][2];

            kf_tri_ver[vertex] = kf_tri_ray[vertex] / kf_tri_idepth[vertex];

            kf_tri_pix[vertex](0) = cam.fx[lvl] * kf_tri_ray[vertex](0) + cam.cx[lvl];
            kf_tri_pix[vertex](1) = cam.fy[lvl] * kf_tri_ray[vertex](1) + cam.cy[lvl];
        }

        int min_x = std::min(std::min(kf_tri_pix[0](0), kf_tri_pix[1](0)), kf_tri_pix[2](0));
        int max_x = std::max(std::max(kf_tri_pix[0](0), kf_tri_pix[1](0)), kf_tri_pix[2](0));
        int min_y = std::min(std::min(kf_tri_pix[0](1), kf_tri_pix[1](1)), kf_tri_pix[2](1));
        int max_y = std::max(std::max(kf_tri_pix[0](1), kf_tri_pix[1](1)), kf_tri_pix[2](1));

        // triangle outside of frame
        if (min_x >= cam.width[lvl] || max_x < 0.0)
            continue;
        if (min_y >= cam.height[lvl] || max_y < 0.0)
            continue;

        Eigen::Vector3f f_tri_ver[3];
        Eigen::Vector3f f_tri_ray[3];
        Eigen::Vector2f f_tri_pix[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            // vertex from world reference to camera reference system
            f_tri_ver[vertex] = frame.pose * kf_tri_ver[vertex];
            f_tri_ray[vertex] = f_tri_ver[vertex] / f_tri_ver[vertex](2);

            f_tri_pix[vertex](0) = cam.fx[lvl] * f_tri_ray[vertex](0) + cam.cx[lvl];
            f_tri_pix[vertex](1) = cam.fy[lvl] * f_tri_ray[vertex](1) + cam.cy[lvl];
        }

        Eigen::Vector3f f_tri_nor = (f_tri_ver[0] - f_tri_ver[2]).cross(f_tri_ver[0] - f_tri_ver[1]);

        // back-face culling
        float point_dot_normal = f_tri_ver[0].dot(f_tri_nor);
        if (point_dot_normal <= 0.0)
            continue;

        min_x = std::min(std::min(f_tri_pix[0](0), f_tri_pix[1](0)), f_tri_pix[2](0));
        max_x = std::max(std::max(f_tri_pix[0](0), f_tri_pix[1](0)), f_tri_pix[2](0));
        min_y = std::min(std::min(f_tri_pix[0](1), f_tri_pix[1](1)), f_tri_pix[2](1));
        max_y = std::max(std::max(f_tri_pix[0](1), f_tri_pix[1](1)), f_tri_pix[2](1));

        // triangle outside of frame
        if (min_x >= cam.width[lvl] || max_x < 0.0)
            continue;
        if (min_y >= cam.height[lvl] || max_y < 0.0)
            continue;

        Eigen::Vector3f n_p[3];
        n_p[0] = (kf_tri_ver[0] - kf_tri_ver[1]).cross(kf_tri_ver[2] - kf_tri_ver[1]);
        n_p[1] = (kf_tri_ver[1] - kf_tri_ver[0]).cross(kf_tri_ver[2] - kf_tri_ver[0]);
        n_p[2] = (kf_tri_ver[2] - kf_tri_ver[1]).cross(kf_tri_ver[0] - kf_tri_ver[1]);

        Eigen::Vector3f pw2mpw1[3];
        pw2mpw1[0] = (kf_tri_ver[2] - kf_tri_ver[1]);
        pw2mpw1[1] = (kf_tri_ver[2] - kf_tri_ver[0]);
        pw2mpw1[2] = (kf_tri_ver[0] - kf_tri_ver[1]);

        float n_p_dot_point[3];
        n_p_dot_point[0] = n_p[0].dot(kf_tri_ver[1]);
        n_p_dot_point[1] = n_p[1].dot(kf_tri_ver[0]);
        n_p_dot_point[2] = n_p[2].dot(kf_tri_ver[1]);

        Eigen::Vector3f pr_p[3];
        pr_p[0] = kf_tri_ver[1];
        pr_p[1] = kf_tri_ver[0];
        pr_p[2] = kf_tri_ver[1];

        Eigen::Vector3f d_n_d_z[3];
        float d_z_d_iz[3];
        for (int i = 0; i < 3; i++)
        {
            d_n_d_z[i] = kf_tri_ver[i].cross(pw2mpw1[i]);
            d_z_d_iz[i] = -1.0 / (kf_tri_idepth[i] * kf_tri_idepth[i]);
        }

        Eigen::Matrix2f T;
        T(0, 0) = f_tri_pix[0](0) - f_tri_pix[2](0);
        T(0, 1) = f_tri_pix[1](0) - f_tri_pix[2](0);
        T(1, 0) = f_tri_pix[0](1) - f_tri_pix[2](1);
        T(1, 1) = f_tri_pix[1](1) - f_tri_pix[2](1);
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

                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (f_pix(0) < 0.0 || f_pix(0) >= cam.width[lvl] || f_pix(1) < 0.0 || f_pix(1) >= cam.height[lvl])
                    continue;

                Eigen::Vector2f lambda = T_inv * (f_pix - f_tri_pix[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                Eigen::Vector2f kf_pix = lambda(0) * kf_tri_pix[0] + lambda(1) * kf_tri_pix[1] + (1 - lambda(0) - lambda(1)) * kf_tri_pix[2];
                // this -1 is for the bilinear interpolation
                if (kf_pix(0) < 0.0 || kf_pix(0) >= cam.width[lvl] - 1 || kf_pix(1) < 0.0 || kf_pix(1) >= cam.height[lvl] - 1)
                    continue;

                Eigen::Vector3f kf_ray = lambda(0) * kf_tri_ray[0] + lambda(1) * kf_tri_ray[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ray[2];
                // Eigen::Vector3f kf_ver = lambda(0) * kf_tri_ver[0] + lambda(1) * kf_tri_ver[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ver[2];

                // Eigen::Vector3f f_ray = lambda(0) * f_tri_ray[0] + lambda(1) * f_tri_ray[1] + (1 - lambda(0) - lambda(1)) * f_tri_ray[2];
                Eigen::Vector3f f_ver = lambda(0) * f_tri_ver[0] + lambda(1) * f_tri_ver[1] + (1 - lambda(0) - lambda(1)) * f_tri_ver[2];

                float f_idepth = 1.0 / f_ver(2);

                // z-buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                Eigen::Vector3f d_f_i_d_f_ver;
                d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx[lvl] * f_idepth;
                d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy[lvl] * f_idepth;
                // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) / f_ver(2) + d_f_i_d_f_ver(1) * f_ver(1) / f_ver(2));
                d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) * f_idepth;

                Eigen::Vector3f d_f_i_d_tra = d_f_i_d_f_ver;
                Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                Eigen::Matrix<float, 6, 1> J_pose;
                J_pose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                Eigen::Vector3f d_f_ver_d_kf_depth = frame.pose.rotationMatrix() * kf_ray;

                float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                float error = f_i - kf_i;

                float J_depth[3];
                for (int i = 0; i < 3; i++)
                {
                    float n_p_dot_ray = n_p[i].dot(kf_ray);
                    float d_kf_depth_d_z = d_n_d_z[i].dot(pr_p[i]) / n_p_dot_ray - n_p_dot_point[i] * d_n_d_z[i].dot(kf_ray) / (n_p_dot_ray * n_p_dot_ray);
                    float d_f_i_d_z = d_f_i_d_kf_depth * d_kf_depth_d_z * d_z_d_iz[i];
                    J_depth[i] = d_f_i_d_z;
                }

                hg.count += 1;
                for (int i = 0; i < 6; i++)
                {
                    hg.G(i + frame_index * 6) += J_pose[i] * error;
                    hg.G_count(i + frame_index * 6) += 1;
                    for (int j = i; j < 6; j++)
                    {
                        float jj = J_pose[i] * J_pose[j];
                        hg.H.coeffRef(i + frame_index * 6, j + frame_index * 6) += jj;
                        hg.H.coeffRef(j + frame_index * 6, i + frame_index * 6) += jj;
                    }
                }

                for (int i = 0; i < 3; i++)
                {
                    // if the jacobian is 0
                    // we really cannot say anything about the depth
                    // can make the hessian non-singular
                    if (J_depth[i] == 0)
                        continue;
                    hg.G(vertex_id[i] + NUM_FRAMES * 6) += J_depth[i] * error; // / (cam.width[lvl]*cam.height[lvl]);
                    hg.G_count(vertex_id[i] + NUM_FRAMES * 6) += 1;
                    for (int j = i; j < 3; j++)
                    {
                        float jj = J_depth[i] * J_depth[j]; // / (cam.width[lvl]*cam.height[lvl]);
                        hg.H.coeffRef(vertex_id[i] + NUM_FRAMES * 6, vertex_id[j] + NUM_FRAMES * 6) += jj;
                        hg.H.coeffRef(vertex_id[j] + NUM_FRAMES * 6, vertex_id[i] + NUM_FRAMES * 6) += jj;
                    }
                }
            }
        }
    }
}

Error rayDepthMeshSceneCPU::errorRegu()
{
    Error error;
    // for each triangle
    for (size_t i = 0; i < sceneMesh.triangle_index_to_vertices_indeces.size(); i++)
    {
        float idepth[3];
        for (int j = 0; j < 3; j++)
        {
            int vertexIndex = sceneMesh.triangle_index_to_vertices_indeces[i][j];
            idepth[j] = sceneMesh.vertices[vertexIndex][2];
        }
        float diff1 = idepth[0] - idepth[1];
        float diff2 = idepth[0] - idepth[2];
        float diff3 = idepth[1] - idepth[2];

        error.error += diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    }
    // divided by the number of triangles
    // we don't want to have less error if we have less triangles
    error.count = sceneMesh.triangle_index_to_vertices_indeces.size();
    return error;
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

HGPoseMapMesh rayDepthMeshSceneCPU::HGRegu()
{
    HGPoseMapMesh hg;
    for (size_t i = 0; i < sceneMesh.triangle_index_to_vertices_indeces.size(); i++)
    {
        int vertexIndex[3];
        float idepth[3];
        for (int j = 0; j < 3; j++)
        {
            vertexIndex[j] = sceneMesh.triangle_index_to_vertices_indeces[i][j];
            idepth[j] = sceneMesh.vertices[vertexIndex[j]][2];
        }
        float diff1 = idepth[0] - idepth[1];
        float diff2 = idepth[0] - idepth[2];
        float diff3 = idepth[1] - idepth[2];

        float J1[3] = {1.0, -1.0, 0.0};
        float J2[3] = {1.0, 0.0, -1.0};
        float J3[3] = {0.0, 1.0, -1.0};

        for (int j = 0; j < 3; j++)
        {
            // if (hg.G(NUM_FRAMES*6 + vertexIndex[j]) == 0)
            //     continue;
            hg.G(NUM_FRAMES * 6 + vertexIndex[j]) += (diff1 * J1[j] + diff2 * J2[j] + diff3 * J3[j]);
            hg.G_count(NUM_FRAMES * 6 + vertexIndex[j]) += 1;
            // J_joint(MAX_FRAMES * 6 + vertexIndex[j]) += (MESH_REGU / (VERTEX_HEIGH * VERTEX_WIDTH)) * (diff1 * J1[j] + diff2 * J2[j] + diff3 * J3[j]);
            for (int k = 0; k < 3; k++)
            {
                hg.H.coeffRef(NUM_FRAMES * 6 + vertexIndex[j], NUM_FRAMES * 6 + vertexIndex[k]) += (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
                // H_joint(MAX_FRAMES * 6 + vertexIndex[j], MAX_FRAMES * 6 + vertexIndex[k]) += (MESH_REGU / (VERTEX_WIDTH * VERTEX_HEIGH)) * (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
            }
        }
    }

    hg.count = sceneMesh.triangle_index_to_vertices_indeces.size();

    return hg;

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

void rayDepthMeshSceneCPU::optPose(frameCPU &frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    tic_toc t;
    Error e;

    for (int lvl = 4; lvl >= 1; lvl--)
    {
        std::cout << "*************************lvl " << lvl << std::endl;
        t.tic();
        Sophus::SE3f best_pose = frame.pose;
        e = computeError(frame, lvl);
        float last_error = e.error / e.count;

        std::cout << "init error " << last_error << " time " << t.toc() << std::endl;

        for (int it = 0; it < maxIterations[lvl]; it++)
        {
            t.tic();
            HGPose hg = computeHGPose(frame, lvl);
            hg.H /= hg.count;
            hg.G /= hg.count;
            std::cout << "HGPose time " << t.toc() << std::endl;

            float lambda = 0.0;
            int n_try = 0;
            while (true)
            {
                Eigen::Matrix<float, 6, 6> H_lambda;
                H_lambda = hg.H;

                for (int j = 0; j < 6; j++)
                    H_lambda(j, j) *= 1.0 + lambda;

                Eigen::Matrix<float, 6, 1> inc = H_lambda.ldlt().solve(hg.G);

                // Sophus::SE3f new_pose = frame.pose * Sophus::SE3f::exp(inc_pose);
                frame.pose = best_pose * Sophus::SE3f::exp(inc).inverse();
                // Sophus::SE3f new_pose = Sophus::SE3f::exp(inc_pose).inverse() * frame.pose;

                t.tic();
                e = computeError(frame, lvl);
                float error = e.error / e.count;
                std::cout << "new error " << error << " time " << t.toc() << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda

                    best_pose = frame.pose;
                    float p = error / last_error;

                    if (lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

                    last_error = error;

                    if (p > 0.999f)
                    {
                        // if converged, do next level
                        it = maxIterations[lvl];
                    }
                    // if update accepted, do next iteration
                    break;
                }
                else
                {
                    frame.pose = best_pose;

                    n_try++;

                    if (lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= std::pow(2.0, n_try);

                    // reject update, increase lambda, use un-updated data
                    // std::cout << "update rejected " << std::endl;

                    if (!(inc.dot(inc) > 1e-8))
                    // if(!(inc.dot(inc) > 1e-6))
                    {
                        // std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
                        // if too small, do next level!
                        it = maxIterations[lvl];
                        break;
                    }
                }
            }
        }
    }
}

void rayDepthMeshSceneCPU::optMap(std::vector<frameCPU> &frames)
{
    tic_toc t;

    Error e;
    Error e_regu;
    HGPoseMapMesh hg;
    HGPoseMapMesh hg_regu;

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();
        std::vector<std::array<float, 3>> best_vertices = sceneMesh.vertices;

        e.setZero();
        for (std::size_t i = 0; i < frames.size(); i++)
            e += computeError(frames[i], lvl);
        e.error /= e.count;

        e_regu = errorRegu();
        e_regu.error /= e_regu.count;

        float last_error = e.error + meshRegularization * e_regu.error;

        std::cout << "--------lvl " << lvl << " initial error " << last_error << " " << t.toc() << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            t.tic();

            hg.setZero();
            for (std::size_t i = 0; i < frames.size(); i++)
                hg += computeHGMap(frames[i], lvl);
            hg.H /= hg.count;
            hg.G /= hg.count;

            hg_regu = HGRegu();
            hg_regu.H /= hg_regu.count;
            hg_regu.G /= hg_regu.count;

            hg.H += meshRegularization * hg_regu.H;
            hg.G += meshRegularization * hg_regu.G;
            hg.G_count += hg_regu.G_count;

            // check that the hessian is nonsingular
            // if it is "fix" it
            for (int i = 0; i < hg.G.size(); i++)
            {
                int gcount = hg.G_count(i);
                if (gcount == 0)
                    hg.H.coeffRef(i, i) = 1.0;
                // else
                //     hgmap.G_depth(i) /= float(gcount);
            }

            std::cout << "HG time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                Eigen::SparseMatrix<float> H_lambda = hg.H;

                for (int j = 0; j < H_lambda.rows(); j++)
                {
                    H_lambda.coeffRef(j, j) *= (1.0 + lambda);
                }

                t.tic();

                H_lambda.makeCompressed();
                // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
                Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
                // Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
                // Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;
                solver.analyzePattern(H_lambda);
                // std::cout << solver.info() << std::endl;
                solver.factorize(H_lambda);
                if (solver.info() != Eigen::Success)
                {
                    // some problem i have still to debug
                    it = maxIterations;
                    break;
                }
                // std::cout << solver.lastErrorMessage() << std::endl;
                Eigen::VectorXf inc = -solver.solve(hg.G);
                // inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                // inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                // inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

                std::cout << "solve time " << t.toc() << std::endl;

                for (int index = 0; index < sceneMesh.vertices.size(); index++)
                {
                    vertices[index][2] = best_vertices[index][2] + inc(index);
                    if (vertices[index][2] < 0.001 || vertices[index][2] > 100.0)
                        vertices[index][2] = best_vertices[index][2];
                }

                t.tic();

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                    e += computeError(frames[i], lvl);
                e.error /= e.count;

                e_regu = errorRegu();
                e_regu.error /= e_regu.count;

                float error = e.error + meshRegularization * e_regu.error;

                std::cout << "new error " << error << " " << t.toc() << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    best_vertices = sceneMesh.vertices;

                    float p = error / last_error;

                    if (lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

                    last_error = error;

                    if (p > 0.999f)
                    {
                        // std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
                        //  if converged, do next level
                        it = maxIterations;
                    }

                    // if update accepted, do next iteration
                    break;
                }
                else
                {
                    sceneMesh.vertices = best_vertices;

                    n_try++;

                    if (lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0; // std::pow(2.0, n_try);

                    // reject update, increase lambda, use un-updated data

                    if (inc.dot(inc) < 1e-8)
                    {
                        // if too small, do next level!
                        it = maxIterations;
                        break;
                    }
                }
            }
        }
    }
}

void rayDepthMeshSceneCPU::optPoseMap(std::vector<frameCPU> &frames)
{
    tic_toc t;
    for (int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();
        std::vector<Sophus::SE3f> best_poses;
        for (size_t i = 0; i < frames.size(); i++)
            best_poses.push_back(frames[i].pose);
        std::vector<std::array<float, 3>> best_vertices = sceneMesh.vertices;

        Error last_error;
        for (std::size_t i = 0; i < frames.size(); i++)
            last_error += computeError(frames[i], lvl);
        // last_error += scene.errorRegu();

        std::cout << "--------lvl " << lvl << " initial error " << last_error.error << " " << t.toc() << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            t.tic();

            HGPoseMapMesh hg;

            for (std::size_t i = 0; i < frames.size(); i++)
                computeHGPoseMap(frames[i], hg, i, lvl);

            // check that the hessian is nonsingular
            // if it is "fix" it
            for (int i = 0; i < hg.G.size(); i++)
            {
                int gcount = hg.G_count(i);
                if (gcount == 0)
                    hg.H.coeffRef(i, i) = 1.0;
                // else
                //     hgmap.G_depth(i) /= float(gcount);
            }

            std::cout << "HG time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                Eigen::SparseMatrix<float> H_lambda = hg.H;

                for (int j = 0; j < H_lambda.rows(); j++)
                {
                    H_lambda.coeffRef(j, j) *= (1.0 + lambda);
                }

                t.tic();

                H_lambda.makeCompressed();
                // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
                Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
                // Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
                // Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;
                solver.analyzePattern(H_lambda);
                // std::cout << solver.info() << std::endl;
                solver.factorize(H_lambda);
                if (solver.info() != Eigen::Success)
                {
                    it = maxIterations;
                    break;
                }

                Eigen::VectorXf inc = solver.solve(hg.G);

                std::cout << "solve time " << t.toc() << std::endl;

                // update poses
                for (size_t i = 0; i < frames.size(); i++)
                {
                    Eigen::Matrix<float, 6, 1> pose_inc;
                    for (int j = 0; j < 6; j++)
                        pose_inc(j) = inc(j + i * 6);
                    frames[i].pose = best_poses[i] * Sophus::SE3f::exp(pose_inc).inverse();
                }

                // update map
                for (int index = 0; index < sceneMesh.vertices.size(); index++)
                {
                    // I have to check this - sign in inc
                    //  and maybe the inverse in pose_inc
                    vertices[index][2] = best_vertices[index][2] - inc(index + frames.size() * 6);
                    if (vertices[index][2] < 0.001 || vertices[index][2] > 100.0)
                        vertices[index][2] = vertices[index][2];
                }

                t.tic();

                Error error;
                for (std::size_t i = 0; i < frames.size(); i++)
                    error += computeError(frames[i], lvl);
                // error += errorRegu();

                std::cout << "new error " << error.error << " " << t.toc() << std::endl;

                if (error.error < last_error.error)
                {
                    // accept update, decrease lambda
                    for (size_t i = 0; i < frames.size(); i++)
                        best_poses[i] = frames[i].pose;
                    best_vertices = sceneMesh.vertices;

                    float p = error.error / last_error.error;

                    if (lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

                    last_error = error;

                    if (p > 0.999f)
                    {
                        //  if converged, do next level
                        it = maxIterations;
                    }

                    // if update accepted, do next iteration
                    break;
                }
                else
                {
                    for (size_t i = 0; i < frames.size(); i++)
                        frames[i].pose = best_poses[i];
                    sceneMesh.vertices = best_vertices;

                    n_try++;

                    if (lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0; // std::pow(2.0, n_try);

                    // reject update, increase lambda, use un-updated data

                    if (inc.dot(inc) < 1e-8)
                    {
                        // if too small, do next level!
                        it = maxIterations;
                        break;
                    }
                }
            }
        }
    }
}
