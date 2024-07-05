#include "optimizers/meshOptimizerCPU.h"
#include <math.h>
#include "utils/tictoc.h"

meshOptimizerCPU::meshOptimizerCPU(camera &_cam)
    : z_buffer(_cam.width, _cam.height, -1.0),
      keyframe(_cam.width, _cam.height)
{
    cam[0] = _cam;
    for (int i = 1; i < MAX_LEVELS; i++)
    {
        cam[i] = _cam;
        cam[i].resize(1.0 / std::pow(2.0, i));
    }

    multiThreading = false;
    meshRegularization = 200.0;
}

void meshOptimizerCPU::init(frameCPU &frame, dataCPU<float> &idepth)
{
    frame.copyTo(keyframe);
    keyframeMesh.init(cam[0], idepth, 0);
}

void meshOptimizerCPU::renderIdepth(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    keyframeMesh.computeTexCoords(cam[lvl]);
    MeshCPU frameMesh = keyframeMesh.getCopy();
    frameMesh.transform(pose * keyframe.pose.inverse());
    frameMesh.computeTexCoords(cam[lvl]);

    std::vector<unsigned int> trianglesIds = frameMesh.getTrianglesIds();

    // for each triangle
    for (int t_index = 0; t_index < (int)trianglesIds.size(); t_index++)
    {
        unsigned int t_id = trianglesIds[t_index];
        // Triangle kf_tri = keyframeMesh.triangles[t_id];
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        // if (kf_tri.isBackFace())
        //     continue;
        Triangle3D f_tri3d = frameMesh.getCartesianTriangle(t_id);
        Triangle2D f_tri2d = frameMesh.getTexCoordTriangle(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_tri2d.getArea() < 1.0)
            continue;

        f_tri2d.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri2d.getMinMax();

        for (int y = int(minmax[0](1)) - 1; y <= int(minmax[1](1)) + 1; y++)
        {
            for (int x = int(minmax[0](0)) - 1; x <= int(minmax[1](0)) + 1; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam[lvl].isPixVisible(f_pix))
                    continue;

                f_tri2d.computeBarycentric(f_pix);
                if (!f_tri2d.isBarycentricOk())
                    continue;

                float depth = f_tri2d.interpolate(f_tri3d.vertices[0](2), f_tri3d.vertices[1](2), f_tri3d.vertices[2](2));

                if (depth <= 0.0)
                    continue;

                // float l_idepth = idepth.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != idepth.nodata)
                //     continue;

                buffer.set(1.0 / depth, f_pix(1), f_pix(0), lvl);
            }
        }
    }
}

Error meshOptimizerCPU::computeError(frameCPU &frame, int lvl)
{
    Error e;

    keyframeMesh.computeTexCoords(cam[lvl]);
    MeshCPU frameMesh = keyframeMesh.getCopy();
    frameMesh.transform(frame.pose * keyframe.pose.inverse());
    frameMesh.computeTexCoords(cam[lvl]);

    std::vector<unsigned int> trianglesIds = frameMesh.getTrianglesIds();

    if (multiThreading)
    {
        errorTreadReduce.reduce(boost::bind(&meshOptimizerCPU::errorPerIndex, this, frame, frameMesh, trianglesIds, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, trianglesIds.size(), 0);
        e = errorTreadReduce.stats;
    }
    else
    {
        errorPerIndex(frame, frameMesh, trianglesIds, lvl, 0, trianglesIds.size(), &e, 0);
    }

    // if (e.count > 0)
    //     e.error /= e.count;

    // return e.error;
    return e;
}

void meshOptimizerCPU::errorPerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, Error *e, int tid)
{
    // z_buffer.reset(lvl);
    // for each triangle
    for (std::size_t index = tmin; index < tmax; index++)
    {
        unsigned int t_id = trisIds[index];

        Triangle2D kf_tri = keyframeMesh.getTexCoordTriangle(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        float kf_tri_area = kf_tri.getArea();
        if (kf_tri_area < 1.0)
            continue;
        // std::array<float, 3> kf_tri_angles = kf_tri.getAngles();
        // if (fabs(kf_tri_angles[0]) < MIN_TRIANGLE_ANGLE || fabs(kf_tri_angles[1]) < MIN_TRIANGLE_ANGLE || fabs(kf_tri_angles[2]) < MIN_TRIANGLE_ANGLE)
        //     continue;
        Triangle2D f_tri = frameMesh.getTexCoordTriangle(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        float f_tri_area = f_tri.getArea();
        if (f_tri_area < 1.0)
            continue;
        // std::array<float, 3> f_tri_angles = f_tri.getAngles();
        // if (fabs(f_tri_angles[0]) < MIN_TRIANGLE_ANGLE || fabs(f_tri_angles[1]) < MIN_TRIANGLE_ANGLE || fabs(f_tri_angles[2]) < MIN_TRIANGLE_ANGLE)
        //     continue;

        f_tri.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri.getMinMax();

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam[lvl].isPixVisible(f_pix))
                    continue;

                f_tri.computeBarycentric(f_pix);
                if (!f_tri.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri.interpolate(kf_tri.vertices[0], kf_tri.vertices[1], kf_tri.vertices[2]);
                if (!cam[lvl].isPixVisible(kf_pix))
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

void meshOptimizerCPU::renderError(frameCPU &frame, dataCPU<float> &buffer, int lvl)
{
    keyframeMesh.computeTexCoords(cam[lvl]);

    MeshCPU frameMesh = keyframeMesh.getCopy();
    frameMesh.transform(frame.pose * keyframe.pose.inverse());
    frameMesh.computeTexCoords(cam[lvl]);

    std::vector<unsigned int> trisIds = frameMesh.getTrianglesIds();

    // for each triangle
    for (int t_index = 0; t_index < (int)trisIds.size(); t_index++)
    {
        unsigned int t_id = trisIds[t_index];

        Triangle2D kf_tri = keyframeMesh.getTexCoordTriangle(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (kf_tri.getArea() < 1.0)
            continue;
        Triangle2D f_tri = frameMesh.getTexCoordTriangle(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_tri.getArea() < 1.0)
            continue;

        f_tri.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri.getMinMax();

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam[lvl].isPixVisible(f_pix))
                    continue;
                f_tri.computeBarycentric(f_pix);
                if (!f_tri.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri.interpolate(kf_tri.vertices[0], kf_tri.vertices[1], kf_tri.vertices[2]);
                if (!cam[lvl].isPixVisible(kf_pix))
                    continue;

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame.image.get(f_pix(1), f_pix(0), lvl));
                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata)
                    continue;

                float residual = f_i - kf_i + 255.0;
                // float residual_2 = residual * residual;

                // z buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                buffer.set(residual, y, x, lvl);
            }
        }
    }
}

void meshOptimizerCPU::renderImage(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    keyframeMesh.computeTexCoords(cam[lvl]);

    MeshCPU frameMesh = keyframeMesh.getCopy();
    frameMesh.transform(pose * keyframe.pose.inverse());
    frameMesh.computeTexCoords(cam[lvl]);

    std::vector<unsigned int> trisIds = frameMesh.getTrianglesIds();

    // for each triangle
    for (std::size_t index = 0; index < trisIds.size(); index++)
    {
        unsigned int t_id = trisIds[index];

        Triangle2D kf_tri = keyframeMesh.getTexCoordTriangle(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (kf_tri.getArea() < 1.0)
            continue;

        Triangle2D f_tri = frameMesh.getTexCoordTriangle(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_tri.getArea() < 1.0)
            continue;

        f_tri.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri.getMinMax();

        for (int y = int(minmax[0](1)) - 1; y <= int(minmax[1](1)) + 1; y++)
        {
            for (int x = int(minmax[0](0)) - 1; x <= int(minmax[1](0)) + 1; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam[lvl].isPixVisible(f_pix))
                    continue;

                f_tri.computeBarycentric(f_pix);
                if (!f_tri.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri.interpolate(kf_tri.vertices[0], kf_tri.vertices[1], kf_tri.vertices[2]);
                if (!cam[lvl].isPixVisible(kf_pix))
                    continue;

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                if (kf_i == keyframe.image.nodata)
                    continue;

                // float f_idepth = 1.0 / f_ver(2);

                // z buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                buffer.set(kf_i, y, x, lvl);
            }
        }
    }
}

void meshOptimizerCPU::renderDebug(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    keyframeMesh.computeTexCoords(cam[lvl]);

    MeshCPU frameMesh = keyframeMesh.getCopy();
    frameMesh.transform(pose * keyframe.pose.inverse());
    frameMesh.computeTexCoords(cam[lvl]);

    std::vector<unsigned int> trisIds = frameMesh.getTrianglesIds();

    // for each triangle
    for (std::size_t index = 0; index < trisIds.size(); index++)
    {
        unsigned int t_id = trisIds[index];
        // Triangle kf_tri = keyframeMesh.triangles[index];
        //  if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //      continue;
        // if (kf_tri.isBackFace())
        //     continue;
        Triangle2D f_tri = frameMesh.getTexCoordTriangle(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_tri.getArea() < 1.0)
            continue;

        f_tri.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri.getMinMax();

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam[lvl].isPixVisible(f_pix))
                    continue;

                f_tri.computeBarycentric(f_pix);
                if (!f_tri.isBarycentricOk())
                    continue;

                bool isLine = f_tri.isLine();

                // z buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                if (isLine)
                    buffer.set(1.0, y, x, lvl);
                else
                    buffer.set(0.0, y, x, lvl);
            }
        }
    }
}

HGMapped meshOptimizerCPU::computeHGPose(frameCPU &frame, int lvl)
{
    HGMapped hg;

    keyframeMesh.computeTexCoords(cam[lvl]);

    MeshCPU frameMesh = keyframeMesh.getCopy();
    frameMesh.transform(frame.pose * keyframe.pose.inverse());
    frameMesh.computeTexCoords(cam[lvl]);

    std::vector<unsigned int> trisIds = frameMesh.getTrianglesIds();

    if (multiThreading)
    {
        hgMappedTreadReduce.reduce(boost::bind(&meshOptimizerCPU::HGPosePerIndex, this, frame, frameMesh, trisIds, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, trisIds.size(), 0);
        hg = hgMappedTreadReduce.stats;
    }
    else
    {
        HGPosePerIndex(frame, frameMesh, trisIds, lvl, 0, trisIds.size(), &hg, 0);
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

void meshOptimizerCPU::HGPosePerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, HGMapped *hg, int tid)
{
    // z_buffer.reset(lvl);

    float min_area = (float(cam[lvl].width) / MESH_WIDTH) * (float(cam[lvl].height) / MESH_HEIGHT) / 4;
    float min_angle = M_PI / 16.0;

    // for each triangle
    for (int index = tmin; index < tmax; index++)
    {
        unsigned int t_id = trisIds[index];

        Triangle2D kf_tri = keyframeMesh.getTexCoordTriangle(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (kf_tri.getArea() < min_area)
            continue;
        std::array<float, 3> kf_angle = kf_tri.getAngles();
        if (fabs(kf_angle[0]) < min_angle || fabs(kf_angle[1]) < min_angle || fabs(kf_angle[2]) < min_angle)
            continue;
        Triangle3D f_tri_3d = frameMesh.getCartesianTriangle(t_id);
        Triangle2D f_tri_2d = frameMesh.getTexCoordTriangle(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> f_angle = f_tri_2d.getAngles();
        if (fabs(f_angle[0]) < min_angle || fabs(f_angle[1]) < min_angle || fabs(f_angle[2]) < min_angle)
            continue;

        f_tri_2d.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri_2d.getMinMax();

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam[lvl].isPixVisible(f_pix))
                    continue;

                f_tri_2d.computeBarycentric(f_pix);

                if (!f_tri_2d.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri_2d.interpolate(kf_tri.vertices[0], kf_tri.vertices[1], kf_tri.vertices[2]);
                if (!cam[lvl].isPixVisible(kf_pix))
                    continue;

                Eigen::Vector3f f_ver = f_tri_2d.interpolate(f_tri_3d.vertices[0], f_tri_3d.vertices[1], f_tri_3d.vertices[2]);

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

                float v0 = d_f_i_d_pix(0) * cam[lvl].fx / f_ver(2);
                float v1 = d_f_i_d_pix(1) * cam[lvl].fy / f_ver(2);
                float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_i_d_tra = Eigen::Vector3f(v0, v1, v2);
                Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);

                float residual = (f_i - kf_i);
                // float residual_2 = residual * residual;

                Eigen::Matrix<float, 6, 1> J;
                J << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                hg->count++;
                for (int i = 0; i < 6; i++)
                {
                    hg->G.add(J[i] * residual, i - 6);
                    // hg->G[i - 6] = J[i] * residual;
                    for (int j = i; j < 6; j++)
                    {
                        float jj = J[i] * J[j];
                        hg->H.add(jj, i - 6, j - 6);
                        hg->H.add(jj, j - 6, i - 6);
                    }
                }
            }
        }
    }
}

HGMapped meshOptimizerCPU::computeHGMap(frameCPU &frame, int lvl)
{
    HGMapped hg;

    keyframeMesh.computeTexCoords(cam[lvl]);

    MeshCPU frameMesh = keyframeMesh.getCopy();
    frameMesh.transform(frame.pose * keyframe.pose.inverse());
    frameMesh.computeTexCoords(cam[lvl]);

    std::vector<unsigned int> trisIds = frameMesh.getTrianglesIds();

    if (multiThreading)
    {
        hgMappedTreadReduce.reduce(boost::bind(&meshOptimizerCPU::HGMapPerIndex, this, frame, frameMesh, trisIds, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, trisIds.size(), 0);
        hg = hgMappedTreadReduce.stats;
    }
    else
    {
        HGMapPerIndex(frame, frameMesh, trisIds, lvl, 0, trisIds.size(), &hg, 0);
    }

    return hg;
}

void meshOptimizerCPU::HGMapPerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, HGMapped *hg, int tid)
{
    // z_buffer.reset(lvl);

    float min_area = (float(cam[lvl].width) / MESH_WIDTH) * (float(cam[lvl].height) / MESH_HEIGHT) / 4;
    float min_angle = M_PI / 16.0;

    // for each triangle
    for (std::size_t index = tmin; index < tmax; index++)
    {
        unsigned int t_id = trisIds[index];

        std::array<unsigned int, 3> v_ids = keyframeMesh.getTriangleIndices(t_id);

        Triangle3D kf_tri_3d = keyframeMesh.getCartesianTriangle(t_id);
        Triangle2D kf_tri_2d = keyframeMesh.getTexCoordTriangle(t_id);
        if (kf_tri_3d.vertices[0](2) <= 0.0 || kf_tri_3d.vertices[1](2) <= 0.0 || kf_tri_3d.vertices[2](2) <= 0.0)
            continue;
        if (kf_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
        if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
            continue;

        Triangle3D f_tri_3d = frameMesh.getCartesianTriangle(t_id);
        Triangle2D f_tri_2d = frameMesh.getTexCoordTriangle(t_id);
        if (f_tri_3d.vertices[0](2) <= 0.0 || f_tri_3d.vertices[1](2) <= 0.0 || f_tri_3d.vertices[2](2) <= 0.0)
            continue;
        if (f_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
        if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
            continue;

        f_tri_2d.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri_2d.getMinMax();

        Eigen::Vector3f n_p[3];
        n_p[0] = (kf_tri_3d.vertices[0] - kf_tri_3d.vertices[1]).cross(kf_tri_3d.vertices[2] - kf_tri_3d.vertices[1]);
        n_p[1] = (kf_tri_3d.vertices[1] - kf_tri_3d.vertices[0]).cross(kf_tri_3d.vertices[2] - kf_tri_3d.vertices[0]);
        n_p[2] = (kf_tri_3d.vertices[2] - kf_tri_3d.vertices[1]).cross(kf_tri_3d.vertices[0] - kf_tri_3d.vertices[1]);

        Eigen::Vector3f pw2mpw1[3];
        pw2mpw1[0] = (kf_tri_3d.vertices[2] - kf_tri_3d.vertices[1]);
        pw2mpw1[1] = (kf_tri_3d.vertices[2] - kf_tri_3d.vertices[0]);
        pw2mpw1[2] = (kf_tri_3d.vertices[0] - kf_tri_3d.vertices[1]);

        float n_p_dot_point[3];
        n_p_dot_point[0] = n_p[0].dot(kf_tri_3d.vertices[1]);
        n_p_dot_point[1] = n_p[1].dot(kf_tri_3d.vertices[0]);
        n_p_dot_point[2] = n_p[2].dot(kf_tri_3d.vertices[1]);

        Eigen::Vector3f pr_p[3];
        pr_p[0] = kf_tri_3d.vertices[1];
        pr_p[1] = kf_tri_3d.vertices[0];
        pr_p[2] = kf_tri_3d.vertices[1];

        Eigen::Vector3f d_n_d_z[3];
        float d_z_d_iz[3];
        for (int i = 0; i < 3; i++)
        {
            d_n_d_z[i] = kf_tri_3d.vertices[i].cross(pw2mpw1[i]);
            // d_z_d_iz[i] = -1.0 / (kf_tri_idepth[i] * kf_tri_idepth[i]);
            d_z_d_iz[i] = -(kf_tri_3d.vertices[i](2) * kf_tri_3d.vertices[i](2));
        }

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam[lvl].isPixVisible(f_pix))
                    continue;

                f_tri_2d.computeBarycentric(f_pix);
                if (!f_tri_2d.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri_2d.interpolate(kf_tri_2d.vertices[0], kf_tri_2d.vertices[1], kf_tri_2d.vertices[2]);
                if (!cam[lvl].isPixVisible(kf_pix))
                    continue;

                Eigen::Vector3f kf_ver = f_tri_2d.interpolate(kf_tri_3d.vertices[0], kf_tri_3d.vertices[1], kf_tri_3d.vertices[2]);
                Eigen::Vector3f f_ver = f_tri_2d.interpolate(f_tri_3d.vertices[0], f_tri_3d.vertices[1], f_tri_3d.vertices[2]);

                if (kf_ver(2) <= 0.0 || f_ver(2) <= 0.0)
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

                Eigen::Vector3f d_f_i_d_f_ver;
                d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam[lvl].fx / f_ver(2);
                d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam[lvl].fy / f_ver(2);
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

                    (*hg).G.add(J[i] * error, v_ids[i]);
                    //(*hg).G[v_ids[i]] += J[i] * error;

                    for (int j = i; j < 3; j++)
                    {
                        float jj = J[i] * J[j];
                        (*hg).H.add(jj, v_ids[i], v_ids[j]);
                        (*hg).H.add(jj, v_ids[j], v_ids[i]);
                        //(*hg).H[v_ids[i]][v_ids[j]] += jj;
                        //(*hg).H[v_ids[j]][v_ids[i]] += jj;
                    }
                }
            }
        }
    }
}

HGMapped meshOptimizerCPU::computeHGPoseMap(frameCPU &frame, int frame_index, int lvl)
{
    HGMapped hg;

    keyframeMesh.computeTexCoords(cam[lvl]);

    MeshCPU frameMesh = keyframeMesh.getCopy();
    frameMesh.transform(frame.pose * keyframe.pose.inverse());
    frameMesh.computeTexCoords(cam[lvl]);

    std::vector<unsigned int> trisIds = frameMesh.getTrianglesIds();

    if (multiThreading)
    {
        hgMappedTreadReduce.reduce(boost::bind(&meshOptimizerCPU::HGPoseMapPerIndex, this, frame, frameMesh, trisIds, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, trisIds.size(), 0);
        hg = hgMappedTreadReduce.stats;
    }
    else
    {
        HGPoseMapPerIndex(frame, frameMesh, trisIds, lvl, 0, trisIds.size(), &hg, 0);
    }

    return hg;
}

void meshOptimizerCPU::HGPoseMapPerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, HGMapped *hg, int tid)
{
    // z_buffer.reset(lvl);

    float min_area = (float(cam[lvl].width) / MESH_WIDTH) * (float(cam[lvl].height) / MESH_HEIGHT) / 4;
    float min_angle = M_PI / 16.0;

    for (std::size_t index = tmin; index < tmax; index++)
    {
        unsigned int t_id = trisIds[index];

        std::array<unsigned int, 3> v_ids = keyframeMesh.getTriangleIndices(t_id);

        Triangle2D kf_tri_2d = keyframeMesh.getTexCoordTriangle(t_id);
        Triangle3D kf_tri_3d = keyframeMesh.getCartesianTriangle(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (kf_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
        if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
            continue;

        Triangle3D f_tri_3d = frameMesh.getCartesianTriangle(t_id);
        Triangle2D f_tri_2d = frameMesh.getTexCoordTriangle(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
        if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
            continue;

        f_tri_2d.computeTinv();
        std::array<Eigen::Vector2f, 2> minmax = f_tri_2d.getMinMax();

        Eigen::Vector3f n_p[3];
        n_p[0] = (kf_tri_3d.vertices[0] - kf_tri_3d.vertices[1]).cross(kf_tri_3d.vertices[2] - kf_tri_3d.vertices[1]);
        n_p[1] = (kf_tri_3d.vertices[1] - kf_tri_3d.vertices[0]).cross(kf_tri_3d.vertices[2] - kf_tri_3d.vertices[0]);
        n_p[2] = (kf_tri_3d.vertices[2] - kf_tri_3d.vertices[1]).cross(kf_tri_3d.vertices[0] - kf_tri_3d.vertices[1]);

        Eigen::Vector3f pw2mpw1[3];
        pw2mpw1[0] = (kf_tri_3d.vertices[2] - kf_tri_3d.vertices[1]);
        pw2mpw1[1] = (kf_tri_3d.vertices[2] - kf_tri_3d.vertices[0]);
        pw2mpw1[2] = (kf_tri_3d.vertices[0] - kf_tri_3d.vertices[1]);

        float n_p_dot_point[3];
        n_p_dot_point[0] = n_p[0].dot(kf_tri_3d.vertices[1]);
        n_p_dot_point[1] = n_p[1].dot(kf_tri_3d.vertices[0]);
        n_p_dot_point[2] = n_p[2].dot(kf_tri_3d.vertices[1]);

        Eigen::Vector3f pr_p[3];
        pr_p[0] = kf_tri_3d.vertices[1];
        pr_p[1] = kf_tri_3d.vertices[0];
        pr_p[2] = kf_tri_3d.vertices[1];

        Eigen::Vector3f d_n_d_z[3];
        float d_z_d_iz[3];
        for (int i = 0; i < 3; i++)
        {
            d_n_d_z[i] = kf_tri_3d.vertices[i].cross(pw2mpw1[i]);
            // d_z_d_iz[i] = -1.0 / (kf_tri_idepth[i] * kf_tri_idepth[i]);
            d_z_d_iz[i] = -(kf_tri_3d.vertices[i](2) * kf_tri_3d.vertices[i](2));
        }

        for (int y = minmax[0](1); y <= minmax[1](1); y++)
        {
            for (int x = minmax[0](0); x <= minmax[1](0); x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam[lvl].isPixVisible(f_pix))
                    continue;

                f_tri_2d.computeBarycentric(f_pix);
                if (!f_tri_2d.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri_2d.interpolate(kf_tri_2d.vertices[0], kf_tri_2d.vertices[1], kf_tri_2d.vertices[2]);
                if (!cam[lvl].isPixVisible(kf_pix))
                    continue;

                Eigen::Vector3f kf_ver = f_tri_2d.interpolate(kf_tri_3d.vertices[0], kf_tri_3d.vertices[1], kf_tri_3d.vertices[2]);
                Eigen::Vector3f f_ver = f_tri_2d.interpolate(f_tri_3d.vertices[0], f_tri_3d.vertices[1], f_tri_3d.vertices[2]);

                if (kf_ver(2) <= 0.0 || f_ver(2) <= 0.0)
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

                Eigen::Vector3f d_f_i_d_f_ver;
                d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam[lvl].fx / f_ver(2);
                d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam[lvl].fy / f_ver(2);
                // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) / f_ver(2) + d_f_i_d_f_ver(1) * f_ver(1) / f_ver(2));
                d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_i_d_tra = d_f_i_d_f_ver;
                Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                Eigen::Matrix<float, 6, 1> J_pose;
                J_pose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

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

                for (int i = 0; i < 6; i++)
                {
                    (*hg).G.add(J_pose[i] * error, i - (frame.id + 1) * 6);

                    for (int j = i; j < 6; j++)
                    {
                        float jj = J_pose[i] * J_pose[j];
                        (*hg).H.add(jj, i - (frame.id + 1) * 6, j - (frame.id + 1) * 6);
                        (*hg).H.add(jj, j - (frame.id + 1) * 6, i - (frame.id + 1) * 6);
                    }
                }

                for (int i = 0; i < 3; i++)
                {
                    // if the jacobian is 0
                    // we really cannot say anything about the depth
                    // can make the hessian non-singular
                    if (J[i] == 0)
                        continue;
                    (*hg).G.add(J[i] * error, v_ids[i]);
                    //(*hg).G[v_ids[i]] += J[i] * error;

                    for (int j = i; j < 3; j++)
                    {
                        float jj = J[i] * J[j];
                        (*hg).H.add(jj, v_ids[i], v_ids[j]);
                        (*hg).H.add(jj, v_ids[j], v_ids[i]);
                        //(*hg).H[v_ids[i]][v_ids[j]] += jj;
                        //(*hg).H[v_ids[j]][v_ids[i]] += jj;
                    }
                }
            }
        }
    }
}

Error meshOptimizerCPU::errorRegu()
{
    Error error;

    std::vector<unsigned int> triIds = keyframeMesh.getTrianglesIds();
    // for each triangle
    for (size_t index = 0; index < triIds.size(); index++)
    {
        unsigned int id = triIds[index];
        std::array<unsigned int, 3> tri = keyframeMesh.getTriangleIndices(id);

        float idepth[3];
        for (int j = 0; j < 3; j++)
            idepth[j] = keyframeMesh.getVerticeIdepth(tri[j]);

        if (idepth[0] <= 0.0 || idepth[1] <= 0.0 || idepth[2] <= 0.0)
            continue;

        float diff1 = idepth[0] - idepth[1];
        float diff2 = idepth[0] - idepth[2];
        float diff3 = idepth[1] - idepth[2];

        error.error += diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    }
    // divided by the number of triangles
    // we don't want to have less error if we have less triangles
    error.count = triIds.size();
    return error;
}

HGMapped meshOptimizerCPU::HGRegu()
{
    HGMapped hg;

    std::vector<unsigned int> triIds = keyframeMesh.getTrianglesIds();

    for (size_t i = 0; i < triIds.size(); i++)
    {
        unsigned int t_id = triIds[i];

        std::array<unsigned int, 3> v_ids = keyframeMesh.getTriangleIndices(t_id);

        float idepth[3];
        for (int j = 0; j < 3; j++)
            idepth[j] = keyframeMesh.getVerticeIdepth(v_ids[j]);

        if (idepth[0] <= 0.0 || idepth[1] <= 0.0 || idepth[2] <= 0.0)
            continue;

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
            hg.G[v_ids[j]] += (diff1 * J1[j] + diff2 * J2[j] + diff3 * J3[j]);
            for (int k = 0; k < 3; k++)
            {
                hg.H[v_ids[j]][v_ids[k]] += (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
            }
        }
    }

    hg.count = triIds.size();

    return hg;
}

void meshOptimizerCPU::optPose(frameCPU &frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    tic_toc t;
    Error e;
    HGMapped hg;

    for (int lvl = 2; lvl >= 1; lvl--)
    {
        std::cout << "*************************lvl " << lvl << std::endl;
        t.tic();
        Sophus::SE3f best_pose = frame.pose;
        e.setZero();
        e = computeError(frame, lvl);
        float last_error = e.error / e.count;

        std::cout << "init error " << last_error << " time " << t.toc() << std::endl;

        for (int it = 0; it < maxIterations[lvl]; it++)
        {
            t.tic();
            hg.setZero();
            hg += computeHGPose(frame, lvl);

            std::vector<int> ids = hg.G.getIds();

            Eigen::VectorXf G = hg.G.toEigen(ids);
            Eigen::SparseMatrix<float> H = hg.H.toEigen(ids);

            H /= hg.count;
            G /= hg.count;

            std::cout << "HGPose time " << t.toc() << std::endl;

            float lambda = 0.0;
            int n_try = 0;
            while (true)
            {
                Eigen::Matrix<float, 6, 6> H_lambda;
                H_lambda = H;

                for (int j = 0; j < 6; j++)
                    H_lambda(j, j) *= 1.0 + lambda;

                Eigen::Matrix<float, 6, 1> inc = H_lambda.ldlt().solve(G);

                // Sophus::SE3f new_pose = frame.pose * Sophus::SE3f::exp(inc_pose);
                frame.pose = best_pose * Sophus::SE3f::exp(inc).inverse();
                // Sophus::SE3f new_pose = Sophus::SE3f::exp(inc_pose).inverse() * frame.pose;

                t.tic();
                e.setZero();
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

void meshOptimizerCPU::optMap(std::vector<frameCPU> &frames)
{
    tic_toc t;

    Error e;
    Error e_regu;
    HGMapped hg;
    HGMapped hg_regu;

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();
        // keyframeMesh.toRayIdepth();
        // std::vector<float> best_idepths = keyframeMesh.getVerticesIdepths();

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

            std::vector<int> ids = hg.G.getIds();

            Eigen::VectorXf G = hg.G.toEigen(ids);
            Eigen::SparseMatrix<float> H = hg.H.toEigen(ids);

            H /= hg.count;
            G /= hg.count;

            hg_regu = HGRegu();

            Eigen::VectorXf G_regu = hg_regu.G.toEigen(ids);
            Eigen::SparseMatrix<float> H_regu = hg_regu.H.toEigen(ids);

            H_regu /= hg_regu.count;
            G_regu /= hg_regu.count;

            H += meshRegularization * H_regu;
            G += meshRegularization * G_regu;

            std::cout << "HG time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                Eigen::SparseMatrix<float> H_lambda = H;

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
                Eigen::VectorXf inc = -solver.solve(G);
                // inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                // inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                // inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

                std::cout << "solve time " << t.toc() << std::endl;

                std::vector<float> best_idepths;
                for (int index = 0; index < (int)ids.size(); index++)
                {
                    float best_idepth = keyframeMesh.getVerticeIdepth(ids[index]);
                    float new_idepth = best_idepth + inc(index);
                    if (new_idepth < 0.001 || new_idepth > 100.0)
                        new_idepth = best_idepth;
                    best_idepths.push_back(best_idepth);
                    keyframeMesh.setVerticeIdepth(new_idepth, ids[index]);
                }
                // keyframeMesh.setVerticesIdepths(new_idepths);

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
                    for (int index = 0; index < ids.size(); index++)
                        keyframeMesh.setVerticeIdepth(best_idepths[index], ids[index]);

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

void meshOptimizerCPU::optPoseMap(std::vector<frameCPU> &frames)
{
    tic_toc t;

    Error e;
    Error e_regu;
    HGMapped hg;
    HGMapped hg_regu;

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();

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
                hg += computeHGPoseMap(frames[i], i, lvl);

            std::vector<int> ids = hg.G.getIds();

            Eigen::VectorXf G = hg.G.toEigen(ids);
            Eigen::SparseMatrix<float> H = hg.H.toEigen(ids);

            H /= hg.count;
            G /= hg.count;

            hg_regu = HGRegu();

            Eigen::VectorXf G_regu = hg_regu.G.toEigen(ids);
            Eigen::SparseMatrix<float> H_regu = hg_regu.H.toEigen(ids);

            H_regu /= hg_regu.count;
            G_regu /= hg_regu.count;

            H += meshRegularization * H_regu;
            G += meshRegularization * G_regu;

            std::cout << "HG time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                Eigen::SparseMatrix<float> H_lambda = H;

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
                Eigen::VectorXf inc = -solver.solve(G);
                // inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                // inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                // inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

                std::cout << "solve time " << t.toc() << std::endl;

                // update pose
                std::vector<Sophus::SE3f> best_poses;
                for (size_t i = 0; i < frames.size(); i++)
                {
                    Eigen::Matrix<float, 6, 1> pose_inc;
                    // if ids are in order, like this I get the correct pose increment
                    // have to fix it some better way
                    for (int j = 0; j < 6; j++)
                    {
                        int index = j - (i + 1) * 6 + frames.size() * 6;
                        pose_inc(j) = inc(index);
                    }
                    best_poses.push_back(frames[i].pose);
                    frames[i].pose = frames[i].pose * Sophus::SE3f::exp(pose_inc).inverse();
                }

                // update map
                std::map<unsigned int, float> best_idepths;
                for (int index = 0; index < (int)ids.size(); index++)
                {
                    // negative ids are for the poses
                    if (ids[index] < 0)
                        continue;

                    float best_idepth = keyframeMesh.getVerticeIdepth(ids[index]);
                    float new_idepth = best_idepth + inc(index);
                    if (new_idepth < 0.001 || new_idepth > 100.0)
                        new_idepth = best_idepth;
                    best_idepths[ids[index]] = best_idepth;
                    keyframeMesh.setVerticeIdepth(new_idepth, ids[index]);
                }
                // keyframeMesh.setVerticesIdepths(new_idepths);

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
                    for (int index = 0; index < frames.size(); index++)
                        frames[index].pose = best_poses[index];

                    for (int index = 0; index < ids.size(); index++)
                    {
                        // negative ids are for the poses
                        if (ids[index] < 0)
                            continue;
                        keyframeMesh.setVerticeIdepth(best_idepths[ids[index]], ids[index]);
                    }

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