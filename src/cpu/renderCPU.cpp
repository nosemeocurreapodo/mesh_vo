#include "cpu/renderCPU.h"
#include <math.h>
#include "utils/tictoc.h"

void renderCPU::renderIdepth(Mesh &mesh, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    Mesh frameMesh = mesh;
    frameMesh.transform(pose);
    frameMesh.projectToCamera(cam);

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
        Polygon f_tri3d = frameMesh.getCartesianTriangle(t_id);
        Triangle2D f_tri2d = frameMesh.getTexCoordTriangle(t_id);
        // if (f_tri2d.vertices[0](2) <= 0.0 || f_tri2d.vertices[1](2) <= 0.0 || f_tri2d.vertices[2](2) <= 0.0)
        //      continue;
        if (f_tri2d.getArea() < 1.0)
            continue;

        f_tri2d.computeTinv();
        std::array<int, 4> minmax = f_tri2d.getMinMax();

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;

                f_tri2d.computeBarycentric(f_pix);
                if (!f_tri2d.isBarycentricOk())
                    continue;

                float depth = f_tri2d.interpolate(f_tri3d.vertices[0](2), f_tri3d.vertices[1](2), f_tri3d.vertices[2](2));

                if (depth <= 0.0)
                    continue;

                float z_depth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                if (z_depth > depth && z_depth != z_buffer.nodata)
                    continue;

                buffer.set(1.0 / depth, f_pix(1), f_pix(0), lvl);

                z_buffer.set(depth, f_pix(1), f_pix(0), lvl);
            }
        }
    }
}

void renderCPU::renderIdepth(Mesh &mesh, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    Mesh frameMesh = mesh;
    frameMesh.transform(pose);
    frameMesh.computeTexCoords(cam);

    std::vector<unsigned int> trianglesIds = frameMesh.getTrianglesIds();

    // for each triangle
    for (auto t_id : trianglesIds)
    {
        Polygon f_tri3d = frameMesh.getCartesianTriangle(t_id);
        Triangle2D f_tri2d = frameMesh.getTexCoordTriangle(t_id);

        if (f_tri2d.getArea() < 1.0)
            continue;

        f_tri2d.computeTinv();
        std::array<int, 4> minmax = f_tri2d.getMinMax();

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;

                f_tri2d.computeBarycentric(f_pix);
                if (!f_tri2d.isBarycentricOk())
                    continue;

                // float depth = f_tri2d.interpolate(f_tri3d.vertices[0](2), f_tri3d.vertices[1](2), f_tri3d.vertices[2](2));

                Eigen::Vector3f f_ray = cam.pixToRay(f_pix);

                float depth = 0;
                for (int i = 0; i < 3; i++)
                {
                    depth += f_tri3d.getDepth(f_ray);
                }
                depth /= 3;

                if (depth <= 0.0)
                    continue;

                float z_depth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                if (z_depth > depth && z_depth != z_buffer.nodata)
                    continue;

                buffer.set(1.0 / depth, f_pix(1), f_pix(0), lvl);

                z_buffer.set(depth, f_pix(1), f_pix(0), lvl);
            }
        }
    }
}

void renderCPU::renderImage(Mesh &mesh, camera &cam, dataCPU<float> &image, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    mesh.computeTexCoords(cam);
    Mesh frameMesh = mesh;
    frameMesh.transform(pose);
    frameMesh.computeTexCoords(cam);

    std::vector<unsigned int> trisIds = frameMesh.getTrianglesIds();

    // for each triangle
    for (std::size_t index = 0; index < trisIds.size(); index++)
    {
        unsigned int t_id = trisIds[index];

        Triangle2D kf_tri = mesh.getTexCoordTriangle(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (kf_tri.getArea() < 1.0)
            continue;

        Polygon f_tri3d = frameMesh.getCartesianTriangle(t_id);
        Triangle2D f_tri = frameMesh.getTexCoordTriangle(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_tri.getArea() < 1.0)
            continue;

        f_tri.computeTinv();
        std::array<int, 4> minmax = f_tri.getMinMax();

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;

                f_tri.computeBarycentric(f_pix);
                if (!f_tri.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri.interpolate(kf_tri.vertices[0], kf_tri.vertices[1], kf_tri.vertices[2]);
                if (!cam.isPixVisible(kf_pix))
                    continue;

                float depth = f_tri.interpolate(f_tri3d.vertices[0](2), f_tri3d.vertices[1](2), f_tri3d.vertices[2](2));

                if (depth <= 0.0)
                    continue;

                float kf_i = float(image.get(kf_pix(1), kf_pix(0), lvl));
                if (kf_i == image.nodata)
                    continue;

                float z_depth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                if (z_depth > depth && z_depth != z_buffer.nodata)
                    continue;

                buffer.set(kf_i, y, x, lvl);

                z_buffer.set(depth, f_pix(1), f_pix(0), lvl);
            }
        }
    }
}

void renderCPU::renderDebug(Mesh &mesh, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    Mesh frameMesh = mesh;
    frameMesh.transform(pose);
    frameMesh.computeTexCoords(cam);

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
        std::array<int, 4> minmax = f_tri.getMinMax();

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
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

void renderCPU::renderJMapDepth(Mesh &mesh, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &j_buffer, dataCPU<float> &e_buffer, dataCPU<Eigen::Vector3i> &id_buffer, MapJacobianMethod jacMethod, int lvl)
{
    // z_buffer.reset(lvl);

    float min_area = (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) / 16;
    float min_angle = M_PI / 64.0;

    Mesh frame1Mesh = mesh;
    Mesh frame2Mesh = mesh;

    frame1Mesh.transform(frame1.pose);
    frame1Mesh.computeTexCoords(cam);

    frame2Mesh.transform(frame2.pose);
    frame2Mesh.computeTexCoords(cam);

    // for each triangle
    std::vector<unsigned int> t_ids = frame1Mesh.getTrianglesIds();
    for (auto t_id : t_ids)
    {
        std::array<unsigned int, 3> v_ids = frame1Mesh.getTriangleIndices(t_id);

        Polygon kf_tri_3d = frame1Mesh.getCartesianTriangle(t_id);
        Triangle2D kf_tri_2d = frame1Mesh.getTexCoordTriangle(t_id);
        if (kf_tri_3d.vertices[0](2) <= 0.0 || kf_tri_3d.vertices[1](2) <= 0.0 || kf_tri_3d.vertices[2](2) <= 0.0)
            continue;
        if (kf_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
        if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
            continue;

        Polygon f_tri_3d = frame2Mesh.getCartesianTriangle(t_id);
        Triangle2D f_tri_2d = frame2Mesh.getTexCoordTriangle(t_id);
        if (f_tri_3d.vertices[0](2) <= 0.0 || f_tri_3d.vertices[1](2) <= 0.0 || f_tri_3d.vertices[2](2) <= 0.0)
            continue;
        if (f_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
        if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
            continue;

        f_tri_2d.computeTinv();
        std::array<int, 4> minmax = f_tri_2d.getMinMax();

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
            // with respect to depth
            if (jacMethod == MapJacobianMethod::depth)
                d_z_d_iz[i] = 1.0;
            // with respecto to idepth (depth = 1/idepth)
            if (jacMethod == MapJacobianMethod::idepth)
                d_z_d_iz[i] = -(kf_tri_3d.vertices[i](2) * kf_tri_3d.vertices[i](2));
            // width respect to depth = exp(z)
            if (jacMethod == MapJacobianMethod::log_depth)
                d_z_d_iz[i] = kf_tri_3d.vertices[i](2);
            if (jacMethod == MapJacobianMethod::log_idepth)
                d_z_d_iz[i] = -kf_tri_3d.vertices[i](2);
        }

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;

                f_tri_2d.computeBarycentric(f_pix);
                if (!f_tri_2d.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri_2d.interpolate(kf_tri_2d.vertices[0], kf_tri_2d.vertices[1], kf_tri_2d.vertices[2]);
                if (!cam.isPixVisible(kf_pix))
                    continue;

                Eigen::Vector3f kf_ver = f_tri_2d.interpolate(kf_tri_3d.vertices[0], kf_tri_3d.vertices[1], kf_tri_3d.vertices[2]);
                Eigen::Vector3f f_ver = f_tri_2d.interpolate(f_tri_3d.vertices[0], f_tri_3d.vertices[1], f_tri_3d.vertices[2]);

                if (kf_ver(2) <= 0.0 || f_ver(2) <= 0.0)
                    continue;

                // z-buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                float kf_i = float(frame1.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame2.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame2.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame2.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == frame1.image.nodata || f_i == frame2.image.nodata || dx == frame2.dx.nodata || dy == frame2.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                Eigen::Vector3f d_f_i_d_f_ver;
                d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) / f_ver(2) + d_f_i_d_f_ver(1) * f_ver(1) / f_ver(2));
                d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_ver_d_kf_depth = frame2.pose.rotationMatrix() * kf_ver / kf_ver(2);

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

                j_buffer.set(Eigen::Vector3f(J[0], J[1], J[2]), y, x, lvl);
                e_buffer.set(error, y, x, lvl);
                id_buffer.set(Eigen::Vector3i(v_ids[0], v_ids[1], v_ids[2]), y, x, lvl);
            }
        }
    }
}

void renderCPU::renderJMapNormalDepth(Mesh &mesh, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &j1_buffer, dataCPU<Eigen::Vector3f> &j2_buffer, dataCPU<Eigen::Vector3f> &j3_buffer, dataCPU<float> &e_buffer, dataCPU<Eigen::Vector3i> &id_buffer, MapJacobianMethod jacMethod, int lvl)
{
    // z_buffer.reset(lvl);

    float min_area = (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) / 16;
    float min_angle = M_PI / 64.0;

    Mesh frame1Mesh = mesh;
    Mesh frame2Mesh = mesh;

    frame1Mesh.transform(frame1.pose);
    frame1Mesh.computeTexCoords(cam);

    frame2Mesh.transform(frame2.pose);
    frame2Mesh.computeTexCoords(cam);

    // for each triangle
    std::vector<unsigned int> t_ids = frame1Mesh.getTrianglesIds();
    for (auto t_id : t_ids)
    {
        std::array<unsigned int, 3> v_ids = frame1Mesh.getTriangleIndices(t_id);

        Polygon kf_tri_3d = frame1Mesh.getCartesianTriangle(t_id);
        Triangle2D kf_tri_2d = frame1Mesh.getTexCoordTriangle(t_id);
        if (kf_tri_3d.vertices[0](2) <= 0.0 || kf_tri_3d.vertices[1](2) <= 0.0 || kf_tri_3d.vertices[2](2) <= 0.0)
            continue;
        if (kf_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
        if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
            continue;

        Polygon f_tri_3d = frame2Mesh.getCartesianTriangle(t_id);
        Triangle2D f_tri_2d = frame2Mesh.getTexCoordTriangle(t_id);
        if (f_tri_3d.vertices[0](2) <= 0.0 || f_tri_3d.vertices[1](2) <= 0.0 || f_tri_3d.vertices[2](2) <= 0.0)
            continue;
        if (f_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
        if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
            continue;

        f_tri_2d.computeTinv();
        std::array<int, 4> minmax = f_tri_2d.getMinMax();

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
            // with respect to depth
            if (jacMethod == MapJacobianMethod::depth)
                d_z_d_iz[i] = 1.0;
            // with respecto to idepth (depth = 1/idepth)
            if (jacMethod == MapJacobianMethod::idepth)
                d_z_d_iz[i] = -(kf_tri_3d.vertices[i](2) * kf_tri_3d.vertices[i](2));
            // width respect to depth = exp(z)
            if (jacMethod == MapJacobianMethod::log_depth)
                d_z_d_iz[i] = kf_tri_3d.vertices[i](2);
            if (jacMethod == MapJacobianMethod::log_idepth)
                d_z_d_iz[i] = -kf_tri_3d.vertices[i](2);
        }

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;

                f_tri_2d.computeBarycentric(f_pix);
                if (!f_tri_2d.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri_2d.interpolate(kf_tri_2d.vertices[0], kf_tri_2d.vertices[1], kf_tri_2d.vertices[2]);
                if (!cam.isPixVisible(kf_pix))
                    continue;

                Eigen::Vector3f kf_ver = f_tri_2d.interpolate(kf_tri_3d.vertices[0], kf_tri_3d.vertices[1], kf_tri_3d.vertices[2]);
                Eigen::Vector3f f_ver = f_tri_2d.interpolate(f_tri_3d.vertices[0], f_tri_3d.vertices[1], f_tri_3d.vertices[2]);

                if (kf_ver(2) <= 0.0 || f_ver(2) <= 0.0)
                    continue;

                // z-buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                float kf_i = float(frame1.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame2.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame2.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame2.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == frame1.image.nodata || f_i == frame2.image.nodata || dx == frame2.dx.nodata || dy == frame2.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                Eigen::Vector3f d_f_i_d_f_ver;
                d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) / f_ver(2) + d_f_i_d_f_ver(1) * f_ver(1) / f_ver(2));
                d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_ver_d_kf_depth = frame2.pose.rotationMatrix() * kf_ver / kf_ver(2);

                float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                float error = f_i - kf_i;

                Eigen::Vector3f J_depth;
                Eigen::Vector3f J_normal[3];
                for (int i = 0; i < 3; i++)
                {
                    float n_p_dot_ray = n_p[i].dot(kf_ver / kf_ver(2));
                    float d_kf_depth_d_z = d_n_d_z[i].dot(pr_p[i]) / n_p_dot_ray - n_p_dot_point[i] * d_n_d_z[i].dot(kf_ver / kf_ver(2)) / (n_p_dot_ray * n_p_dot_ray);
                    float d_f_i_d_z = d_f_i_d_kf_depth * d_kf_depth_d_z * d_z_d_iz[i];
                    J_depth(i) = d_f_i_d_z;
                    J_normal[i] = kf_tri_3d.vertices[i] / n_p_dot_ray - (kf_ver / kf_ver(2)) * n_p_dot_point[i] / (n_p_dot_ray * n_p_dot_ray);
                }

                j_buffer.set(J_depth, y, x, lvl);
                e_buffer.set(error, y, x, lvl);
                id_buffer.set(Eigen::Vector3i(v_ids[0], v_ids[1], v_ids[2]), y, x, lvl);
            }
        }
    }
}

void renderCPU::renderJPose(Mesh &mesh, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &jtra_buffer, dataCPU<Eigen::Vector3f> &jrot_buffer, dataCPU<float> &e_buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    float min_area = (float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) / 16;
    float min_angle = M_PI / 64.0;

    Mesh frame1Mesh = mesh;
    Mesh frame2Mesh = mesh;

    frame1Mesh.transform(frame1.pose);
    frame1Mesh.computeTexCoords(cam);

    frame2Mesh.transform(frame2.pose);
    frame2Mesh.computeTexCoords(cam);

    // for each triangle
    std::vector<unsigned int> t_ids = frame1Mesh.getTrianglesIds();
    for (auto t_id : t_ids)
    {
        Triangle2D kf_tri = frame1Mesh.getTexCoordTriangle(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (kf_tri.getArea() < min_area)
            continue;
        std::array<float, 3> kf_angle = kf_tri.getAngles();
        if (fabs(kf_angle[0]) < min_angle || fabs(kf_angle[1]) < min_angle || fabs(kf_angle[2]) < min_angle)
            continue;
        Polygon f_tri_3d = frame2Mesh.getCartesianTriangle(t_id);
        Triangle2D f_tri_2d = frame2Mesh.getTexCoordTriangle(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> f_angle = f_tri_2d.getAngles();
        if (fabs(f_angle[0]) < min_angle || fabs(f_angle[1]) < min_angle || fabs(f_angle[2]) < min_angle)
            continue;

        f_tri_2d.computeTinv();
        std::array<int, 4> minmax = f_tri_2d.getMinMax();

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;

                f_tri_2d.computeBarycentric(f_pix);

                if (!f_tri_2d.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri_2d.interpolate(kf_tri.vertices[0], kf_tri.vertices[1], kf_tri.vertices[2]);
                if (!cam.isPixVisible(kf_pix))
                    continue;

                Eigen::Vector3f f_ver = f_tri_2d.interpolate(f_tri_3d.vertices[0], f_tri_3d.vertices[1], f_tri_3d.vertices[2]);

                if (f_ver(2) <= 0.0)
                    continue;

                // z-buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                float kf_i = float(frame1.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame2.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame2.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame2.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == frame1.image.nodata || f_i == frame2.image.nodata || dx == frame2.dx.nodata || dy == frame2.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                float v0 = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                float v1 = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_i_d_tra = Eigen::Vector3f(v0, v1, v2);
                Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);

                float residual = (f_i - kf_i);

                jtra_buffer.set(d_f_i_d_tra, y, x, lvl);
                jrot_buffer.set(d_f_i_d_rot, y, x, lvl);
                e_buffer.set(residual, y, x, lvl);
            }
        }
    }
}

void renderCPU::renderJPose(dataCPU<float> &frame1Idepth, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &jtra_buffer, dataCPU<Eigen::Vector3f> &jrot_buffer, dataCPU<float> &e_buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    Sophus::SE3f relPos = frame2.pose * frame1.pose.inverse();

    for (int y = 0; y < cam.height; y++)
    {
        for (int x = 0; x < cam.width; x++)
        {
            Eigen::Vector2f f1_pix(x, y);
            Eigen::Vector3f f1_ray = cam.pixToRay(f1_pix);
            float f1_idepth = frame1Idepth.get(y, x, lvl);
            if (f1_idepth <= 0.0 || f1_idepth == frame1Idepth.nodata)
                continue;
            Eigen::Vector3f f1_ver = f1_ray / f1_idepth;

            Eigen::Vector3f f2_ver = relPos * f1_ver;
            if (f2_ver(2) <= 0.0)
                continue;
            Eigen::Vector3f f2_ray = f2_ver / f2_ver(2);
            Eigen::Vector2f f2_pix = cam.rayToPix(f2_ray);

            if (!cam.isPixVisible(f2_pix))
                continue;

            // z-buffer
            // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
            // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
            //    continue;

            float f1_i = float(frame1.image.get(f1_pix(1), f1_pix(0), lvl));
            float f2_i = float(frame2.image.get(f2_pix(1), f2_pix(0), lvl));
            float dx = frame2.dx.get(f2_pix(1), f2_pix(0), lvl);
            float dy = frame2.dy.get(f2_pix(1), f2_pix(0), lvl);

            if (f1_i == frame1.image.nodata || f2_i == frame2.image.nodata || dx == frame2.dx.nodata || dy == frame2.dy.nodata)
                continue;

            Eigen::Vector2f d_f_i_d_pix(dx, dy);

            float v0 = d_f_i_d_pix(0) * cam.fx / f2_ver(2);
            float v1 = d_f_i_d_pix(1) * cam.fy / f2_ver(2);
            float v2 = -(v0 * f2_ver(0) + v1 * f2_ver(1)) / f2_ver(2);

            Eigen::Vector3f d_f_i_d_tra = Eigen::Vector3f(v0, v1, v2);
            Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f2_ver(2) * v1 + f2_ver(1) * v2, f2_ver(2) * v0 - f2_ver(0) * v2, -f2_ver(1) * v0 + f2_ver(0) * v1);

            float residual = (f2_i - f1_i);

            jtra_buffer.set(d_f_i_d_tra, y, x, lvl);
            jrot_buffer.set(d_f_i_d_rot, y, x, lvl);
            e_buffer.set(residual, y, x, lvl);
        }
    }
}

void renderCPU::renderJPoseMap(Mesh &mesh, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &j1_buffer, dataCPU<Eigen::Vector3f> &j2_buffer, dataCPU<Eigen::Vector3f> &j3_buffer, dataCPU<float> &e_buffer, dataCPU<Eigen::Vector3i> &id_buffer, MapJacobianMethod jacMethod, int lvl)
{
    // z_buffer.reset(lvl);

    float min_area = (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) / 16.0;
    float min_angle = M_PI / 64.0;

    Mesh frame1Mesh = mesh;
    Mesh frame2Mesh = mesh;

    frame1Mesh.transform(frame1.pose);
    frame1Mesh.computeTexCoords(cam);

    frame2Mesh.transform(frame2.pose);
    frame2Mesh.computeTexCoords(cam);

    // for each triangle
    std::vector<unsigned int> t_ids = frame1Mesh.getTrianglesIds();
    for (auto t_id : t_ids)
    {
        std::array<unsigned int, 3> v_ids = frame1Mesh.getTriangleIndices(t_id);

        Triangle2D kf_tri_2d = frame1Mesh.getTexCoordTriangle(t_id);
        Polygon kf_tri_3d = frame1Mesh.getCartesianTriangle(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (kf_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
        if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
            continue;

        Polygon f_tri_3d = frame2Mesh.getCartesianTriangle(t_id);
        Triangle2D f_tri_2d = frame2Mesh.getTexCoordTriangle(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_tri_2d.getArea() < min_area)
            continue;
        std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
        if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
            continue;

        f_tri_2d.computeTinv();
        std::array<int, 4> minmax = f_tri_2d.getMinMax();

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
            // with respect to depth
            if (jacMethod == MapJacobianMethod::depth)
                d_z_d_iz[i] = 1.0;
            // with respecto to idepth (depth = 1/idepth)
            if (jacMethod == MapJacobianMethod::idepth)
                d_z_d_iz[i] = -(kf_tri_3d.vertices[i](2) * kf_tri_3d.vertices[i](2));
            // width respect to depth = exp(z)
            if (jacMethod == MapJacobianMethod::log_depth)
                d_z_d_iz[i] = kf_tri_3d.vertices[i](2);
            if (jacMethod == MapJacobianMethod::log_idepth)
                d_z_d_iz[i] = -(kf_tri_3d.vertices[i](2) * kf_tri_3d.vertices[i](2));
        }

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;

                f_tri_2d.computeBarycentric(f_pix);
                if (!f_tri_2d.isBarycentricOk())
                    continue;

                Eigen::Vector2f kf_pix = f_tri_2d.interpolate(kf_tri_2d.vertices[0], kf_tri_2d.vertices[1], kf_tri_2d.vertices[2]);
                if (!cam.isPixVisible(kf_pix))
                    continue;

                Eigen::Vector3f kf_ver = f_tri_2d.interpolate(kf_tri_3d.vertices[0], kf_tri_3d.vertices[1], kf_tri_3d.vertices[2]);
                Eigen::Vector3f f_ver = f_tri_2d.interpolate(f_tri_3d.vertices[0], f_tri_3d.vertices[1], f_tri_3d.vertices[2]);

                if (kf_ver(2) <= 0.0 || f_ver(2) <= 0.0)
                    continue;

                // z-buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                float kf_i = float(frame1.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame2.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame2.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame2.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == frame1.image.nodata || f_i == frame2.image.nodata || dx == frame2.dx.nodata || dy == frame2.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                Eigen::Vector3f d_f_i_d_f_ver;
                d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) / f_ver(2) + d_f_i_d_f_ver(1) * f_ver(1) / f_ver(2));
                d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_i_d_tra = d_f_i_d_f_ver;
                Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                j1_buffer.set(d_f_i_d_tra, y, x, lvl);
                j2_buffer.set(d_f_i_d_rot, y, x, lvl);

                Eigen::Vector3f d_f_ver_d_kf_depth = frame2.pose.rotationMatrix() * kf_ver / kf_ver(2);

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

                j3_buffer.set(Eigen::Vector3f(J[0], J[1], J[2]), y, x, lvl);
            }
        }
    }
}
