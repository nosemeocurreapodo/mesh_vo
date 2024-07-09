#include "cpu/renderCPU.h"
#include <math.h>
#include "utils/tictoc.h"

void renderCPU::renderIdepth(MeshCPU &mesh, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    MeshCPU frameMesh = mesh.getCopy();
    frameMesh.transform(pose);
    frameMesh.computeTexCoords(cam);

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

                // float l_idepth = idepth.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != idepth.nodata)
                //     continue;

                buffer.set(1.0 / depth, f_pix(1), f_pix(0), lvl);
            }
        }
    }
}

void renderCPU::renderImage(MeshCPU &mesh, camera &cam, dataCPU<float> &image, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    mesh.computeTexCoords(cam);
    MeshCPU frameMesh = mesh.getCopy();
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

                float kf_i = float(image.get(kf_pix(1), kf_pix(0), lvl));
                if (kf_i == image.nodata)
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

void renderCPU::renderError(MeshCPU &mesh, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<float> &buffer, int lvl)
{
    MeshCPU frame1Mesh = mesh.getCopy();
    frame1Mesh.transform(frame1.pose);
    frame1Mesh.computeTexCoords(cam);

    MeshCPU frame2Mesh = mesh.getCopy();
    frame2Mesh.transform(frame2.pose);
    frame2Mesh.computeTexCoords(cam);

    std::vector<unsigned int> trisIds = frame1Mesh.getTrianglesIds();

    // for each triangle
    for (int t_index = 0; t_index < (int)trisIds.size(); t_index++)
    {
        unsigned int t_id = trisIds[t_index];

        Triangle2D kf_tri = frame1Mesh.getTexCoordTriangle(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (kf_tri.getArea() < 1.0)
            continue;
        Triangle2D f_tri = frame2Mesh.getTexCoordTriangle(t_id);
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

                float kf_i = float(frame1.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame2.image.get(f_pix(1), f_pix(0), lvl));
                if (kf_i == frame1.image.nodata || f_i == frame2.image.nodata)
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

void renderCPU::renderDebug(MeshCPU &mesh, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    MeshCPU frameMesh = mesh.getCopy();
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
/*
void renderCPU::renderInvVar(MeshCPU &refMesh, frameCPU &frame, dataCPU<float> &buffer, int lvl)
{
    refMesh.computeTexCoords(cam[lvl]);

    MeshCPU frameMesh = refMesh.getCopy();
    frameMesh.transform(frame.pose);
    frameMesh.computeTexCoords(cam[lvl]);

    std::vector<unsigned int> trisIds = frameMesh.getTrianglesIds();

    // for each triangle
    for (std::size_t index = 0; index < trisIds.size(); index++)
    {
        unsigned int t_id = trisIds[index];
        std::array<unsigned int, 3> v_ids = frameMesh.getTriangleIndices(t_id);
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

        std::array<float, 3> idepthVars;
        idepthVars[0] = invVar[v_ids[0]][v_ids[0]];
        idepthVars[1] = invVar[v_ids[1]][v_ids[1]];
        idepthVars[2] = invVar[v_ids[2]][v_ids[2]];

        f_tri.computeTinv();
        std::array<int, 4> minmax = f_tri.getMinMax();

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam[lvl].isPixVisible(f_pix))
                    continue;

                f_tri.computeBarycentric(f_pix);
                if (!f_tri.isBarycentricOk())
                    continue;

                float var = f_tri.interpolate(idepthVars[0], idepthVars[1], idepthVars[2]);

                // z buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                buffer.set(var, y, x, lvl);
            }
        }
    }
}
*/