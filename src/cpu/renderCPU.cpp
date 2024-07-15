#include "cpu/renderCPU.h"
#include <math.h>
#include "utils/tictoc.h"

void renderCPU::renderIdepth(SceneBase &scene, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    std::unique_ptr<SceneBase> frameMesh = scene.clone();
    frameMesh->transform(pose);

    std::vector<unsigned int> shapesIds = frameMesh->getShapesIds();

    // for each triangle
    for (auto t_id : shapesIds)
    {
        // Triangle kf_tri = keyframeMesh.triangles[t_id];
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        // if (kf_tri.isBackFace())
        //     continue;
        auto f_pol = frameMesh->getShape(t_id);
        // if (f_tri2d.vertices[0](2) <= 0.0 || f_tri2d.vertices[1](2) <= 0.0 || f_tri2d.vertices[2](2) <= 0.0)
        //      continue;
        if (f_pol->getArea() < 0.0)
            continue;

        std::array<int, 4> minmax = f_pol->getScreenBounds(cam);

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;
                Eigen::Vector3f f_ray = cam.pixToRay(f_pix);

                f_pol->prepareForRay(f_ray);
                if (!f_pol->rayHitsShape())
                    continue;

                float f_depth = f_pol->getRayDepth();

                if (f_depth <= 0.0)
                    continue;

                float z_depth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                if (z_depth < f_depth && z_depth != z_buffer.nodata)
                    continue;

                buffer.set(1.0 / f_depth, f_pix(1), f_pix(0), lvl);

                z_buffer.set(f_depth, f_pix(1), f_pix(0), lvl);
            }
        }
    }
}

void renderCPU::renderImage(SceneBase &scene, camera &cam, dataCPU<float> &image, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    std::unique_ptr<SceneBase> frameMesh = scene.clone();
    frameMesh->transform(pose);
    Sophus::SE3f relPose = pose * scene.getPose().inverse();
    Sophus::SE3f relPoseInv = relPose.inverse();

    std::vector<unsigned int> ids = frameMesh->getShapesIds();

    // for each triangle
    for (auto t_id : ids)
    {
        // Polygon kf_pol = mesh.getPolygon(t_id);
        //  if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //      continue;
        // if (kf_tri.getArea() < 1.0)
        //     continue;

        auto f_pol = frameMesh->getShape(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_pol->getArea() < 0.0)
            continue;

        std::array<int, 4> minmax = f_pol->getScreenBounds(cam);

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;
                Eigen::Vector3f f_ray = cam.pixToRay(f_pix);

                f_pol->prepareForRay(f_ray);
                if (!f_pol->rayHitsShape())
                    continue;

                float f_depth = f_pol->getRayDepth();
                if (f_depth <= 0.0)
                    continue;

                Eigen::Vector3f f_ver = f_ray * f_depth;

                Eigen::Vector3f kf_ver = relPoseInv * f_ver;
                if (kf_ver(2) <= 0.0)
                    continue;
                Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);
                Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);

                if (!cam.isPixVisible(kf_pix))
                    continue;

                float kf_i = float(image.get(kf_pix(1), kf_pix(0), lvl));
                if (kf_i == image.nodata)
                    continue;

                float z_depth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                if (z_depth < f_depth && z_depth != z_buffer.nodata)
                    continue;

                buffer.set(kf_i, y, x, lvl);

                z_buffer.set(f_depth, f_pix(1), f_pix(0), lvl);
            }
        }
    }
}

void renderCPU::renderDebug(SceneBase &scene, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    std::unique_ptr<SceneBase> frameMesh = scene.clone();
    frameMesh->transform(pose);

    std::vector<unsigned int> ids = frameMesh->getShapesIds();

    // for each triangle
    for (auto t_id : ids)
    {
        // Triangle kf_tri = keyframeMesh.triangles[index];
        //  if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //      continue;
        // if (kf_tri.isBackFace())
        //     continue;
        auto f_pol = frameMesh->getShape(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_pol->getArea() < 0.0)
            continue;

        std::array<int, 4> minmax = f_pol->getScreenBounds(cam);

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;

                Eigen::Vector3f f_ray = cam.pixToRay(f_pix);

                f_pol->prepareForRay(f_ray);
                if (!f_pol->rayHitsShape())
                    continue;

                bool isLine = false; // f_pol.isLine();

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

void renderCPU::renderJMap(SceneBase &scene, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &j_buffer, dataCPU<float> &e_buffer, dataCPU<Eigen::Vector3i> &pId_buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    float min_area = 0.0 * (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) / 16;
    float min_angle = M_PI / 64.0;

    std::unique_ptr<SceneBase> frame1Mesh = scene.clone();
    std::unique_ptr<SceneBase> frame2Mesh = scene.clone();

    frame1Mesh->transform(frame1.pose);
    frame2Mesh->transform(frame2.pose);

    Sophus::SE3f relPose = frame2.pose * frame1.pose.inverse();
    Sophus::SE3f relPoseInv = relPose.inverse();

    // for each triangle
    std::vector<unsigned int> t_ids = frame1Mesh->getShapesIds();
    for (auto t_id : t_ids)
    {
        std::vector<unsigned int> p_ids = frame1Mesh->getShapeParamsIds(t_id);

        auto kf_pol = frame1Mesh->getShape(t_id);

        // if (kf_tri_3d.vertices[0](2) <= 0.0 || kf_tri_3d.vertices[1](2) <= 0.0 || kf_tri_3d.vertices[2](2) <= 0.0)
        //     continue;
        if (kf_pol->getArea() < min_area)
            continue;
        // std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
        // if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
        //     continue;

        auto f_pol = frame2Mesh->getShape(t_id);

        // if (f_tri_3d.vertices[0](2) <= 0.0 || f_tri_3d.vertices[1](2) <= 0.0 || f_tri_3d.vertices[2](2) <= 0.0)
        //     continue;
        if (f_pol->getArea() < min_area)
            continue;
        // std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
        // if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
        //     continue;

        std::array<int, 4> minmax = f_pol->getScreenBounds(cam);

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;
                Eigen::Vector3f f_ray = cam.pixToRay(f_pix);

                f_pol->prepareForRay(f_ray);
                if (!f_pol->rayHitsShape())
                    continue;

                float f_depth = f_pol->getRayDepth();
                if (f_depth <= 0.0)
                    continue;

                // z-buffer
                float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                    continue;

                Eigen::Vector3f f_ver = f_ray * f_depth;
                Eigen::Vector3f kf_ver = relPoseInv * f_ver;
                if (kf_ver(2) <= 0.0)
                    continue;

                Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);

                kf_pol->prepareForRay(kf_ray);
                if (!kf_pol->rayHitsShape())
                    continue;

                Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);
                if (!cam.isPixVisible(kf_pix))
                    continue;

                float kf_i = float(frame1.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame2.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame2.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame2.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == frame1.image.nodata || f_i == frame2.image.nodata || dx == frame2.dx.nodata || dy == frame2.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                float error = f_i - kf_i;

                Eigen::Vector3f d_f_i_d_f_ver = cam.d_f_i_d_f_ver(d_f_i_d_pix, f_ver);

                // Eigen::Vector3f d_f_i_d_f_ver;
                // d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                // d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_ver_d_kf_depth = relPose.rotationMatrix() * kf_ray;
                float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                // this could be the jacobian of the depth of the 3 vertices in a triangle
                // or the jacobian of the normal + depth of a surfel
                std::vector<float> Jacobian = kf_pol->getJacobian(d_f_i_d_kf_depth);

                Eigen::Vector3f jacs;
                Eigen::Vector3i ids;
                for (int i = 0; i < p_ids.size(); i++)
                {
                    if (i >= 3)
                        break;
                    jacs[i] = Jacobian[i];
                    ids[i] = p_ids[i];
                }

                e_buffer.set(error, y, x, lvl);
                j_buffer.set(jacs, y, x, lvl);
                pId_buffer.set(ids, y, x, lvl);
            }
        }
    }
}

void renderCPU::renderJPose(SceneBase &scene, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &jtra_buffer, dataCPU<Eigen::Vector3f> &jrot_buffer, dataCPU<float> &e_buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    float min_area = 0.0 * (float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) / 16;
    float min_angle = M_PI / 64.0;

    std::unique_ptr<SceneBase> frame1Mesh = scene.clone();
    std::unique_ptr<SceneBase> frame2Mesh = scene.clone();

    frame1Mesh->transform(frame1.pose);
    frame2Mesh->transform(frame2.pose);

    Sophus::SE3f relPose = frame2.pose * frame1.pose.inverse();
    Sophus::SE3f relPoseInv = relPose.inverse();

    // for each triangle
    std::vector<unsigned int> t_ids = frame1Mesh->getShapesIds();
    for (auto t_id : t_ids)
    {
        auto kf_pol = frame1Mesh->getShape(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (kf_pol->getArea() < min_area)
            continue;
        // std::array<float, 3> kf_angle = kf_tri.getAngles();
        // if (fabs(kf_angle[0]) < min_angle || fabs(kf_angle[1]) < min_angle || fabs(kf_angle[2]) < min_angle)
        //    continue;
        auto f_pol = frame2Mesh->getShape(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_pol->getArea() < min_area)
            continue;
        // std::array<float, 3> f_angle = f_tri_2d.getAngles();
        // if (fabs(f_angle[0]) < min_angle || fabs(f_angle[1]) < min_angle || fabs(f_angle[2]) < min_angle)
        //    continue;

        std::array<int, 4> minmax = f_pol->getScreenBounds(cam);

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;
                Eigen::Vector3f f_ray = cam.pixToRay(f_pix);

                f_pol->prepareForRay(f_ray);
                if (!f_pol->rayHitsShape())
                    continue;

                float f_depth = f_pol->getRayDepth();
                if (f_depth <= 0.0)
                    continue;

                Eigen::Vector3f f_ver = f_ray * f_depth;
                Eigen::Vector3f kf_ver = relPoseInv * f_ver;
                if (kf_ver(2) <= 0.0)
                    continue;

                Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);

                kf_pol->prepareForRay(kf_ray);
                if (!kf_pol->rayHitsShape())
                    continue;

                Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);

                if (!cam.isPixVisible(kf_pix))
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

                // Eigen::MatrixXf d_pix_d_f_ver = cam.dPixdPoint(f_ver);

                // Eigen::Vector3f d_f_i_d_f_ver = d_f_i_d_pix * d_pix_d_f_ver;
                // Eigen::Vector3f d_f_i_d_tra = d_f_i_d_f_ver;
                // Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

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
            // float dx = frame2.dx.get(f2_pix(1), f2_pix(0), lvl);
            // float dy = frame2.dy.get(f2_pix(1), f2_pix(0), lvl);
            // Eigen::Vector2f d_f_i_d_pix(dx, dy);

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

void renderCPU::renderJPoseMap(SceneBase &mesh, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &j1_buffer, dataCPU<Eigen::Vector3f> &j2_buffer, dataCPU<Eigen::Vector3f> &j3_buffer, dataCPU<float> &e_buffer, dataCPU<Eigen::Vector3i> &pId_buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    float min_area = 0.0 * (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) / 16.0;
    float min_angle = M_PI / 64.0;

    std::unique_ptr<SceneBase> frame1Scene = mesh.clone();
    std::unique_ptr<SceneBase> frame2Scene = mesh.clone();

    frame1Scene->transform(frame1.pose);
    frame2Scene->transform(frame2.pose);

    Sophus::SE3f relPose = frame2.pose * frame1.pose.inverse();
    Sophus::SE3f relPoseInv = relPose.inverse();

    // for each triangle
    std::vector<unsigned int> t_ids = frame1Scene->getShapesIds();
    for (auto t_id : t_ids)
    {
        std::vector<unsigned int> p_ids = frame1Scene->getShapeParamsIds(t_id);

        auto kf_pol = frame1Scene->getShape(t_id);
        // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (kf_pol->getArea() < min_area)
            continue;
        // std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
        // if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
        //    continue;

        auto f_pol = frame2Scene->getShape(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_pol->getArea() < min_area)
            continue;
        // std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
        // if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
        //    continue;

        std::array<int, 4> minmax = f_pol->getScreenBounds(cam);

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;
                Eigen::Vector3f f_ray = cam.pixToRay(f_pix);

                f_pol->prepareForRay(f_ray);
                if (!f_pol->rayHitsShape())
                    continue;

                float f_depth = f_pol->getRayDepth();
                if (f_depth <= 0.0)
                    continue;

                // z-buffer
                float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                    continue;

                Eigen::Vector3f f_ver = f_ray * f_depth;
                Eigen::Vector3f kf_ver = relPoseInv * f_ver;
                if (kf_ver(2) <= 0.0)
                    continue;
                Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);

                kf_pol->prepareForRay(kf_ray);
                if (!kf_pol->rayHitsShape())
                    continue;

                Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);

                if (!cam.isPixVisible(kf_pix))
                    continue;

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
                d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_i_d_tra = d_f_i_d_f_ver;
                Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                j1_buffer.set(d_f_i_d_tra, y, x, lvl);
                j2_buffer.set(d_f_i_d_rot, y, x, lvl);

                Eigen::Vector3f d_f_ver_d_kf_depth = relPose.rotationMatrix() * kf_ray;
                float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                std::vector<float> Jacobian = kf_pol->getJacobian(d_f_i_d_kf_depth);

                float error = f_i - kf_i;

                Eigen::Vector3f jacs;
                Eigen::Vector3i ids;
                for (int i = 0; i < p_ids.size(); i++)
                {
                    if (i >= 3)
                        break;
                    jacs[i] = Jacobian[i];
                    ids[i] = p_ids[i];
                }

                e_buffer.set(error, y, x, lvl);
                j3_buffer.set(jacs, y, x, lvl);
                pId_buffer.set(ids, y, x, lvl);
            }
        }
    }
}
