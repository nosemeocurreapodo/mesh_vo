#pragma once

#include "common/types.h"

static bool isTriangleEqual(vec3i tri_indices_1, vec3i tri_indices_2)
{
    bool isIndicePresent[3];
    for (int tri_indice = 0; tri_indice < 3; tri_indice++)
    {
        isIndicePresent[tri_indice] = false;
        if (tri_indices_1(tri_indice) == tri_indices_2(0) || tri_indices_1(tri_indice) == tri_indices_2(1) || tri_indices_1(tri_indice) == tri_indices_2(2))
            isIndicePresent[tri_indice] = true;
    }
    if (isIndicePresent[0] && isIndicePresent[1] && isIndicePresent[2])
        return true;
    return false;
}

static bool isEdgeEqual(vec2i edge_indices_1, vec2i edge_indices_2)
{
    bool isIndicePresent[2];
    for (int edge_indice = 0; edge_indice < 2; edge_indice++)
    {
        isIndicePresent[edge_indice] = false;
        if (edge_indices_1(edge_indice) == edge_indices_2(0) || edge_indices_1(edge_indice) == edge_indices_2(1))
            isIndicePresent[edge_indice] = true;
    }
    if (isIndicePresent[0] && isIndicePresent[1])
        return true;
    return false;
}

float meanViewAngle(keyFrameCPU &kframe, SE3f pose1, SE3f pose2)
{
    int lvl = 1;

    geometryType scene1 = kframe.getGeometry();
    scene1.transform(pose1);
    //scene1.project(cam);

    geometryType scene2 = kframe.getGeometry();
    scene2.transform(pose2);
    //scene2.project(cam);

    SE3f relativePose = pose1 * pose2.inverse();

    SE3f frame1PoseInv = relativePose.inverse();
    SE3f frame2PoseInv = SE3f();

    vec3f frame1Translation = frame1PoseInv.translation();
    vec3f frame2Translation = frame2PoseInv.translation();

    std::vector<int> vIds = scene2.getVerticesIds();

    float accAngle = 0;
    int count = 0;
    for (int vId : vIds)
    {
        vertex vert = scene2.getVertex(vId);

        vec3f diff1 = vert.ver - frame1Translation;
        vec3f diff2 = vert.ver - frame2Translation;

        assert(diff1.norm() > 0 && diff2.norm() > 0);

        vec3f diff1Normalized = diff1 / diff1.norm();
        vec3f diff2Normalized = diff2 / diff2.norm();

        float cos_angle = diff1Normalized.dot(diff2Normalized);
        cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);
        float angle = std::acos(cos_angle);

        assert(!std::isnan(angle));

        accAngle += angle;
        count += 1;
    }

    return accAngle / count;
}


