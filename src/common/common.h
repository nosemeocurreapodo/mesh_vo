#pragma once

#include "backends/cpu/texturecpu.h"
#include "backends/cpu/meshcpu.h"

inline MeshCPU CreateMesh(const TextureCPU<float> &depth, Camera &cam, int grid_size)
{
    std::vector<Vec2> grid_uv = UniformTexCoords(grid_size, grid_size);

    std::vector<float> vertices, texcoords, weights;

    vertices.clear();
    texcoords.clear();
    weights.clear();

    vertices.reserve(grid_uv.size() * 3);
    texcoords.reserve(grid_uv.size() * 2);
    weights.reserve(grid_uv.size());

    int w = depth.width(0);
    int h = depth.height(0);
    auto depth_mm = depth.MapRead(0);

    for (const Vec2 &uv : grid_uv)
    {
        const float ix = uv(0) * (w - 1);
        const float iy = uv(1) * (h - 1);
        const int x = static_cast<int>(ix);
        const int y = static_cast<int>(iy);
        const float depth = depth_mm[y * w + x];

        if (depth <= 0.0f)
            continue;

        const Vec3 ray = cam.PixToRay(uv);
        const Vec3 vertex = ray * depth;

        vertices.push_back(vertex(0));
        vertices.push_back(vertex(1));
        vertices.push_back(vertex(2));
        texcoords.push_back(uv(0));
        texcoords.push_back(uv(1));
        weights.push_back(1.0f);
    }

    return MeshCPU(vertices, texcoords, weights);
}
