#pragma once

#include "common/types.h"

bool isTriangleEqual(vec3i tri_indices_1, vec3i tri_indices_2)
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

bool isEdgeEqual(vec2i edge_indices_1, vec2i edge_indices_2)
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




