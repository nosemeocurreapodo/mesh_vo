#pragma once

class mesh
{
public:
    mesh()
    {

    };

    // scene
    //the vertices, actual data of the scene
    std::vector<std::array<float, 3> > vertices;
    //the indices gives me, for each triangle, the vertices indices
    std::vector<std::array<unsigned int, 3> > triangle_index_to_vertices_indeces;
    //for each vertices, gives me the triangles indices
    std::vector<std::vector<unsigned int> > vertex_index_to_triangles_indeces;
};
