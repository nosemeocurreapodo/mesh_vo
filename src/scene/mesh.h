#pragma once

class mesh
{
public:
    mesh(){

    };

    void init()
    {
        // preallocate scene vertices to zero
        for (int y = 0; y < VERTEX_HEIGHT; y++)
        {
            for (int x = 0; x < VERTEX_WIDTH; x++)
            {
                std::array<float, 3> vertex;
                vertex[0] = 0.0;
                vertex[1] = 0.0;
                vertex[2] = 0.0;
                sceneMesh.vertices.push_back(vertex);
                // vertices.push_back(0.0);
                // vertices.push_back(0.0);
                // vertices.push_back(0.0);

                // just emtpy list of triangles now
                std::vector<unsigned int> triangle_indices;
                sceneMesh.vertex_index_to_triangles_indeces.push_back(triangle_indices);
            }
        }

        // init scene indices
        for (int y = 0; y < VERTEX_HEIGHT; y++)
        {
            for (int x = 0; x < VERTEX_WIDTH; x++)
            {
                if (x > 0 && y > 0)
                {
                    // if (((x % 2 == 0)))
                    //  if(((x % 2 == 0) && (y % 2 == 0)) || ((x % 2 != 0) && (y % 2 != 0)))
                    if (rand() > 0.5 * RAND_MAX)
                    {
                        std::array<unsigned int, 3> vertices_indices1;
                        vertices_indices1[0] = x - 1 + y * (VERTEX_WIDTH);
                        vertices_indices1[1] = x + (y - 1) * (VERTEX_WIDTH);
                        vertices_indices1[2] = x - 1 + (y - 1) * (VERTEX_WIDTH);

                        sceneMesh.triangle_index_to_vertices_indeces.push_back(vertices_indices1);

                        std::vector<unsigned int> triangles_indices11 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices1[0]];
                        std::vector<unsigned int> triangles_indices12 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices1[1]];
                        std::vector<unsigned int> triangles_indices13 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices1[2]];

                        triangles_indices11.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());
                        triangles_indices12.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());
                        triangles_indices13.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());

                        std::array<unsigned int, 3> vertices_indices2;
                        vertices_indices2[0] = x + y * (VERTEX_WIDTH);
                        vertices_indices2[1] = x + (y - 1) * (VERTEX_WIDTH);
                        vertices_indices2[2] = x - 1 + y * (VERTEX_WIDTH);

                        sceneMesh.triangle_index_to_vertices_indeces.push_back(vertices_indices2);

                        std::vector<unsigned int> triangles_indices21 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices2[0]];
                        std::vector<unsigned int> triangles_indices22 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices2[1]];
                        std::vector<unsigned int> triangles_indices23 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices2[2]];

                        triangles_indices21.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());
                        triangles_indices22.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());
                        triangles_indices23.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());
                    }
                    else
                    {
                        std::array<unsigned int, 3> vertices_indices1;
                        vertices_indices1[0] = x + y * (VERTEX_WIDTH);
                        vertices_indices1[1] = x - 1 + (y - 1) * (VERTEX_WIDTH);
                        vertices_indices1[2] = x - 1 + y * (VERTEX_WIDTH);

                        sceneMesh.triangle_index_to_vertices_indeces.push_back(vertices_indices1);

                        std::vector<unsigned int> triangles_indices11 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices1[0]];
                        std::vector<unsigned int> triangles_indices12 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices1[1]];
                        std::vector<unsigned int> triangles_indices13 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices1[2]];

                        triangles_indices11.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());
                        triangles_indices12.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());
                        triangles_indices13.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());

                        std::array<unsigned int, 3> vertices_indices2;
                        vertices_indices2[0] = x + y * (VERTEX_WIDTH);
                        vertices_indices2[1] = x + (y - 1) * (VERTEX_WIDTH);
                        vertices_indices2[2] = x - 1 + (y - 1) * (VERTEX_WIDTH);

                        sceneMesh.triangle_index_to_vertices_indeces.push_back(vertices_indices2);

                        std::vector<unsigned int> triangles_indices21 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices2[0]];
                        std::vector<unsigned int> triangles_indices22 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices2[1]];
                        std::vector<unsigned int> triangles_indices23 = sceneMesh.vertex_index_to_triangles_indeces[vertices_indices2[2]];

                        triangles_indices21.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());
                        triangles_indices22.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());
                        triangles_indices23.push_back(sceneMesh.triangle_index_to_vertices_indeces.size());
                    }
                }
            }
        }
    }

    std::array<float, 3> getVertex(unsigned int local_vertex_index)
    {
        return vertices[local_vertex_index];
    }

    int getVertexIndex(std::array<float, 3> vertex)
    {
        int vertex_index = -1.0;
        for (size_t i; i > vertices.size(); i++)
        {
            std::array<float, 3> vert = vertices[i];
            if (vert[0] == vertex[0] && vert[1] == vertex[1] && vert[2] == vertex[2])
            {
                vertex_index = i;
                break;
            }
        }
        return vertex_index;
    }

    unsigned int getVertexSize()
    {
        return (unsigned int)vertices.size();
    }

    void addVertex(std::array<float, 3> vertex, unsigned int global_index)
    {
        vertices.push_back(vertex);
        // new vertex is not associated with any triangle yet
        std::vector<unsigned int> triangle_indices;
        vertex_index_to_triangles_indeces.push_back(triangle_indices);
        vertices_global_indices.push_back(global_index);
    }

    std::array<unsigned int, 3> getVertexIndexes(unsigned int local_triangle_index)
    {
        return triangle_index_to_vertices_indeces[local_triangle_index];
    }

    std::vector<unsigned int> getTrianglesIndexes(unsigned int local_vertex_index)
    {
        return vertex_index_to_triangles_indeces[local_vertex_index];
    }

    unsigned int getTriangleSize()
    {
        return (unsigned int)triangle_index_to_vertices_indeces.size();
    }

    void addTriangle(std::array<unsigned int, 3> vertex_indices)
    {
        // hope the triangle is not already in the list
        triangle_index_to_vertices_indeces.push_back(vertex_indices);
        unsigned int triangle_index = triangle_index_to_vertices_indeces.size();
        // for each vertex, add the triangle to it
        for (int i = 0; i < 3; i++)
        {
            // hope the vertex is already present
            vertex_index_to_triangles_indeces[vertex_indices[i]].push_back(triangle_index);
        }
    }

    unsigned int getGlobalVertexIndex(unsigned int local_vertex_index)
    {
        return vertices_global_indices[local_vertex_index];
    }

    int getLocalVertexIndex(unsigned int global_vertex_index)
    {
        int local_vertex_index = -1;
        for (size_t i; i < vertices_global_indices.size(); i++)
        {
            if (vertices_global_indices[i] == global_vertex_index)
            {
                local_vertex_index = i;
                break;
            }
        }
        return local_vertex_index;
    }

    mesh getObservedMesh(Sophus::SE3f &pose, camera &cam)
    {
        mesh obsMesh;
        int lvl = 0;
        // for each triangle
        for (size_t i = 0; i < triangle_index_to_vertices_indeces.size(); i++)
        {
            std::array<unsigned int, 3> vertex_index = getVertexIndexes(i);
            bool isTriangleVisible = false;
            for (int j = 0; j < 3; j++)
            {
                std::array<float, 3> _vertex = getVertex(vertex_index[j]);
                Eigen::Vector3f vertex;
                vertex(0) = _vertex[0];
                vertex(1) = _vertex[1];
                vertex(2) = _vertex[2];
                if (isRayIdepth)
                    vertex = fromRayIdepthToVertex(vertex);

                vertex = pose * vertex;

                Eigen::Vector2f pix = cam.project(vertex, lvl);
                if (cam.isPixVisible(pix, lvl))
                {
                    isTriangleVisible = true;
                }
            }

            if (isTriangleVisible)
            {
                // so if the triangle is visible:
                // 1- check if vertice is not yet added
                //    if not, add vertices
                //            add vertex to global vertex map
                // 3- add triangle
                // 4- add vertex to triangle map
                std::array<unsigned int, 3> local_vertex_indices;
                for (int j = 0; j < 3; j++)
                {
                    int local_vertex_index = obsMesh.getLocalVertexIndex(vertex_index[j]);
                    if (local_vertex_index < 0)
                    {
                        obsMesh.addVertex(getVertex(vertex_index[j]), vertex_index[j]);
                        local_vertex_index = obsMesh.getVertexSize();
                    }
                    local_vertex_indices[j] = local_vertex_index;
                }
                obsMesh.addTriangle(local_vertex_indices);
            }
        }

        return obsMesh;
    }

private:
    // scene
    // the vertices, actual data of the scene
    std::vector<std::array<float, 3>> vertices;
    // the indices gives me, for each triangle index, the vertices indices
    std::vector<std::array<unsigned int, 3>> triangle_index_to_vertices_indeces;
    // for each vertices index, gives me the triangles indices
    std::vector<std::vector<unsigned int>> vertex_index_to_triangles_indeces;
    // for each vertice inices gives the global indice
    std::vector<unsigned int> vertices_global_indices;

    // std::map<int, std::array<unsigned int, 3>> vertices_to_triangles;

    bool isRayIdepth;
};
