#include "cpu/MeshCPU.h"

MeshCPU::MeshCPU()
{
    isRayIdepth = false;
};

void MeshCPU::init(frameCPU &frame, dataCPU<float> &idepth, camera &cam, int lvl)
{
    vertices.clear();
    triangles.clear();

    // preallocate scene vertices to zero
    for (int y = 0; y < MESH_HEIGHT; y++)
    {
        for (int x = 0; x < MESH_WIDTH; x++)
        {
            Eigen::Vector2f pix;
            pix[0] = cam.width[lvl] * float(x) / float(MESH_WIDTH - 1);
            pix[1] = cam.height[lvl] * float(y) / float(MESH_HEIGHT - 1);
            Eigen::Vector3f ray;
            ray(0) = cam.fxinv[lvl] * pix[0] + cam.cxinv[lvl];
            ray(1) = cam.fyinv[lvl] * pix[1] + cam.cyinv[lvl];
            ray(2) = 1.0;
            float id = idepth.get(pix[1], pix[0], lvl);
            if (id <= 0.0)
                id = 0.5;

            Eigen::Vector3f point;
            if (isRayIdepth)
                point = Eigen::Vector3f(ray(0), ray(1), id);
            else
                point = ray / id;
            VerticeCPU vertex(point, pix, vertices.size());

            vertices.push_back(vertex);
        }
    }

    // init scene indices
    for (int y = 0; y < MESH_HEIGHT; y++)
    {
        for (int x = 0; x < MESH_WIDTH; x++)
        {
            if (x > 0 && y > 0)
            {
                // if (((x % 2 == 0)))
                //  if(((x % 2 == 0) && (y % 2 == 0)) || ((x % 2 != 0) && (y % 2 != 0)))
                // if (rand() > 0.5 * RAND_MAX)
                if (true)
                {
                    unsigned int v11 = x - 1 + y * (MESH_WIDTH);
                    unsigned int v12 = x + (y - 1) * (MESH_WIDTH);
                    unsigned int v13 = x - 1 + (y - 1) * (MESH_WIDTH);

                    unsigned int t1_index = triangles.size();
                    TriangleCPU tri1(vertices[v11], vertices[v12], vertices[v13], t1_index);
                    triangles.push_back(tri1);

                    // Triangle* tri11 = &triangles[t1_index];
                    // vertices[v11].triangles.push_back(tri11);
                    // vertices[v12].triangles.push_back(tri11);
                    // vertices[v13].triangles.push_back(tri11);

                    unsigned int v21 = x + y * (MESH_WIDTH);
                    unsigned int v22 = x + (y - 1) * (MESH_WIDTH);
                    unsigned int v23 = x - 1 + y * (MESH_WIDTH);

                    unsigned int t2_index = triangles.size();
                    TriangleCPU tri2(vertices[v21], vertices[v22], vertices[v23], t2_index);
                    triangles.push_back(tri2);

                    // Triangle* tri22 = &triangles[t2_index];
                    // vertices[v21].triangles.push_back(tri22);
                    // vertices[v22].triangles.push_back(tri22);
                    // vertices[v23].triangles.push_back(tri22);
                }
                else
                {
                    unsigned int v11 = x + y * (MESH_WIDTH);
                    unsigned int v12 = x - 1 + (y - 1) * (MESH_WIDTH);
                    unsigned int v13 = x - 1 + y * (MESH_WIDTH);

                    unsigned int t1_index = triangles.size();
                    TriangleCPU tri1(vertices[v11], vertices[v12], vertices[v13], t1_index);
                    triangles.push_back(tri1);

                    // vertices[v11].triangles.push_back(&triangles[t1_index]);
                    // vertices[v12].triangles.push_back(&triangles[t1_index]);
                    // vertices[v13].triangles.push_back(&triangles[t1_index]);

                    unsigned int v21 = x + y * (MESH_WIDTH);
                    unsigned int v22 = x + (y - 1) * (MESH_WIDTH);
                    unsigned int v23 = x - 1 + (y - 1) * (MESH_WIDTH);

                    unsigned int t2_index = triangles.size();
                    TriangleCPU tri2(vertices[v21], vertices[v22], vertices[v23], t2_index);
                    triangles.push_back(tri2);

                    // vertices[v21].triangles.push_back(&triangles[t2_index]);
                    // vertices[v22].triangles.push_back(&triangles[t2_index]);
                    // vertices[v23].triangles.push_back(&triangles[t2_index]);
                }
            }
        }
    }
}

void MeshCPU::initr(frameCPU &frame, dataCPU<float> &idepth, camera &cam, int lvl)
{
    vertices.clear();
    triangles.clear();

    for (int y = 0; y < 2; y++)
    {
        for (int x = 0; x < 2; x++)
        {
            Eigen::Vector2f pix;
            pix[0] = x * (cam.width[lvl] - 1);
            pix[1] = y * (cam.height[lvl] - 1);
            Eigen::Vector3f ray;
            ray(0) = cam.fxinv[lvl] * pix[0] + cam.cxinv[lvl];
            ray(1) = cam.fyinv[lvl] * pix[1] + cam.cyinv[lvl];
            ray(2) = 1.0;
            float id = idepth.get(pix[1], pix[0], lvl);
            if (id <= 0.0)
                id = 0.5;

            Eigen::Vector3f point;
            if (isRayIdepth)
                point = Eigen::Vector3f(ray(0), ray(1), id);
            else
                point = ray / id;
            VerticeCPU vertex(point, pix, vertices.size());

            vertices.push_back(vertex);
        }
    }

    for (int i = 0; i < MESH_HEIGHT * MESH_HEIGHT - 4; i++)
    {
        Eigen::Vector2f pix;
        pix[0] = rand() % cam.width[lvl];
        pix[1] = rand() % cam.height[lvl];
        Eigen::Vector3f ray;
        ray(0) = cam.fxinv[lvl] * pix[0] + cam.cxinv[lvl];
        ray(1) = cam.fyinv[lvl] * pix[1] + cam.cyinv[lvl];
        ray(2) = 1.0;
        float id = idepth.get(pix[1], pix[0], lvl);
        if (id <= 0.0)
            id = 0.5;

        Eigen::Vector3f point;
        if (isRayIdepth)
            point = Eigen::Vector3f(ray(0), ray(1), id);
        else
            point = ray / id;
        VerticeCPU vertex(point, pix, vertices.size());

        vertices.push_back(vertex);
    }

    buildTriangles(cam, lvl);
}

int MeshCPU::getVertexIndexFromId(unsigned int id)
{
    for (int i = 0; i < (int)vertices.size(); i++)
    {
        if (id == vertices[i].id)
            return i;
    }
    return -1;
}

int MeshCPU::getTriangleIndexFromId(unsigned int id)
{
    for (int i = 0; i < (int)triangles.size(); i++)
    {
        if (id == triangles[i].id)
            return i;
    }
    return -1;
}

int MeshCPU::getVertexIdFromIndex(unsigned int in)
{
    if (in < vertices.size())
        return vertices[in].id;
    return -1;
}

int MeshCPU::getTriangleIdFromIndex(unsigned int in)
{
    if (in < triangles.size())
        return triangles[in].id;
    return -1;
}

VerticeCPU MeshCPU::getVertexFromId(unsigned int id)
{
    for (int i = 0; i < (int)vertices.size(); i++)
    {
        if (id == vertices[i].id)
            return vertices[i];
    }
    return VerticeCPU();
}

TriangleCPU MeshCPU::getTriangleFromId(unsigned int id)
{
    for (int i = 0; i < (int)triangles.size(); i++)
    {
        if (id == triangles[i].id)
            return triangles[i];
    }
    return TriangleCPU();
}

VerticeCPU MeshCPU::getVertexFromIndex(unsigned int in)
{
    if (in < vertices.size())
        return vertices[in];
    return VerticeCPU();
}

TriangleCPU MeshCPU::getTriangleFromIndex(unsigned int in)
{
    if (in < triangles.size())
        return triangles[in];
    return TriangleCPU();
}

void MeshCPU::toRayIdepth()
{
    if (!isRayIdepth)
    {
        for (int i = 0; i < (int)vertices.size(); i++)
        {
            vertices[i].position = fromVertexToRayIdepth(vertices[i].position);
        }
        isRayIdepth = true;
    }
}

void MeshCPU::toVertex()
{
    if (isRayIdepth)
    {
        for (int i = 0; i < (int)vertices.size(); i++)
        {
            vertices[i].position = fromRayIdepthToVertex(vertices[i].position);
        }
        isRayIdepth = false;
    }
}

void MeshCPU::transform(Sophus::SE3f &pose)
{
    for (int i = 0; i < (int)vertices.size(); i++)
    {
        Eigen::Vector3f pos = vertices[i].position;
        if (isRayIdepth)
            pos = fromRayIdepthToVertex(pos);
        pos = pose * pos;
        if (isRayIdepth)
            pos = fromVertexToRayIdepth(pos);
        vertices[i].position = pos;
    }
}

void MeshCPU::computeTexCoords(camera &cam, int lvl)
{
    for (int i = 0; i < (int)vertices.size(); i++)
    {
        Eigen::Vector3f ray;
        if (isRayIdepth)
            ray = vertices[i].position;
        else
            ray = vertices[i].position / vertices[i].position(2);

        Eigen::Vector2f pix;
        pix(0) = cam.fx[lvl] * ray(0) + cam.cx[lvl];
        pix(1) = cam.fy[lvl] * ray(1) + cam.cy[lvl];

        vertices[i].texcoord = pix;
    }
}

bool MeshCPU::isTrianglePresent(TriangleCPU &tri)
{
    for (int i = 0; i < (int)triangles.size(); i++)
    {
        TriangleCPU tri2 = triangles[i];
        bool isVertThere[3];
        isVertThere[0] = false;
        isVertThere[1] = false;
        isVertThere[2] = false;
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                if (tri.vertices[j]->id == tri2.vertices[k]->id)
                    isVertThere[j] = true;
            }
        }
        if (isVertThere[0] == true && isVertThere[1] == true && isVertThere[2] == true)
            return true;
    }
    return false;
}

void MeshCPU::removeVerticeByIndex(int vertice_index)
{
    vertices.erase(vertices.begin() + vertice_index);
    buildTriangles(triangles);
}

void MeshCPU::removeVerticeById(int vertice_id)
{
    int vertice_index = getVertexIndexFromId(vertice_id);
    vertices.erase(vertices.begin() + vertice_index);
    buildTriangles(triangles);
}

void MeshCPU::removeTriangleByIndex(int triangle_index)
{
    triangles.erase(triangles.begin() + triangle_index);
}

void MeshCPU::removeTriangleById(int triangle_id)
{
    int triangle_index = getTriangleIndexFromId(triangle_id);
    vertices.erase(vertices.begin() + triangle_index);
}

void MeshCPU::buildTriangles(camera &cam, int lvl)
{
    toVertex();

    triangles.clear();

    dataCPU<int> triIdImage(-1);

    for (int y = 0; y < cam.height[lvl]; y++)
    {
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            Eigen::Vector2f pix(x, y);
            if (triIdImage.get(pix(1), pix(0), lvl) != triIdImage.nodata)
                continue;

            MeshCPU copyMesh = getCopy();

            // search closest
            int closest_vertice_index = copyMesh.getClosestVerticeIndex(pix);
            int closest_vertice_id = copyMesh.vertices[closest_vertice_index].id;

            // search second closest
            copyMesh.removeVerticeByIndex(closest_vertice_index);
            int second_closest_vertice_index = copyMesh.getClosestVerticeIndex(pix);
            int second_closest_vertice_id = copyMesh.vertices[second_closest_vertice_index].id;

            // search third closest
            copyMesh.removeVerticeByIndex(second_closest_vertice_index);
            int third_closest_vertice_index = copyMesh.getClosestVerticeIndex(pix);
            int third_closest_vertice_id = copyMesh.vertices[third_closest_vertice_index].id;

            closest_vertice_index = getVertexIndexFromId(closest_vertice_id);
            second_closest_vertice_index = getVertexIndexFromId(closest_vertice_id);
            third_closest_vertice_id = getVertexIndexFromId(closest_vertice_id);

            TriangleCPU tri(vertices[closest_vertice_index], vertices[second_closest_vertice_index], vertices[third_closest_vertice_index], triangles.size());
            if (isTrianglePresent(tri))
                continue;

            Eigen::Vector3f ray = cam.toRay(pix, lvl);
            tri.arrageClockwise(ray);
            tri.computeTinv();

            tri.computeBarycentric(pix);
            if (!tri.isBarycentricOk())
                continue;

            // rasterize triangle, so we know which pixels are already taken by a triangle
            bool otherTriangle = false;
            std::array<Eigen::Vector2f, 2> minmax = tri.getMinMax();
            for (int py = minmax[0](1); py <= minmax[1](1); py++)
            {
                if (otherTriangle)
                    break;
                for (int px = minmax[0](0); px <= minmax[1](0); px++)
                {
                    Eigen::Vector2f ppix = Eigen::Vector2f(px, py);
                    if (!cam.isPixVisible(ppix, lvl))
                        continue;

                    tri.computeBarycentric(ppix);
                    if (!tri.isBarycentricOk())
                        continue;

                    if (triIdImage.get(ppix(1), ppix(0), lvl) != triIdImage.nodata)
                    {
                        otherTriangle = true;
                        break;
                    }
                }
            }

            if (otherTriangle)
                continue;

            triangles.push_back(tri);

            // rasterize triangle, so we know which pixels are already taken by a triangle

            for (int py = minmax[0](1); py <= minmax[1](1); py++)
            {
                for (int px = minmax[0](0); px <= minmax[1](0); px++)
                {
                    Eigen::Vector2f ppix = Eigen::Vector2f(px, py);
                    if (!cam.isPixVisible(ppix, lvl))
                        continue;

                    tri.computeBarycentric(ppix);
                    if (!tri.isBarycentricOk())
                        continue;

                    triIdImage.set(tri.id, ppix(1), ppix(0), lvl);
                }
            }
        }
    }

    return;

    for (int i = 0; i < (int)vertices.size(); i++)
    {
        VerticeCPU vert = vertices[i];

        /*
        Eigen::Rotation2Df rot(2.0*3.14*1.0/8.0);
        Eigen::Vector2f shifts[8];
        shifts[0] = Eigen::Rotation2Df(2.0*3.14*1.0/16.0)*Eigen::Vector2f(3, 0);
        shifts[1] = rot*shifts[0];
        shifts[2] = rot*shifts[1];
        shifts[3] = rot*shifts[2];
        shifts[4] = rot*shifts[3];
        shifts[5] = rot*shifts[4];
        shifts[6] = rot*shifts[5];
        shifts[7] = rot*shifts[6];
        */

        float pixel_shift = 3.0;
        Eigen::Vector2f shifts[8];
        shifts[0] = Eigen::Vector2f(pixel_shift, 1.0);
        shifts[1] = Eigen::Vector2f(pixel_shift, pixel_shift);
        shifts[2] = Eigen::Vector2f(1.0, pixel_shift);
        shifts[3] = Eigen::Vector2f(-pixel_shift, pixel_shift);
        shifts[4] = Eigen::Vector2f(-pixel_shift, 1.0);
        shifts[5] = Eigen::Vector2f(-pixel_shift, -pixel_shift);
        shifts[6] = Eigen::Vector2f(1.0, -pixel_shift);
        shifts[7] = Eigen::Vector2f(pixel_shift, -pixel_shift);

        // check for triangles moving slighly in the 4 directions
        for (int j = 0; j < 8; j++)
        {
            VerticeCPU vert_shifted = vert;
            vert_shifted.texcoord += shifts[j];
            if (!cam.isPixVisible(vert_shifted.texcoord, lvl))
                continue;

            if (triIdImage.get(int(vert_shifted.texcoord(1)), int(vert_shifted.texcoord(0)), lvl) != triIdImage.nodata)
                continue;

            MeshCPU copyMesh = getCopy();

            // search closest
            copyMesh.removeVerticeByIndex(i);
            int closest_vertice_index = copyMesh.getClosestVerticeIndex(vert_shifted.texcoord);
            int closest_vertice_id = copyMesh.vertices[closest_vertice_index].id;

            // search second closest
            copyMesh.removeVerticeByIndex(closest_vertice_index);
            int second_closest_vertice_index = copyMesh.getClosestVerticeIndex(vert_shifted.texcoord);
            int second_closest_vertice_id = copyMesh.vertices[second_closest_vertice_index].id;

            closest_vertice_index = getVertexIndexFromId(closest_vertice_id);
            second_closest_vertice_index = getVertexIndexFromId(second_closest_vertice_id);

            TriangleCPU tri(vertices[i], vertices[closest_vertice_index], vertices[second_closest_vertice_index], triangles.size());
            if (isTrianglePresent(tri))
                continue;

            Eigen::Vector3f ray = cam.toRay(vert_shifted.texcoord, lvl);
            tri.arrageClockwise(ray);
            triangles.push_back(tri);

            // rasterize triangle, so we know which pixels are already taken by a triangle
            tri.computeTinv();

            std::array<Eigen::Vector2f, 2> minmax = tri.getMinMax();

            for (int y = minmax[0](1); y <= minmax[1](1); y++)
            {
                for (int x = minmax[0](0); x <= minmax[1](0); x++)
                {
                    Eigen::Vector2f pix = Eigen::Vector2f(x, y);
                    if (!cam.isPixVisible(pix, lvl))
                        continue;

                    tri.computeBarycentric(pix);
                    if (!tri.isBarycentricOk())
                        continue;

                    triIdImage.set(tri.id, pix(1), pix(0), lvl);
                }
            }
        }
    }
}

void MeshCPU::buildTriangles(std::vector<TriangleCPU> &tris)
{
    std::vector<TriangleCPU> new_triangles;
    for (int i = 0; i < (int)tris.size(); i++)
    {
        TriangleCPU tri = tris[i];

        unsigned int v1_in = getVertexIndexFromId(tri.vertices[0]->id);
        unsigned int v2_in = getVertexIndexFromId(tri.vertices[1]->id);
        unsigned int v3_in = getVertexIndexFromId(tri.vertices[2]->id);

        if (v1_in < 0 || v2_in < 0 || v3_in < 0)
            continue;

        unsigned int id = tri.id;

        TriangleCPU new_tri(vertices[v1_in], vertices[v2_in], vertices[v3_in], id);
        new_triangles.push_back(new_tri);

        // vertices[v1_in].triangles.push_back(&new_tri);
        // vertices[v1_in].triangles.push_back(&new_tri);
        // vertices[v1_in].triangles.push_back(&new_tri);
    }
    triangles = new_triangles;
}

MeshCPU MeshCPU::getCopy()
{
    MeshCPU meshCopy;

    meshCopy.vertices = vertices;
    meshCopy.buildTriangles(triangles);
    meshCopy.isRayIdepth = isRayIdepth;

    return meshCopy;
}

int MeshCPU::getClosestTriangleIndex(Eigen::Vector3f &pos)
{
    float closest_distance = std::numeric_limits<float>::max();
    int closest_index = -1;
    for (int t = 0; t < (int)triangles.size(); t++)
    {
        TriangleCPU tri = triangles[t];
        Eigen::Vector3f tri_mean = tri.getMeanPosition();
        float distance = (tri_mean - pos).norm();
        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_index = t;
        }
    }
    return closest_index;
}

int MeshCPU::getClosestTriangleIndex(Eigen::Vector2f &pix)
{
    float closest_distance = std::numeric_limits<float>::max();
    int closest_index = -1;
    for (int t = 0; t < (int)triangles.size(); t++)
    {
        TriangleCPU tri = triangles[t];
        Eigen::Vector2f tri_mean = tri.getMeanTexCoord();
        float distance = (tri_mean - pix).norm();
        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_index = t;
        }
    }
    return closest_index;
}

TriangleCPU MeshCPU::getClosestTriangle(Eigen::Vector3f &pos)
{
    int closest_index = getClosestTriangleIndex(pos);
    if(closest_index < 0)
        return TriangleCPU();
    return triangles[closest_index];
}

TriangleCPU MeshCPU::getClosestTriangle(Eigen::Vector2f &pix)
{
    int closest_index = getClosestTriangleIndex(pix);
    if(closest_index < 0)
        return TriangleCPU();
    return triangles[closest_index];
}

int MeshCPU::getClosestVerticeIndex(Eigen::Vector2f &pix)
{
    float closest_distance = std::numeric_limits<float>::max();
    int closest_index = -1;
    for (int i = 0; i < (int)vertices.size(); i++)
    {
        float distance = (vertices[i].texcoord - pix).norm();

        // float x_distance = std::fabs(v_vector[i].texcoord(0) - v.texcoord(0));
        // float y_distance = std::fabs(v_vector[i].texcoord(1) - v.texcoord(1));
        // float distance = std::max(x_distance, y_distance);

        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_index = i;
        }
    }
    return closest_index;
}

int MeshCPU::getClosestVerticeIndex(Eigen::Vector3f &v)
{
    float closest_distance = std::numeric_limits<float>::max();
    int closest_vertice = -1;
    for (int i = 0; i < (int)vertices.size(); i++)
    {
        float distance = (vertices[i].position - v).norm();

        // float x_distance = std::fabs(v_vector[i].texcoord(0) - v.texcoord(0));
        // float y_distance = std::fabs(v_vector[i].texcoord(1) - v.texcoord(1));
        // float distance = std::max(x_distance, y_distance);

        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_vertice = i;
        }
    }
    return closest_vertice;
}

VerticeCPU MeshCPU::getClosestVertice(Eigen::Vector3f &pos)
{
    int closest_index = getClosestVerticeIndex(pos);
    if(closest_index < 0)
        return VerticeCPU();
    return vertices[closest_index];
}

VerticeCPU MeshCPU::getClosestVertice(Eigen::Vector2f &pix)
{
    int closest_index = getClosestVerticeIndex(pix);
    if(closest_index < 0)
        return VerticeCPU();
    return vertices[closest_index];
}
