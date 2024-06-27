#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "params.h"

class MapVector
{
public:
    MapVector()
    {
    }

    void clear()
    {
        vector.clear();
    }

    float &operator[](unsigned int id)
    {
        if (!vector.count(id))
            vector[id] = 0.0;
        return vector[id];
    }

    /*
    void add(float value, unsigned int id)
    {
        if(vector.count(id))
            vector[id] += value;
        else
            vector[id] = value;
    }
    */

    void operator+=(MapVector &a)
    {
        for (std::map<unsigned int, float>::iterator it = a.vector.begin(); it != a.vector.end(); ++it)
        {
            if (!vector.count(it->first))
            {
                vector[it->first] = a.vector[it->first];
            }
            else
            {
                vector[it->first] = vector[it->first] + a.vector[it->first];
            }
        }
    }

    MapVector operator+(MapVector &a)
    {
        MapVector result;
        result += a;
        result += *this;
        return result;
    }

    std::vector<unsigned int> getIds()
    {
        std::vector<unsigned int> ids;
        for (std::map<unsigned int, float>::iterator it = vector.begin(); it != vector.end(); ++it)
        {
            ids.push_back(it->first);
        }
        return ids;
    }

    Eigen::VectorXf toEigen(std::vector<unsigned int> &ids)
    {
        Eigen::VectorXf eigenVector;
        eigenVector = Eigen::VectorXf::Zero(ids.size());

        for (int index = 0; index < ids.size(); index++)
        {
            unsigned int id = ids[index];
            eigenVector(index) = vector[id];
        }
        return eigenVector;
    }

    std::map<unsigned int, float> vector;

private:
};

class MapMatrix
{
public:
    MapMatrix()
    {
    }

    void clear()
    {
        matrix.clear();
    }

    MapVector &operator[](unsigned int id)
    {
        if (!matrix.count(id))
            matrix[id] = MapVector();
        return matrix[id];
    }

    /*
    void add(float value, unsigned int id1, unsigned int id2)
    {
        if(!matrix.count(id1))
            matrix[id1] = MapVector();

        if(matrix[id1].vector.count(id2))
            matrix[id1].vector[id2] = value;
        else
            matrix[id1].vector[id2] += value;

    }
    */

    void operator+=(MapMatrix &a)
    {
        for (std::map<unsigned int, MapVector>::iterator it = a.matrix.begin(); it != a.matrix.end(); ++it)
        {
            if (!matrix.count(it->first))
            {
                matrix[it->first] = a.matrix[it->first];
            }
            else
            {
                matrix[it->first] = matrix[it->first] + a.matrix[it->first];
            }
        }
    }

    MapMatrix operator+(MapMatrix &a)
    {
        MapMatrix result;
        result += a;
        result += *this;
        return result;
    }

    Eigen::SparseMatrix<float> toEigen(std::vector<unsigned int> ids)
    {
        Eigen::SparseMatrix<float> eigenMatrix;
        eigenMatrix = Eigen::SparseMatrix<float>(ids.size(), ids.size());

        for (int y = 0; y < ids.size(); y++)
        {
            if (!matrix.count(ids[y]))
                continue;

            MapVector row = matrix[ids[y]];

            for (int x = 0; x < ids.size(); x++)
            {
                if (!row.vector.count(ids[x]))
                    continue;
    
                float value = row.vector[ids[y]];

                eigenMatrix.coeffRef(y, x) = value;
            }
        }
        return eigenMatrix;
    }

    std::map<unsigned int, MapVector> matrix;

private:
};

class HGPoseMapMesh
{
public:
    HGPoseMapMesh()
    {
        count = 0;
    }

    void setZero()
    {
        H.clear();
        G.clear();
        count = 0;
    }

    HGPoseMapMesh operator+(HGPoseMapMesh a)
    {
        HGPoseMapMesh sum;

        sum.H = H + a.H;
        sum.G = G + a.G;
        sum.count = count + a.count;

        return sum;
    }

    void operator+=(HGPoseMapMesh a)
    {
        G += a.G;
        H += a.H;
        count += a.count;
    }

    MapMatrix H;
    MapVector G;
    unsigned int count;

private:
};