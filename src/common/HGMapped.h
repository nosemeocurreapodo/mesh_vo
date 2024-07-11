#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "params.h"

class VectorMapped
{
public:
    VectorMapped()
    {
    }

    VectorMapped(const VectorMapped &other)
    {
        vector = other.vector;
    }

    VectorMapped &operator=(const VectorMapped &other)
    {
        if (this != &other)
        {
            vector = other.vector;
        }
        return *this;
    }

    void clear()
    {
        vector.clear();
    }

    float &operator[](int id)
    {
        if (!vector.count(id))
            vector[id] = 0.0;
        return vector[id];
    }

    void add(float value, int id)
    {
        if(vector.count(id))
            vector[id] += value;
        else
            vector[id] = value;
    }

    void operator+=(VectorMapped &a)
    {
        for (auto it = a.vector.begin(); it != a.vector.end(); ++it)
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

    VectorMapped operator+(VectorMapped &a)
    {
        VectorMapped result;
        result += a;
        result += *this;
        return result;
    }

    std::vector<int> getIds()
    {
        std::vector<int> ids;
        for (auto it = vector.begin(); it != vector.end(); ++it)
        {
            ids.push_back(it->first);
        }
        return ids;
    }

    Eigen::VectorXf toEigen(std::vector<int> &ids)
    {
        Eigen::VectorXf eigenVector;
        eigenVector = Eigen::VectorXf::Zero(ids.size());

        for (int index = 0; index < ids.size(); index++)
        {
            int id = ids[index];
            eigenVector(index) = vector[id];
        }
        return eigenVector;
    }

    std::map<int, float> vector;

private:
};

class MatrixMapped
{
public:
    MatrixMapped()
    {
    }

    MatrixMapped(const MatrixMapped &other)
    {
        matrix = other.matrix;
    }

    MatrixMapped &operator=(const MatrixMapped &other)
    {
        if (this != &other)
        {
            matrix = other.matrix;
        }
        return *this;
    }

    void clear()
    {
        matrix.clear();
    }

    VectorMapped &operator[](int id)
    {
        if (!matrix.count(id))
            matrix[id] = VectorMapped();
        return matrix[id];
    }

    void add(float value, int id1, int id2)
    {
        if(!matrix.count(id1))
            matrix[id1] = VectorMapped();

        if(!matrix[id1].vector.count(id2))
            matrix[id1].vector[id2] = value;
        else
            matrix[id1].vector[id2] += value;
    }

    void operator+=(MatrixMapped &a)
    {
        for (std::map<int, VectorMapped>::iterator it = a.matrix.begin(); it != a.matrix.end(); ++it)
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

    MatrixMapped operator+(MatrixMapped &a)
    {
        MatrixMapped result;
        result += a;
        result += *this;
        return result;
    }

    Eigen::SparseMatrix<float> toEigen(std::vector<int> ids)
    {
        Eigen::SparseMatrix<float> eigenMatrix;
        eigenMatrix = Eigen::SparseMatrix<float>(ids.size(), ids.size());

        for (int y = 0; y < ids.size(); y++)
        {
            if (!matrix.count(ids[y]))
                continue;

            VectorMapped row = matrix[ids[y]];

            for (int x = 0; x < ids.size(); x++)
            {
                if (!row.vector.count(ids[x]))
                    continue;
    
                float value = row.vector[ids[x]];

                eigenMatrix.coeffRef(y, x) = value;
            }
        }
        return eigenMatrix;
    }

    std::map<int, VectorMapped> matrix;

private:
};

class HGMapped
{
public:
    HGMapped()
    {
        count = 0;
    }

    void setZero()
    {
        H.clear();
        G.clear();
        count = 0;
    }

    HGMapped operator+(HGMapped a)
    {
        HGMapped sum;

        sum.H = H + a.H;
        sum.G = G + a.G;
        sum.count = count + a.count;

        return sum;
    }

    void operator+=(HGMapped a)
    {
        G += a.G;
        H += a.H;
        count += a.count;
    }

    MatrixMapped H;
    VectorMapped G;
    int count;

private:
};