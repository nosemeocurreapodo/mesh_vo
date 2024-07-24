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

    float &operator[](int paramId)
    {
        if (!vector.count(paramId))
            vector[paramId] = 0.0;
        return vector[paramId];
    }

    void add(float value, int paramId)
    {
        if (vector.count(paramId))
            vector[paramId] += value;
        else
            vector[paramId] = value;
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

    std::vector<int> getParamIds()
    {
        std::vector<int> ids;
        for (auto it = vector.begin(); it != vector.end(); ++it)
        {
            ids.push_back(it->first);
        }
        return ids;
    }

    Eigen::VectorXf toEigen(std::vector<int> &paramIds)
    {
        Eigen::VectorXf eigenVector;
        eigenVector = Eigen::VectorXf::Zero(paramIds.size());

        for (size_t index = 0; index < paramIds.size(); index++)
        {
            int id = paramIds[index];
            eigenVector(index) = vector[id];
        }
        return eigenVector;
    }

    std::unordered_map<int, float> vector;
    //std::map<int, float> vector;

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

    VectorMapped &operator[](int paramId)
    {
        if (!matrix.count(paramId))
            matrix[paramId] = VectorMapped();
        return matrix[paramId];
    }

    void add(float value, int paramId1, int paramId2)
    {
        if (!matrix.count(paramId1))
            matrix[paramId1] = VectorMapped();

        if (!matrix[paramId1].vector.count(paramId2))
            matrix[paramId1].vector[paramId2] = value;
        else
            matrix[paramId1].vector[paramId2] += value;
    }

    void operator+=(MatrixMapped &a)
    {
        for (auto it = a.matrix.begin(); it != a.matrix.end(); ++it)
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

    Eigen::SparseMatrix<float> toEigen2(std::vector<int> paramIds)
    {
        Eigen::SparseMatrix<float> eigenMatrix;
        eigenMatrix = Eigen::SparseMatrix<float>(paramIds.size(), paramIds.size());

        for (size_t y = 0; y < paramIds.size(); y++)
        {
            if (!matrix.count(paramIds[y]))
                continue;

            VectorMapped row = matrix[paramIds[y]];

            for (size_t x = 0; x < paramIds.size(); x++)
            {
                if (!row.vector.count(paramIds[x]))
                    continue;

                float value = row.vector[paramIds[x]];

                eigenMatrix.coeffRef(y, x) = value;
            }
        }
        return eigenMatrix;
    }

    Eigen::SparseMatrix<float> toEigen(std::vector<int> paramIds)
    {
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(paramIds.size() * 3);

        for (size_t y = 0; y < paramIds.size(); y++)
        {
            if (!matrix.count(paramIds[y]))
                continue;

            VectorMapped row = matrix[paramIds[y]];

            for (size_t x = 0; x < paramIds.size(); x++)
            {
                if (!row.vector.count(paramIds[x]))
                    continue;

                float value = row.vector[paramIds[x]];

                tripletList.push_back(T(y, x, value));
                //eigenMatrix.coeffRef(y, x) = value;
            }
        }

        Eigen::SparseMatrix<float> eigenMatrix(paramIds.size(), paramIds.size());
        eigenMatrix.setFromTriplets(tripletList.begin(), tripletList.end());

        return eigenMatrix;
    }

    std::unordered_map<int, VectorMapped> matrix;
    //std::map<int, VectorMapped> matrix;

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