#pragma once

/*
#include <Eigen/Core>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include "sophus/se3.hpp"

using vec2f = Eigen::Vector2f;
using vec3f = Eigen::Vector3f;
using vec6f = Eigen::Matrix<float, 6, 1>;
using vecxf = Eigen::VectorXf;

using mat3f = Eigen::Matrix3f;
using mat6f = Eigen::Matrix<float, 6, 6>;
using matxf = Eigen::MatrixXf;

using vec2i = Eigen::Vector2i;
using vec3i = Eigen::Vector3i;
using vecxi = Eigen::VectorXi;

using SE3f = Sophus::SE3f;

using solverType = Eigen::LDLT<Eigen::MatrixXf>;

    // Eigen::LLT<Eigen::MatrixXf> solver;
    Eigen::LDLT<Eigen::MatrixXf> solver;

    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> ssolver;
    // Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
    // Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
    //  Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
    //  Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;
    //  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<float>, Eigen::Lower> solver;
    //  Eigen::SPQR<Eigen::SparseMatrix<float>> solver;

*/

#include "common/lingalglib.h"

using vec2f = vec2<float>;
using vec3f = vec3<float>;
using vec6f = vec6<float>;
using vecxf = vecx<float>;

using mat3f = mat3<float>;
using mat6f = mat6<float>;
using matxf = matx<float>;

using vec2i = vec2<int>;
using vec3i = vec3<int>;
using vecxi = vecx<int>;

using SE3f = SE3<float>;

using imageType = float;
using jmapType = vec3f;
using idsType = vec3i;

class ShapeTriangleFlat;
class SceneMesh;

using shapeType = ShapeTriangleFlat;
using sceneType = SceneMesh;
using solverType = LDLT<matxf>;

struct vertex
{
    vertex()
    {
        used = false;
    }

    vertex(vec3f v, vec3f r, vec2f p)
    {
        ver = v;
        ray = r;
        pix = p;
        // weight = w;
        used = true;
    }

    vec3f ver;
    vec3f ray;
    vec2f pix;
    // float weight;
    bool used;
};

struct triangle
{
    triangle()
    {
        used = false;
    }

    triangle(vec3i i)
    {
        vertexIds = i;
        used = true;
    }

    vec3i vertexIds;
    bool used;
};