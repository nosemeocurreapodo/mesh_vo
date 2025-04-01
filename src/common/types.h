#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
// #include <Eigen/Cholesky>
// #include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include "sophus/se3.hpp"
#include "sophus/sim3.hpp"

template <typename type, int rows>
using vec = Eigen::Matrix<type, rows, 1>;

using vec2f = Eigen::Matrix<float, 2, 1>;
using vec3f = Eigen::Matrix<float, 3, 1>;
using vec4f = Eigen::Matrix<float, 4, 1>;
using vec5f = Eigen::Matrix<float, 5, 1>;
using vec6f = Eigen::Matrix<float, 6, 1>;
using vecxf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using vec2i = Eigen::Matrix<int, 2, 1>;
using vec3i = Eigen::Matrix<int, 3, 1>;
using vecxi = Eigen::Matrix<int, Eigen::Dynamic, 1>;

template <typename type, int rows, int cols>
using mat = Eigen::Matrix<type, rows, cols>;
using mat3f = Eigen::Matrix<float, 3, 3>;
using mat4f = Eigen::Matrix<float, 4, 4>;
using mat6f = Eigen::Matrix<float, 6, 6>;
using matxf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

using SE3f = Sophus::SE3f;
using SIM3f = Sophus::Sim3f;

using solverType = Eigen::LDLT<Eigen::MatrixXf>;
// using solverType = Eigen::LLT<Eigen::MatrixXf>;

// Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> ssolver;
// Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
// Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
// Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
// Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;
// Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<float>, Eigen::Lower> solver;
// Eigen::SPQR<Eigen::SparseMatrix<float>> solver;

/*
#include "common/lingalglib.h"
#include "common/ldlt_solver.h"

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

using solverType = LDLT<matxf>;
*/

class pinholeCamera;
class pinholeDistortedCamera;

using cameraType = pinholeCamera;

class keyFrameCPU;
class frameCPU;

using imageType = float;
using errorType = float;
using jimgType = vec2f;
using jposeType = vec6f;
using jvelType = vec6f;
using jmapType = vec3f;
using jcamType = vec4f;
using idsType = vec3i;

class ShapeTriangleFlat;
class GeometryMesh;

using shapeType = ShapeTriangleFlat;
using geometryType = GeometryMesh;

struct vertex
{
    vertex()
    {
        used = false;
    }

    vertex(vec3f v, vec3f r, vec2f p, float w)
    {
        ver = v;
        ray = r;
        pix = p;
        weight = w;
        used = true;
    }

    vec3f ver;
    vec3f ray;
    vec2f pix;
    float weight;
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