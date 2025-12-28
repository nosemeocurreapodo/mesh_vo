#pragma once

// #include <Eigen/Core>
// include <Eigen/Dense>
//  #include <Eigen/Cholesky>
//  #include <Eigen/Sparse>
// #include <unsupported/Eigen/MatrixFunctions>
// #include "sophus/se3.hpp"
// #include "sophus/sim3.hpp"

#include "core/types.h"
#include "core/camera.h"

#include "backends/cpu/buffercpu.h"
#include "backends/cpu/texturecpu.h"
#include "backends/cpu/meshcpu.h"
#include "backends/cpu/renderercpu.h"

using DepthRenderer = DepthRendererCPU;
using ImageRenderer = ImageRendererCPU;
using ResidualRenderer = ResidualRendererCPU;
using DIDxyRenderer = DIDxyRendererCPU;
using DIDexpRenderer = DIDexpRendererCPU;
using JPoseFDRenderer = JPoseFDRendererCPU;
using JPoseExpRenderer = JPoseExpRendererCPU;
using JPoseVelExpRenderer = JPoseVelExpRendererCPU;
using JMapFDRenderer = JMapFDRendererCPU;
using JMapExpRenderer = JMapExpRendererCPU;
using JPoseExpMapRenderer = JPoseExpMapRendererCPU;
using JPoseVelExpMapRenderer = JPoseVelExpMapRendererCPU;

template <typename T>
using Texture = TextureCPU<T>;
using Mesh = MeshCPU;

/*
#include "backends/gl/buffergl.h"
#include "backends/gl/texturegl.h"
#include "backends/gl/meshgl.h"
#include "backends/gl/renderergl.h"

using DepthRenderer = DepthRendererGL;
using ImageRenderer = ImageRendererGL;
using ResidualRenderer = ResidualRendererGL;
using DIDxyRenderer = DIDxyRendererGL;
using JPoseExpRenderer = JPoseExpRendererGL;
using JMapExpRenderer = JMapExpRendererGL;
using JPoseExpMapRenderer = JPoseExpMapRendererGL;

template <typename T>
using Texture = TextureGL<T>;
using Mesh = MeshGL;
*/
using Camera = PinholeCamera<float>;

using Vecxf = Vecx<float>;
using Matxf = Matx<float>;

template <int N>
using Vecf = Vec<float, N>;
using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec5f = Vec5<float>;
using Vec6f = Vec6<float>;
using Vec8f = Vec<float, 8>;

using Vec3i = Vec3<int>;
using Vec5i = Vec5<int>;
using Vec6i = Vec6<int>;
using Vec8i = Vec<int, 8>;

template <int C, int R>
using Matf = Mat<float, C, R>;
using Mat6f = Mat6<float>;

using SE3f = SE3<float>;

template <typename T, int size>
using Solver = LDLT<T, size>;

template <typename T>
using Solverx = LDLTx<T>;

// using Solver = Eigen::LDLT<Eigen::MatrixXf>;

// template <typename type, int rows>
// using vec = Eigen::Matrix<type, rows, 1>;

// using vec2f = Eigen::Matrix<float, 2, 1>;
// using vec3f = Eigen::Matrix<float, 3, 1>;
// using vec4f = Eigen::Matrix<float, 4, 1>;
// using vec5f = Eigen::Matrix<float, 5, 1>;
// using vec6f = Eigen::Matrix<float, 6, 1>;
// using vecxf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

// using vec2i = Eigen::Matrix<int, 2, 1>;
// using vec3i = Eigen::Matrix<int, 3, 1>;
// using vecxi = Eigen::Matrix<int, Eigen::Dynamic, 1>;

// template <typename type, int rows, int cols>
// using mat = Eigen::Matrix<type, rows, cols>;
// using mat3f = Eigen::Matrix<float, 3, 3>;
// using mat4f = Eigen::Matrix<float, 4, 4>;
// using mat6f = Eigen::Matrix<float, 6, 6>;
// using matxf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

// using SE3f = Sophus::SE3f;
// using SIM3f = Sophus::Sim3f;

// using Solver = Eigen::LDLT<Eigen::MatrixXf>;
// using Solver = Eigen::LLT<Eigen::MatrixXf>;

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

/*
using Image = float;
using Jimg = Vec2;
using Jpose = Vec6;
using Jvel = Vec6;
using Jmap = Vec3;
using Jcam = Vec4;
using Ids = Vec3i;

template <typename T>
class BufferCPU;
template <class T>
class TextureCPU;
class MeshCPU;
class DIDxyRendererCPU;
class JtraRendererCPU;
class JrotRendererCPU;

using Buffer = BufferCPU;
using Texture = TextureCPU;
using Mesh = MeshCPU;
using DIDxyRenderer = DIDxyRendererCPU;
using JtraRenderer = JtraRendererCPU;
using JrotRenderer = JrotRendererCPU;
*/