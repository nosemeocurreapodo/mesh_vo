#pragma once

#include "common/types.h" // expects typedefs: using vecxf = Eigen::VectorXf; using matxf = Eigen::MatrixXf; etc.
#include <cassert>
#include <cstdint>
#include <vector>

// ===== Dense normal-equation accumulator with packed H for thread scalability =====
class DenseLinearProblem;

class DenseLinearProblem
{
public:
    DenseLinearProblem() : m_numParams(0), m_count(0) {}
    explicit DenseLinearProblem(int numParams) { reset(numParams); }

    void reset(int n)
    {
        assert(n >= 0);
        m_numParams = n;
        m_Hp = Matx::Zero(n, n);
        m_G = Vecx::Zero(n);
        m_count = 0;
    }

    void clear()
    {
        if (m_numParams == 0)
            return;
        m_Hp.setZero();
        m_G.setZero();
        m_count = 0;
    }

    int size() const { return m_numParams; }

    void add(const Matx &J, const Vecx &r, float w = 1.0f)
    {
        if (w <= 0.0f)
            return;
        m_Hp += w * J.transpose() * J;
        m_G += w * J.transpose() * r;
        m_count++;
    }

    template <typename Jac, typename Idx>
    inline void add(const Jac &J,
                    float r,
                    float w,
                    const Idx &ids)
    {
        if (w <= 0.0f)
            return;

        for (int i = 0; i < J.rows(); i++)
        {
            m_G(ids(i)) += J(i) * r * w;
            m_Hp(ids(i), ids(i)) += J(i) * J(i) * w;

            for (int j = i + 1; j < J.rows(); j++)
            {
                float jj = J(i) * J(j) * w;
                m_Hp(ids(i), ids(j)) += jj;
                m_Hp(ids(j), ids(i)) += jj;
            }
        }

        ++m_count;
    }

    DenseLinearProblem &operator+=(const DenseLinearProblem &other)
    {
        if (other.m_numParams == 0)
            return *this;
        if (m_numParams == 0)
        {
            *this = other;
            return *this;
        }
        assert(m_numParams == other.m_numParams);
        m_Hp += other.m_Hp;
        m_G.noalias() += other.m_G;
        m_count += other.m_count;
        return *this;
    }

    void scale(float s)
    {
        m_Hp *= s;
        m_G *= s;
    }

    Vecx solve(float lambda = 0.0f, int lambda_mode = 1)
    {
        Matx H = m_Hp;

        /*
        if (lambda > 0.0f)
        {
            if (lambda_mode == 0)
            {
                H.diagonal().array() += lambda;
            }
            else
            {
                Eigen::VectorXf d = H.diagonal().array().abs().max(1e-8f);
                H.diagonal().noalias() += lambda * d;
            }
        }
        */
        for (int j = 0; j < m_G.size(); j++)
        {
            H(j, j) *= (1.0 + lambda);
        }
        solver.compute(H);
        assert(solver.info() == Eigen::Success);
        Vecx dx = solver.solve(-m_G);
        assert(solver.info() == Eigen::Success);
        return dx;
    }

    int count() const { return m_count; }
    const Vecx &G() const { return m_G; }

private:
    Matx m_Hp;
    Vecx m_G;
    Solver solver;
    int m_numParams{0};
    int m_count{0};
};