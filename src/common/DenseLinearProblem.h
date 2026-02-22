#pragma once

#include <cassert>
// #include <cstdint>
// #include <vector>

#include "common/types.h"

template <int N>
class DenseLinearProblem
{
public:
    DenseLinearProblem()
    {
        clear();
    }

    void clear()
    {
        m_Hp.setZero();
        m_G.setZero();
        m_count = 0;
    }

    static constexpr int size()
    {
        return N;
    }

    void add(const Vecf<N> &J, float r, float w = 1.0f)
    {
        if (w <= 0.0f)
            return;
        // m_Hp += w * J.transpose() * J;
        // m_G += w * J.transpose() * r;
        m_Hp += w * J * J.transpose();
        m_G += w * J * r;
        m_count++;
    }

    DenseLinearProblem &operator+=(const DenseLinearProblem &other)
    {
        m_Hp += other.m_Hp;
        // m_G.noalias() += other.m_G;
        m_G += other.m_G;
        m_count += other.m_count;
        return *this;
    }

    void scale(float s)
    {
        m_Hp *= s;
        m_G *= s;
    }

    int count() const { return m_count; }
    const Mat<float, N, N> &Hp() const { return m_Hp; }
    const Vecf<N> &G() const { return m_G; }

private:
    Mat<float, N, N> m_Hp;
    Vec<float, N> m_G;
    int m_count{0};
};

class DenseLinearProblemx
{
public:
    DenseLinearProblemx() : m_numParams(0), m_count(0) {}
    DenseLinearProblemx(int n)
    {
        m_numParams = n;
        m_Hp = Matxf::Zero(n, n);
        m_G = Vecxf::Zero(n);
        m_count = 0;
    }
    /*
    DenseLinearProblemx &operator=(const DenseLinearProblemx &other)
    {
        if (this == &other)
            return *this;

        clear(other.m_numParams); // ensure sizes match
        m_Hp = other.m_Hp;
        m_G = other.m_G;
        m_count = other.m_count;
        return *this;
    }

    DenseLinearProblemx(const DenseLinearProblemx &) = default;
    DenseLinearProblemx(DenseLinearProblemx &&) noexcept = default;
    DenseLinearProblemx &operator=(DenseLinearProblemx &&) noexcept = default;
    */
    void clear()
    {
        m_Hp.setZero();
        m_G.setZero();
        m_count = 0;
    }

    int size() const { return m_numParams; }

    void add(const Matxf &J, const Matxf &r, float w = 1.0f)
    {
        if (w <= 0.0f)
            return;
        m_Hp += w * J.transpose() * J;
        m_G += w * J.transpose() * r;
        m_count++;
    }

    template <typename Jac, typename Idx>
    void add(const Jac &J,
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

    DenseLinearProblemx &operator+=(const DenseLinearProblemx &other)
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
        // m_G.noalias() += other.m_G;
        m_G += other.m_G;
        m_count += other.m_count;
        return *this;
    }

    void scale(float s)
    {
        m_Hp *= s;
        m_G *= s;
    }

    const Matxf &Hp() const { return m_Hp; }
    const Vecxf &G() const { return m_G; }
    int count() const { return m_count; }

private:
    Matxf m_Hp;
    Vecxf m_G;
    int m_numParams{0};
    int m_count{0};
};