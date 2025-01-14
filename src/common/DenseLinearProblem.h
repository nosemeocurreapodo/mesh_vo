#pragma once

#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>

class DenseLinearProblem
{
public:
    DenseLinearProblem(int numParams)
    {
        m_numParams = numParams;

        m_H = Eigen::MatrixXf::Zero(numParams, numParams);
        m_lH = Eigen::MatrixXf::Zero(numParams, numParams);
        m_G = Eigen::VectorXf::Zero(numParams);

        m_count = 0;
    }

    void setZero()
    {
        m_H.setZero();
        m_lH.setZero();
        m_G.setZero();
        m_count = 0;
    }

    // for vector error (not scalar error like pixel errors)
    // and matrix jacobian (not vector)
    template <typename jacType, typename errType>
    void add(jacType jac, errType error, float weight)
    {
        assert(error.cols() == 1);
        assert(error.rows() == jac.rows());
        assert(jac.cols() == m_numParams);
        assert(jac.rows() == m_numParams);

        m_count++;

        jacType J = jac * jac.transpose();
        errType G = jac * error;

        m_G += G * weight;
        m_H += J * weight;
    }

    // for vector error (not scalar error like pixel errors)
    template <typename jacType, typename errType, typename idsType>
    void add(jacType jac, errType error, float weight, idsType ids)
    {
        assert(error.cols() == 1);
        assert(error.rows() == jac.rows());
        assert(jac.cols() == m_numParams);
        assert(jac.rows() == m_numParams);

        m_count++;

        jacType J = jac * jac.transpose();
        errType G = jac * error;

        m_G += G * weight;
        m_H += J * weight;
    }

    template <typename jacType>
    void add(jacType jac, float error, float weight)
    {
        assert(jac.rows() == m_numParams);

        m_count++;

        // G += J * error;
        // H += J * J.transpose();

        for (int i = 0; i < jac.rows(); i++)
        {
            m_G(i) += jac(i) * error * weight;
            m_H(i, i) += jac(i) * jac(i) * weight;

            for (int j = i + 1; j < jac.rows(); j++)
            {
                float jj = jac(i) * jac(j) * weight;
                m_H(i, j) += jj;
                m_H(j, i) += jj;
            }
        }
    }

    template <typename jacType, typename idsType>
    void add(jacType jac, float error, float weight, idsType ids)
    {
        assert(jac.rows() == ids.rows());
        assert(jac.cols() == 1);
        assert(ids.cols() == 1);
        assert(jac.rows() <= m_numParams);

        m_count++;

        // G += J * error;
        // H += J * J.transpose();

        for (int i = 0; i < jac.rows(); i++)
        {
            m_G(ids(i)) += jac(i) * error * weight;
            m_H(ids(i), ids(i)) += jac(i) * jac(i) * weight;

            for (int j = i + 1; j < jac.rows(); j++)
            {
                float jj = jac(i) * jac(j) * weight;
                m_H(ids(i), ids(j)) += jj;
                m_H(ids(j), ids(i)) += jj;
            }
        }
    }

    DenseLinearProblem operator+(DenseLinearProblem a)
    {
        assert(m_numParams == a.m_numParams);

        DenseLinearProblem sum(m_numParams);

        /*
        if(m_count == 0)
        {
            sum.m_H = a.m_H/a.m_count;
            sum.m_G = a.m_G/a.m_count;
        }
        else
        {
            sum.m_H = m_H/m_count + a.m_H/a.m_count;
            sum.m_G = m_G/m_count + a.m_G/a.m_count;
        }

        sum.m_count = 1;
        */

        sum.m_H = m_H + a.m_H;
        sum.m_G = m_G + a.m_G;
        sum.m_count = m_count + a.m_count;

        return sum;
    }

    void operator+=(DenseLinearProblem a)
    {
        assert(m_numParams == a.m_numParams);

        /*
        if(m_count == 0)
        {
            m_H = a.m_H/a.m_count;
            m_G = a.m_G/a.m_count;
        }
        else
        {
            m_H = m_H/m_count + a.m_H/a.m_count;
            m_G = m_G/m_count + a.m_G/a.m_count;
        }

        m_count = 1;
        */

        m_H += a.m_H;
        m_G += a.m_G;
        m_count += a.m_count;
    }

    template <typename type>
    void operator*=(type a)
    {
        m_H *= a;
        m_G *= a;
    }

    std::vector<int> removeUnobservedParams()
    {
        std::vector<int> indicesToKeep;

        for (int i = 0; i < m_G.size(); ++i)
        {
            if (m_G[i] != 0.0)
            {
                indicesToKeep.push_back(i);
            }
        }

        Eigen::VectorXi indicesToKeepVector(indicesToKeep.size());

        for (size_t i = 0; i < indicesToKeep.size(); i++)
        {
            indicesToKeepVector(i) = indicesToKeep[i];
        }

        m_H = m_H(indicesToKeepVector, indicesToKeepVector);
        m_G = m_G(indicesToKeepVector);

        return indicesToKeep;
    }

    std::vector<int> getParamIds()
    {
        std::vector<int> indicesToKeep;

        for (int i = 0; i < m_G.size(); ++i)
        {
            indicesToKeep.push_back(i);
        }

        return indicesToKeep;
    }

    bool prepareH(float lambda)
    {
        m_lH = m_H;
        for (int j = 0; j < m_G.size(); j++)
        {
            m_lH(j, j) *= (1.0 + lambda);
        }
        solver.compute(m_lH);
        return (solver.info() == Eigen::Success);
    }

    Eigen::MatrixXf getH()
    {
        return m_H;
    }

    Eigen::VectorXf solve()
    {
        Eigen::VectorXf res = solver.solve(m_G);
        assert(solver.info() == Eigen::Success);
        return res;
    }

    int getCount()
    {
        return m_count;
    }

private:
    Eigen::MatrixXf m_H;
    Eigen::VectorXf m_G;
    Eigen::MatrixXf m_lH;

    int m_numParams;
    int m_count;

    // Eigen::LLT<Eigen::MatrixXf> solver;
    Eigen::LDLT<Eigen::MatrixXf> solver;

    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> ssolver;
    // Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
    // Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
    //  Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
    //  Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;
    //  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<float>, Eigen::Lower> solver;
    //  Eigen::SPQR<Eigen::SparseMatrix<float>> solver;
};
