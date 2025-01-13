#pragma once

#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>

class DenseLinearProblem
{
public:
    DenseLinearProblem(int numPoseParams, int numMapParams)
    {
        m_numPoseParams = numPoseParams;
        m_numMapParams = numMapParams;

        m_H = Eigen::MatrixXf::Zero(numPoseParams + numMapParams, numPoseParams + numMapParams);
        m_lH = Eigen::MatrixXf::Zero(numPoseParams + numMapParams, numPoseParams + numMapParams);
        m_G = Eigen::VectorXf::Zero(numPoseParams + numMapParams);
        m_sH = Eigen::SparseMatrix<float>(1, 1);

        m_count = 0;
    }

    void setZero()
    {
        m_H.setZero();
        m_lH.setZero();
        m_G.setZero();
        m_sH = Eigen::SparseMatrix<float>(1, 1);
        m_count = 0;
    }

    // for vector error (not scalar error like pixel errors)
    template <typename jacType, typename errType, typename idsType>
    void add(jacType jacMap, errType error, float weight, idsType mapIds)
    {
        assert(jacType::size() == idsType::size());
        assert(errType::size() == jacType::size());
        assert(jacType::size() <= m_numPoseParams + m_numMapParams);

        m_count++;

        idsType intMapIds = mapIds + idsType(m_numPoseParams);

        // G += J * error;
        // H += J * J.transpose();

        for (int i = 0; i < jacType::size(); i++)
        {
            m_G(intMapIds(i)) += jacMap(i) * error(i) * weight;
            m_H(intMapIds(i), intMapIds(i)) += jacMap(i) * jacMap(i) * weight;

            for (int j = i + 1; j < jacType::size(); j++)
            {
                float jj = jacMap(i) * jacMap(j) * weight;
                m_H(intMapIds(i), intMapIds(j)) += jj;
                m_H(intMapIds(j), intMapIds(i)) += jj;
            }
        }
    }

    template <typename jacType>
    void add(jacType J, float error, float weight)
    {
        assert(jacType::size() <= m_numPoseParams + m_numMapParams);

        m_count++;

        // G += J * error;
        // H += J * J.transpose();

        for (int i = 0; i < jacType::size(); i++)
        {
            m_G(i) += J(i) * error * weight;
            m_H(i, i) += J(i) * J(i) * weight;

            for (int j = i + 1; j < jacType::size(); j++)
            {
                float jj = J(i) * J(j) * weight;
                m_H(i, j) += jj;
                m_H(j, i) += jj;
            }
        }
    }

    template <typename jacType, typename idsType>
    void add(jacType jacMap, float error, float weight, idsType mapIds)
    {
        assert(jacType::size() == idsType::size());
        assert(jacType::size() <= m_numPoseParams + m_numMapParams);

        m_count++;

        idsType intMapIds = mapIds + idsType(m_numPoseParams);

        // G += J * error;
        // H += J * J.transpose();

        for (int i = 0; i < jacType::size(); i++)
        {
            m_G(intMapIds(i)) += jacMap(i) * error * weight;
            m_H(intMapIds(i), intMapIds(i)) += jacMap(i) * jacMap(i) * weight;

            for (int j = i + 1; j < jacType::size(); j++)
            {
                float jj = jacMap(i) * jacMap(j) * weight;
                m_H(intMapIds(i), intMapIds(j)) += jj;
                m_H(intMapIds(j), intMapIds(i)) += jj;
            }
        }
    }

    template <typename jacPoseType, typename jacMapType, typename idsType>
    void add(jacPoseType jacPose, jacMapType jacMap, float error, float weight, int poseId, idsType mapIds)
    {
        assert(jacMapType::size() == idsType::size());
        assert(m_numPoseParams > jacPoseType::size() - 1 + poseId * jacPoseType::size());

        m_count++;

        idsType intMapIds = mapIds + idsType(m_numPoseParams);

        for (int i = 0; i < jacPoseType::size(); i++)
        {
            int poseParamId1 = i + poseId * jacPoseType::size();

            m_G(poseParamId1) += jacPose(i) * error * weight;
            m_H(poseParamId1, poseParamId1) += jacPose(i) * jacPose(i) * weight;

            for (int j = i + 1; j < jacPoseType::size(); j++)
            {
                int poseParamId2 = j + poseId * jacPoseType::size();

                float jj = jacPose(i) * jacPose(j) * weight;
                m_H(poseParamId1, poseParamId2) += jj;
                m_H(poseParamId2, poseParamId1) += jj;
            }
        }

        for (int i = 0; i < jacPoseType::size(); i++)
        {
            int poseParamId = i + poseId * jacPoseType::size();

            for (int j = 0; j < jacMapType::size(); j++)
            {
                assert(m_numMapParams + m_numPoseParams > intMapIds(j));

                float value = jacPose(i) * jacMap(j) * weight;
                m_H(poseParamId, intMapIds(j)) += value;
                m_H(intMapIds(j), poseParamId) += value;
            }
        }

        for (int j = 0; j < jacMapType::size(); j++)
        {
            m_G(intMapIds(j)) += jacMap(j) * error * weight;
            m_H(intMapIds(j), intMapIds(j)) += jacMap(j) * jacMap(j) * weight;

            for (int k = j + 1; k < jacMapType::size(); k++)
            {
                float value = jacMap(j) * jacMap(k) * weight;

                m_H(intMapIds(j), intMapIds(k)) += value;
                m_H(intMapIds(k), intMapIds(j)) += value;
            }
        }
    }

    DenseLinearProblem operator+(DenseLinearProblem a)
    {
        assert(m_numPoseParams == a.m_numPoseParams && m_numMapParams == a.m_numMapParams);

        DenseLinearProblem sum(m_numPoseParams, m_numMapParams);

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
        assert(m_numPoseParams == a.m_numPoseParams && m_numMapParams == a.m_numMapParams);

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

        for (int i = 0; i < indicesToKeep.size(); i++)
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

    Eigen::VectorXf solve()
    {
        Eigen::VectorXf res = solver.solve(m_G);
        assert(solver.info() == Eigen::Success);
        return res;
    }

    Eigen::VectorXf ssolve(float lambda)
    {
        if (m_sH.cols() == 1 || m_sH.rows() == 1)
        {
            m_sH = toSparseMatrix(m_H);
        }

        Eigen::SparseMatrix<float> H = m_sH;

        for (int j = 0; j < m_G.size(); j++)
        {
            H.coeffRef(j, j) *= (1.0 + lambda);
        }

        // int numParams = m_numPoseParams + m_numMapParams;
        // Eigen::Matrix<float, numParams, numParams> H = m_H;
        // return m_H.llt().solve(m_G);
        // return m_H.ldlt().solve(m_G);
        // solver.compute(m_H);
        // return solver.solve(m_G);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> ssolver;

        ssolver.compute(H);
        // solver.analyzePattern(H_lambda);
        // solver.factorize(H_lambda);

        assert(ssolver.info() == Eigen::Success);

        Eigen::VectorXf inc = ssolver.solve(m_G);

        assert(ssolver.info() == Eigen::Success);

        return inc;
    }

    /*
    void operator=(HGPose _pose)
    {
        H = _pose.H;
        G = _pose.G;
    }
    */

    /*
    std::map<int, int> getObservedParamIds()
    {
        std::map<int, int> ids;
        for (int it = 0; it < m_G.size(); ++it)
        {
            if (m_G[it] != 0.0)
            {
                // ids.push_back(it);
                ids[it] = ids.size();
            }
        }

        return ids;
    }
    */

    /*
    Eigen::MatrixXf getHDense()
    {
        assert(m_count > 0);

        return m_H / m_count;
    }

    Eigen::VectorXf getG()
    {
        assert(m_count > 0);

        return m_G / m_count;
    }

    Eigen::VectorXf getG(std::map<int, int> &pIds)
    {
        assert(m_count > 0);

        Eigen::VectorXf _G(pIds.size());
        for (auto id : pIds)
        {
            int dst = id.second;
            int src = id.first;
            float val = m_G(src);
            _G(dst) = val;
        }

        //for (int id = 0; id < pIds.size(); id++)
        //{
        //    _G[id] = G[pIds[id]];
        //}

        return _G / m_count;
    }
    */

    Eigen::SparseMatrix<float> toSparseMatrix(Eigen::MatrixXf &D)
    {
        /*
        SparseMatrix<double> mat(rows, cols);
        for (int k = 0; k < mat.outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
            {
                it.value();
                it.row();   // row index
                it.col();   // col index (here it is equal to k)
                it.index(); // inner index, here it is equal to it.row()
            }
            */

        Eigen::SparseMatrix<float> S(D.rows(), D.cols());
        S.setZero();

        for (int y = 0; y < D.rows(); y++)
        {
            // for (Eigen::SparseMatrix<float>::InnerIterator it(m_H, src_col); it; ++it)
            for (int x = 0; x < D.cols(); x++)
            {
                // it.value();
                // it.row();   // row index
                // it.col();   // col index (here it is equal to pId[id])
                // it.index(); // inner index, here it is equal to it.row()

                float value = D(y, x);
                if (value == 0.0)
                    continue;

                S.insert(y, x) = value; // it.value();
            }
        }

        S.makeCompressed();

        return S;
    }

    int getCount()
    {
        return m_count;
    }

private:
    Eigen::MatrixXf m_H;
    Eigen::VectorXf m_G;

    Eigen::MatrixXf m_lH;

    Eigen::SparseMatrix<float> m_sH;

    int m_numPoseParams;
    int m_numMapParams;
    // Eigen::Matrix<float, 6, 6> H;
    // Eigen::Matrix<float, 6, 1> G;
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
