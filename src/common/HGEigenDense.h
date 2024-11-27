#pragma once

#include <Eigen/Core>

class HGEigenDense
{
public:
    HGEigenDense(int numPoseParams, int numMapParams)
    {
        m_numPoseParams = numPoseParams;
        m_numMapParams = numMapParams;

        m_H = Eigen::MatrixXf::Zero(numPoseParams + numMapParams, numPoseParams + numMapParams);
        m_G = Eigen::VectorXf::Zero(numPoseParams + numMapParams);

        m_count = 0;
    }

    void setZero()
    {
        m_H.setZero();
        m_G.setZero();
        m_count = 0;
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
    void add(jacType J, float error, float weight, idsType ids)
    {
        assert(jacType::size() == idsType::size());
        assert(jacType::size() <= m_numPoseParams + m_numMapParams);

        m_count++;

        idsType inIds = ids + idsType(m_numPoseParams);

        // G += J * error;
        // H += J * J.transpose();

        for (int i = 0; i < jacType::size(); i++)
        {
            m_G(inIds(i)) += J(i) * error * weight;
            m_H(inIds(i), inIds(i)) += J(i) * J(i) * weight;

            for (int j = i + 1; j < jacType::size(); j++)
            {
                float jj = J(i) * J(j) * weight;
                m_H(inIds(i), inIds(j)) += jj;
                m_H(inIds(j), inIds(i)) += jj;
            }
        }
    }

    template <typename jacPoseType, typename jacMapType, typename idsType>
    void add(jacPoseType jacPose, jacMapType jacMap, float error, float weight, int poseId, idsType mapIds)
    {
        assert(jacMapType::size() == idsType::size());
        assert(m_numPoseParams > jacPoseType::size() - 1 + poseId*jacPoseType::size());

        m_count++;

        idsType intMapIds = mapIds + idsType(m_numPoseParams);

        for (int i = 0; i < jacPoseType::size(); i++)
        {
            int poseParamId1 = i + poseId*jacPoseType::size();

            m_G(poseParamId1) += jacPose(i) * error * weight;
            m_H(poseParamId1, poseParamId1) += jacPose(i) * jacPose(i) * weight;

            for (int j = i + 1; j < jacPoseType::size(); j++)
            {
                int poseParamId2 = j + poseId*jacPoseType::size();

                float jj = jacPose(i) * jacPose(j) * weight;
                m_H(poseParamId1, poseParamId2) += jj;
                m_H(poseParamId2, poseParamId1) += jj;
            }
        }

        for (int i = 0; i < jacPoseType::size(); i++)
        {
            int poseParamId = i + poseId*jacPoseType::size();

            for(int j = 0; j < jacMapType::size(); j++)
            {
                assert(m_numMapParams + m_numPoseParams >  intMapIds(j));

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

    HGEigenDense operator+(HGEigenDense a)
    {
        assert(m_numPoseParams == a.m_numPoseParams && m_numMapParams == a.m_numMapParams);

        HGEigenDense sum(m_numPoseParams, m_numMapParams);
        sum.m_H = m_H + a.m_H;
        sum.m_G = m_G + a.m_G;
        sum.m_count = m_count + a.m_count;
        sum.m_numPoseParams = m_numPoseParams;
        sum.m_numMapParams = m_numMapParams;
        return sum;
    }

    void operator+=(HGEigenDense a)
    {
        assert(m_numPoseParams == a.m_numPoseParams && m_numMapParams == a.m_numMapParams);

        m_H += a.m_H;
        m_G += a.m_G;
        m_count += a.m_count;
    }

    /*
    void operator=(HGPose _pose)
    {
        H = _pose.H;
        G = _pose.G;
    }
    */

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
        /*
        for (int id = 0; id < pIds.size(); id++)
        {
            _G[id] = G[pIds[id]];
        }
        */
        return _G / m_count;
    }

    Eigen::SparseMatrix<float> getHSparse(std::map<int, int> &pIds)
    {
        assert(m_count > 0);

        Eigen::SparseMatrix<float> _H(pIds.size(), pIds.size());

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

        for (auto id : pIds)
        {
            int dst_col = id.second;
            int src_col = id.first;
            //for (Eigen::SparseMatrix<float>::InnerIterator it(m_H, src_col); it; ++it)
            for(int it = 0; it < m_G.size(); it++)
            {
                // it.value();
                // it.row();   // row index
                // it.col();   // col index (here it is equal to pId[id])
                // it.index(); // inner index, here it is equal to it.row()

                int src_row = it;
                if (!pIds.count(src_row))
                    continue;

                float value = m_H(src_row, src_col);
                if(value == 0.0)
                    continue;
                
                int dst_row = pIds[src_row];

                _H.insert(dst_row, dst_col) = value;// it.value();
            }
        }

        _H.makeCompressed();

        _H /= m_count;
        return _H;
    }

private:
    Eigen::MatrixXf m_H;
    Eigen::VectorXf m_G;
    int m_numPoseParams;
    int m_numMapParams;
    // Eigen::Matrix<float, 6, 6> H;
    // Eigen::Matrix<float, 6, 1> G;
    int m_count;
};
