#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>

class HGEigenSparse
{
public:
    HGEigenSparse(int numPoseParams, int numMapParams)
    {
        m_numPoseParams = numPoseParams;
        m_numMapParams = numMapParams;
        m_H = Eigen::SparseMatrix<float>(numPoseParams + numMapParams, numPoseParams + numMapParams);
        m_G = Eigen::VectorXf::Zero(numPoseParams + numMapParams);
        m_count = 0;

        // reserve space, will need roughly
        // for each pixel
        //a 11*11 matrix
        //H.reserve(Eigen::VectorXi::Constant(size,3));
        tripletList.reserve(256*256*(numPoseParams + numMapParams)*(numPoseParams + numMapParams));
    }

    void setZero()
    {
        m_H.setZero();
        m_G.setZero();
        m_count = 0;
        m_numPoseParams = 0;
        m_numMapParams = 0;
    }

    HGEigenSparse operator+(HGEigenSparse &a)
    {
        assert(m_numPoseParams == a.m_numPoseParams && m_numMapParams == a.m_numMapParams);

        setFromTriplets();
        a.setFromTriplets();

        HGEigenSparse sum(m_numPoseParams, m_numMapParams);
        sum.m_H = m_H + a.m_H;
        sum.m_G = m_G + a.m_G;
        sum.m_count = m_count + a.m_count;
        sum.m_numPoseParams = m_numPoseParams;
        sum.m_numMapParams = m_numMapParams;
        return sum;
    }

    void operator+=(HGEigenSparse &a)
    {
        assert(m_numPoseParams == a.m_numPoseParams && m_numMapParams == a.m_numMapParams);

        setFromTriplets();
        a.setFromTriplets();

        m_H += a.m_H;
        m_G += a.m_G;
        m_count += a.m_count;
    }

    /*
    void add(float jac, float error, float weight, int ids)
    {
        m_count++;

        m_G(ids) += jac * error * weight;
        tripletList.push_back(T(ids, ids, jac * jac * weight));
        //H.coeffRef(ids, ids) += jac * jac * weight;
    }
    */

    /*
    void add(vec1<float> jac, float error, float weight, vec1<int> ids)
    {
        count++;

        G(ids(0)) += jac(0) * error * weight;
        tripletList.push_back(T(ids(0), ids(0), jac(0) * jac(0) * weight));
    }
    */

    template <typename jacType>
    void add(jacType jac, float error, float weight)
    {
        assert(jacType::size() <= m_numPoseParams + m_numMapParams);

        m_count++;

        for (int j = 0; j < jacType::size(); j++)
        {
            m_G(j) += jac(j) * error * weight;
            tripletList.push_back(T(j, j, jac(j) * jac(j) * weight));
            //H.coeffRef(ids(j), ids(j)) += jac(j) * jac(j) * weight;

            for (int k = j + 1; k < 3; k++)
            {
                float value = jac(j) * jac(k) * weight;
                tripletList.push_back(T(j, k, value));
                tripletList.push_back(T(k, j, value));
                //H.coeffRef(ids(j), ids(k)) += value;
                //H.coeffRef(ids(k), ids(j)) += value;
            }
        }
    }

    template <typename jacType, typename idsType>
    void add(jacType jac, float error, float weight, idsType ids)
    {
        assert(jacType::size() == idsType::size());
        assert(jacType::size() <= m_numPoseParams + m_numMapParams);

        m_count++;

        for (int j = 0; j < jacType::size(); j++)
        {
            m_G(ids(j)) += jac(j) * error * weight;
            tripletList.push_back(T(ids(j), ids(j), jac(j) * jac(j) * weight));
            //H.coeffRef(ids(j), ids(j)) += jac(j) * jac(j) * weight;

            for (int k = j + 1; k < jacType::size(); k++)
            {
                float value = jac(j) * jac(k) * weight;
                tripletList.push_back(T(ids(j), ids(k), value));
                tripletList.push_back(T(ids(k), ids(j), value));
                //H.coeffRef(ids(j), ids(k)) += value;
                //H.coeffRef(ids(k), ids(j)) += value;
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
            //H.coeffRef(i + poseId*8, i + poseId*8) += jac(j) * jac(j) * weight;
            tripletList.push_back(T(poseParamId1, poseParamId1, jacPose(i) * jacPose(i) * weight));

            for (int j = i + 1; j < poseId*jacPoseType::size(); j++)
            {
                int poseParamId2 = j + poseId*jacPoseType::size();

                float jj = jacPose(i) * jacPose(j) * weight;

                tripletList.push_back(T(poseParamId1, poseParamId2, jj));
                tripletList.push_back(T(poseParamId2, poseParamId1, jj));
            }
        }

        for (int i = 0; i < jacPoseType::size(); i++)
        {
            int poseParamId = i + poseId*jacPoseType::size();
            for(int j = 0; j < jacMapType::size(); j++)
            {
                //m_H(poseParamId, intMapIds(j)) += jacPose(i) * jacMap(j) * weight;
                tripletList.push_back(T(poseParamId, intMapIds(j), jacPose(i) * jacMap(j) * weight));
            }
        }

        for (int j = 0; j < jacMapType::size(); j++)
        {
            m_G(intMapIds(j)) += jacMap(j) * error * weight;
            tripletList.push_back(T(intMapIds(j), intMapIds(j), jacMap(j) * jacMap(j) * weight));
            //H.coeffRef(ids(j), ids(j)) += jac(j) * jac(j) * weight;

            for (int k = j + 1; k < jacMapType::size(); k++)
            {
                float value = jacMap(j) * jacMap(k) * weight;
                tripletList.push_back(T(intMapIds(j), intMapIds(k), value));
                tripletList.push_back(T(intMapIds(k), intMapIds(j), value));
                //H.coeffRef(ids(j), ids(k)) += value;
                //H.coeffRef(ids(k), ids(j)) += value;
            }
        }
    }

    /*
    std::vector<int> getParamIds()
    {
        std::vector<int> ids;
        for (int it = 0; it < G.size(); ++it)
        {
            if (G[it] != 0.0)
                ids.push_back(it);
        }
        return ids;
    }
    */

    std::map<int, int> getParamIds()
    {
        std::map<int, int> ids;
        for (int it = 0; it < m_G.size(); ++it)
        {
            // ids.push_back(it);
            ids[it] = ids.size();
        }
        return ids;
    }

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

        setFromTriplets();

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
            for (Eigen::SparseMatrix<float>::InnerIterator it(m_H, src_col); it; ++it)
            {
                // it.value();
                // it.row();   // row index
                // it.col();   // col index (here it is equal to pId[id])
                // it.index(); // inner index, here it is equal to it.row()

                int src_row = it.row();
                if (!pIds.count(src_row))
                    continue;
                int dst_row = pIds[src_row];

                _H.insert(dst_row, dst_col) = it.value();
            }
        }

        _H.makeCompressed();

        _H /= m_count;
        return _H;
    }

    /*
    void operator=(HGPose _pose)
    {
        H = _pose.H;
        G = _pose.G;
    }
    */

private:

    void setFromTriplets()
    {
        if(tripletList.size() > 0)
        {
            m_H.setFromTriplets(tripletList.begin(), tripletList.end());
            tripletList.clear();
        }
    }

    Eigen::SparseMatrix<float> m_H;
    Eigen::VectorXf m_G;
    int m_numPoseParams;
    int m_numMapParams;
    int m_count;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
};
