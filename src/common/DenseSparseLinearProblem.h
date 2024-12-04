#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>

class DenseSparseLinearProblem
{
public:
    DenseSparseLinearProblem(int numPoseParams, int numMapParams)
    {
        m_numPoseParams = numPoseParams;
        m_numMapParams = numMapParams;

        m_HPose = Eigen::MatrixXf::Zero(numPoseParams, numPoseParams);
        m_GPose = Eigen::VectorXf::Zero(numPoseParams);

        m_HPoseMap = Eigen::MatrixXf::Zero(numPoseParams, numMapParams);

        m_HMap = Eigen::SparseMatrix<float>(numMapParams, numMapParams);
        m_GMap = Eigen::VectorXf::Zero(numMapParams);
        
        m_count = 0;

        // reserve space, will need roughly
        // for each pixel
        //a 11*11 matrix
        //H.reserve(Eigen::VectorXi::Constant(size,3));
        tripletList.reserve(256*256*3*3);
    }

    void setZero()
    {
        m_HPose.setZero();
        m_GPose.setZero();
        m_HPoseMap.setZero();
        m_HMap.setZero();
        m_GMap.setZero();
        m_count = 0;
    }

    DenseSparseLinearProblem operator+(DenseSparseLinearProblem &a)
    {
        setFromTriplets();
        a.setFromTriplets();

        DenseSparseLinearProblem sum(m_numPoseParams, m_numMapParams);
        sum.m_HPose = m_HPose + a.m_HPose;
        sum.m_GPose = m_GPose + a.m_GPose;
        sum.m_HPoseMap = m_HPoseMap + a.m_HPoseMap;
        sum.m_HMap = m_HMap + a.m_HMap;
        sum.m_GMap = m_GMap + a.m_GMap;
        sum.m_count = m_count + a.m_count;
        sum.m_numPoseParams = m_numPoseParams;
        sum.m_numMapParams = m_numMapParams;
        return sum;
    }

    void operator+=(DenseSparseLinearProblem &a)
    {
        setFromTriplets();
        a.setFromTriplets();

        m_HPose += a.m_HPose;
        m_GPose += a.m_GPose;
        m_HPoseMap += a.m_HPoseMap;
        m_HMap += a.m_HMap;
        m_GMap += a.m_GMap;
        m_count += a.m_count;
    }

    /*
    void add(float jac, float error, float weight, int ids)
    {
        count++;

        G(ids) += jac * error * weight;
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

    template <typename jacType, typename idsType>
    void add(vec3<float> jacMap, float error, float weight, vec3<int> mapIds)
    {
        m_count++;

        for (int j = 0; j < 3; j++)
        {
            m_GMap(mapIds(j)) += jacMap(j) * error * weight;
            tripletList.push_back(T(mapIds(j), mapIds(j), jacMap(j) * jacMap(j) * weight));
            //H.coeffRef(ids(j), ids(j)) += jac(j) * jac(j) * weight;

            for (int k = j + 1; k < 3; k++)
            {
                float value = jacMap(j) * jacMap(k) * weight;
                tripletList.push_back(T(mapIds(j), mapIds(k), value));
                tripletList.push_back(T(mapIds(k), mapIds(j), value));
                //H.coeffRef(ids(j), ids(k)) += value;
                //H.coeffRef(ids(k), ids(j)) += value;
            }
        }
    }

    void add(vec8<float> jacPose, vec3<float> jacMap, float error, float weight, int poseId, vec3<int> mapIds)
    {
        count++;

        for (int i = 0; i < 8; i++)
        {
            GPose(i + poseId*8) += jacPose(i) * error * weight;
            HPose(i + poseId*8, i + poseId*8) += jacPose(i) * jacPose(i) * weight;

            for (int j = i + 1; j < 8; j++)
            {
                float jj = jacPose(i) * jacPose(j) * weight;
                HPose(i + poseId*8, j + poseId*8) += jj;
                HPose(j + poseId*8, i + poseId*8) += jj;
            }
        }

        for (int i = 0; i < 8; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                HPoseMap(i + poseId*8, mapIds(j)) += jacPose(i) * jacMap(j) * weight;
            }
        }

        for (int j = 0; j < 3; j++)
        {
            GMap(mapIds(j)) += jacMap(j) * error * weight;
            tripletList.push_back(T(mapIds(j), mapIds(j), jacMap(j) * jacMap(j) * weight));
            //H.coeffRef(ids(j), ids(j)) += jac(j) * jac(j) * weight;

            for (int k = j + 1; k < 3; k++)
            {
                float value = jacMap(j) * jacMap(k) * weight;
                tripletList.push_back(T(mapIds(j), mapIds(k), value));
                tripletList.push_back(T(mapIds(k), mapIds(j), value));
                //H.coeffRef(ids(j), ids(k)) += value;
                //H.coeffRef(ids(k), ids(j)) += value;
            }
        }
    }

    /*
    void add(vecx<size, float> jac, float error, float weight, vecx<size, int> ids)
    {
        count++;

        for (int j = 0; j < size; j++)
        {
            G(ids(j)) += jac(j) * error * weight;
            tripletList.push_back(T(ids(j), ids(j), jac(j) * jac(j) * weight));
            //H.coeffRef(ids(j), ids(j)) += jac(j) * jac(j) * weight;

            for (int k = j + 1; k < size; k++)
            {
                float value = jac(j) * jac(k) * weight;
                tripletList.push_back(T(ids(j), ids(k), value));
                tripletList.push_back(T(ids(k), ids(j), value));
                //H.coeffRef(ids(j), ids(k)) += value;
                //H.coeffRef(ids(k), ids(j)) += value;
            }
        }
    }
    */

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
        for (int it = 0; it < GMap.size(); ++it)
        {
            // ids.push_back(it);
            ids[it] = ids.size();
        }
        return ids;
    }

    std::map<int, int> getObservedParamIds()
    {
        std::map<int, int> ids;
        for (int it = 0; it < GMap.size(); ++it)
        {
            if (GMap[it] != 0.0)
            {
                // ids.push_back(it);
                ids[it] = ids.size();
            }
        }
        return ids;
    }

    Eigen::VectorXf getG(std::map<int, int> &pIds)
    {
        Eigen::VectorXf _G(pIds.size());
        for (auto id : pIds)
        {
            int dst = id.second;
            int src = id.first;
            float val = GMap(src);
            _G(dst) = val;
        }
        /*
        for (int id = 0; id < pIds.size(); id++)
        {
            _G[id] = G[pIds[id]];
        }
        */
        return _G / count;
    }

    Eigen::SparseMatrix<float> getH(std::map<int, int> &pIds)
    {
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
            for (Eigen::SparseMatrix<float>::InnerIterator it(HMap, src_col); it; ++it)
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

        _H /= count;
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
            m_HMap.setFromTriplets(tripletList.begin(), tripletList.end());
            tripletList.clear();
        }
    }

    Eigen::SparseMatrix<float> m_HMap;
    Eigen::VectorXf m_GMap;

    Eigen::MatrixXf m_HPose;
    Eigen::VectorXf m_GPose;

    Eigen::MatrixXf m_HPoseMap;

    int m_count;
    int m_numPoseParams;
    int m_numMapParams;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
    //Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
    //   Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;

    // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;

    // Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<float>, Eigen::Lower> solver;
    // Eigen::SPQR<Eigen::SparseMatrix<float>> solver;
};
