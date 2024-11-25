#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>

class HGEigenDenseSparse
{
public:
    HGEigenDenseSparse(int numFrames, int numMapParams)
    {
        HPose = Eigen::MatrixXf::Zero(numFrames*8, numFrames*8);
        GPose = Eigen::VectorXf::Zero(numFrames*8);

        HGPoseMap = Eigen::MatrixXf::Zero(numFrames*8, numMapParams)

        HMap = Eigen::SparseMatrix<float>(numMapParams, numMapParams);
        GMap = Eigen::VectorXf::Zero(numMapParams);
        
        count = 0;

        // reserve space, will need roughly
        // for each pixel
        //a 11*11 matrix
        //H.reserve(Eigen::VectorXi::Constant(size,3));
        tripletList.reserve(256*256*3*3);
    }

    void setZero()
    {
        HPose.setZero();
        GPose.setZero();
        HGPoseMap.setZero();
        HMap.setZero();
        GMap.setZero();
        count = 0;
    }

    HGEigenDenseSparse operator+(HGEigenDenseSparse &a)
    {
        HGEigenDenseSparse sum(G.size());
        sum.HPose = HPose + a.HPose;
        sum.GPose = GPose + a.GPose;
        sum.HPoseMap = HPoseMap + a.HPoseMap;
        sum.GPoseMap = GPoseMap + a.GPoseMap;
        sum.HMap = HMap + a.HMap;
        sum.GMap = GMap + a.GMap;
        sum.count = count + a.count;
        return sum;
    }

    void operator+=(HGEigenDenseSparse &a)
    {
        HPose += a.HPose;
        GPose += a.GPose;
        HPoseMap += a.HPoseMap;
        GPoseMap += a.GPoseMap;
        HMap += a.HMap;
        GMap += a.GMap;
        count += a.count;
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

    void add(vec3<float> jacMap, float error, float weight, vec3<int> mapIds)
    {
        count++;

        for (int j = 0; j < 3; j++)
        {
            G(mapIds(j)) += jacMap(j) * error * weight;
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

    void add(vec8<float> jacPose, vec3<float> jacMap, float error, float weight, vec8<int> poseIds, vec3<int> mapIds)
    {
        count++;

        for (int i = 0; i < 8; i++)
        {
            GPose(i) += jacPose(i) * error * weight;
            HPose(i, i) += jacPose(i) * jacPose(i) * weight;

            for (int j = i + 1; j < 8; j++)
            {
                float jj = jacPose(i) * jacPose(j) * weight;
                HPose(i, j) += jj;
                HPose(j, i) += jj;
            }
        }

        for (int i = 0; i < 8; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                HPoseMap(i, mapIds(j)) += jacPose(i) * jacMap(j) * weight;
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

    void endAdd()
    {
        HMap.setFromTriplets(tripletList.begin(), tripletList.end());
        tripletList.clear();
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
        for (int it = 0; it < G.size(); ++it)
        {
            // ids.push_back(it);
            ids[it] = ids.size();
        }
        return ids;
    }

    std::map<int, int> getObservedParamIds()
    {
        std::map<int, int> ids;
        for (int it = 0; it < G.size(); ++it)
        {
            if (G[it] != 0.0)
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
            float val = G(src);
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
            for (Eigen::SparseMatrix<float>::InnerIterator it(H, src_col); it; ++it)
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
    Eigen::SparseMatrix<float> HMap;
    Eigen::VectorXf GMap;

    Eigen::MatrixXf HPose;
    Eigen::VectorXf GPose;

    Eigen::MatrixXf HPoseMap;

    int count;

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
};
