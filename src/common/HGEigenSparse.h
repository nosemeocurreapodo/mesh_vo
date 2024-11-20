#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>

class HGEigenSparse
{
public:
    HGEigenSparse(int size)
    {
        H = Eigen::SparseMatrix<float>(size, size);
        G = Eigen::VectorXf::Zero(size);
        count = 0;

        // reserve space, will need roughly
        // 3 entryes for every map parameter
        // every pose parameter will be related with every map parameter
        tripletList.reserve(size * 3 + 5 * 6 * size);
    }

    void setZero()
    {
        H.setZero();
        G.setZero();
        count = 0;
    }

    HGEigenSparse operator+(HGEigenSparse a)
    {
        HGEigenSparse sum(G.size());
        sum.H = H + a.H;
        sum.G = G + a.G;
        sum.count = count + a.count;
        return sum;
    }

    void operator+=(HGEigenSparse &a)
    {
        H += a.H;
        G += a.G;
        count += a.count;
    }

    void sparseAdd(float jac, float error, float weight, int ids)
    {
        count++;

        G(ids) += jac * error * weight;
        tripletList.push_back(T(ids, ids, jac * jac * weight));
    }

    void sparseAdd(vec1<float> jac, float error, float weight, vec1<int> ids)
    {
        count++;

        G(ids(0)) += jac(0) * error * weight;
        tripletList.push_back(T(ids(0), ids(0), jac(0) * jac(0) * weight));
    }

    template <typename type>
    void sparseAdd(vec3<float> jac, float error, float weight, vec3<int> ids)
    {
        count++;

        for (int j = 0; j < ids.size(); j++)
        {
            G(ids(j)) += jac(j) * error * weight;
            tripletList.push_back(T(ids(j), ids(j), jac(j) * jac(j) * weight));

            for (int k = j + 1; k < ids.size(); k++)
            {
                float value = jac(j) * jac(k) * weight;
                tripletList.push_back(T(ids(j), ids(k), value));
                tripletList.push_back(T(ids(k), ids(j), value));
                // hg.H.coeffRef(v_ids[j],v_ids[k]) += (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
            }
        }
    }

    void sparseAdd(vecx<float> jac, float error, float weight, vecx<int> ids)
    {
        count++;

        for (int j = 0; j < ids.size(); j++)
        {
            G(ids(j)) += jac(j) * error * weight;
            tripletList.push_back(T(ids(j), ids(j), jac(j) * jac(j) * weight));

            for (int k = j + 1; k < ids.size(); k++)
            {
                float value = jac(j) * jac(k) * weight;
                tripletList.push_back(T(ids(j), ids(k), value));
                tripletList.push_back(T(ids(k), ids(j), value));
                // hg.H.coeffRef(v_ids[j],v_ids[k]) += (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
            }
        }
    }

    void endSparseAdd()
    {
        H.setFromTriplets(tripletList.begin(), tripletList.end());
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
    Eigen::SparseMatrix<float> H;
    Eigen::VectorXf G;
    int count;

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
};
