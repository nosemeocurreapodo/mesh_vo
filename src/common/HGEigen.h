#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

class HGEigen
{
public:
    HGEigen(int size)
    {
        H = Eigen::SparseMatrix<float>(size, size);
        G = Eigen::VectorXf::Zero(size);
        count = 0;
    }

    void setZero()
    {
        H.setZero();
        G.setZero();
        count = 0;
    }

    HGEigen operator+(HGEigen a)
    {
        HGEigen sum(G.size());
        sum.H = H + a.H;
        sum.G = G + a.G;
        sum.count = count + a.count;
        return sum;
    }

    void operator+=(HGEigen &a)
    {
        H += a.H;
        G += a.G;
        count += a.count;
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
            _G[id.second] = G[id.first];
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

    Eigen::SparseMatrix<float> H;
    Eigen::VectorXf G;
    int count;

private:
};
