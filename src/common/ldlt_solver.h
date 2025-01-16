#pragma once

#include <cmath>
#include "common/lingalglib.h"

template <typename matType>
class LDLT
{
    // Perform LDL^T decomposition on an n x n symmetric matrix A.
    // L will be a lower-triangular matrix with 1.0 on the diagonal.
    // D will be a diagonal vector of length n.

public:
    LDLT()
    {
    }

    void compute(matType _A)
    {
        A = _A;
    }

    // Solve A x = b given A is symmetric, using LDL^T factorization
    vecType solve(vecType b)
    {
        // 1) Factor A = L D L^T
        matType L;
        vecType D;
        ldlt_decompose(A, L, D);

        // 2) Solve L y = b
        vecType y;
        forward_substitution(L, b, y);

        // 3) Solve D z = y
        vecType z;
        diagonal_solve(D, y, z);

        // 4) Solve L^T x = z
        vecType x;
        back_substitution_transpose(L, z, x);

        return x;
    }

private:
    void ldlt_decompose(matType A, matType &L, vecType &D)
    {
        // Initialize L to identity and D to zero.
        L = matType::identity();
        D.setZero();

        for (int i = 0; i < A.rows(); ++i)
        {
            // 1) Compute D[i] = A[i][i] - sum_{k=0 to i-1}(L[i][k]*D[k]*L[i][k])
            double sum = A(i, i);
            for (int k = 0; k < i; ++k)
            {
                sum -= L(i, k) * L(i, k) * D(k);
            }
            D(i) = sum; // D is the diagonal

            // 2) Compute L[j][i] for j = i+1..n-1
            for (int j = i + 1; j < _cols; ++j)
            {
                double val = A(j, i);
                // Subtract the part contributed by previous columns
                for (int k = 0; k < i; ++k)
                {
                    val -= L(j, k) * L(i, k) * D(k);
                }
                // L[j][i] = (A[j][i] - ...) / D[i]
                L(j, i) = val / D(i);
            }
        }
    }

    // Forward substitution for L y = b
    // L is lower triangular (diagonal = 1.0)
    void forward_substitution(const double L[_rows][_cols],
                              const double b[_rows],
                              double y[_rows])
    {
        for (int i = 0; i < _rows; ++i)
        {
            double sum = b[i];
            for (int j = 0; j < _cols; ++j)
            {
                sum -= L[i][j] * y[j];
            }
            // L[i][i] should be 1 in LDL^T factorization
            y[i] = sum / L[i][i];
        }
    }

    // Diagonal solve for D z = y
    // D is diagonal, so z[i] = y[i] / D[i]
    vecType diagonal_solve(vecType D, vecType y)
    {
        vecType z;
        for (int i = 0; i < D.rows(); ++i)
        {
            if (std::fabs(D[i]) < 1e-14)
            {
                throw std::runtime_error("Encountered near-zero diagonal in D.");
            }
            z[i] = y[i] / D[i];
        }
    }

    // Back substitution for L^T x = z
    // L is lower-triangular, so L^T is upper-triangular.
    vecType back_substitution_transpose(matType L, vecType z)
    {
        vecType x;
        for (int i = _rows - 1; i >= 0; --i)
        {
            double sum = z(i);
            for (int j = i + 1; j < _rows; ++j)
            {
                sum -= L(j, i) * x(j); // L^T[i][j] = L[j][i]
            }
            // L[i][i] = 1.0
            x(i) = sum / L(i, i);
        }
        return x;
    }

    // A x = b
    matType A;
};
