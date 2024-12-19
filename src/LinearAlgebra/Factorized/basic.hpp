#ifndef BASIC_HPP
#define BASIC_HPP

#include <iostream>
#include <vector>
#include <cstring> // For memset
#include "DenseObj.hpp"
#include "VectorObj.hpp"

namespace basic {

    // Power Iteration for finding the dominant eigenvector
    template <typename TNum, typename MatrixType>
    void powerIter(MatrixType& A, VectorObj<TNum>& b, int max_iter_num) {
        if (max_iter_num <= 0) return;

        for (int i = 0; i < max_iter_num; ++i) {
            b.normalize(); // Ensure the vector remains normalized
            b = A * b;     // Compute A * b
        }
        b.normalize(); // Final normalization for stability
    }

    // Rayleigh Quotient for estimating the dominant eigenvalue
    template <typename TNum, typename MatrixType>
    double rayleighQuotient(const MatrixType& A, const VectorObj<TNum>& b) {
        if (b.L2norm() == 0) return 0;

        VectorObj<TNum> Ab = A * b;
        TNum dot_product = b * Ab;
        return dot_product / (b * b);
    }

    // Subtract projection of u onto v
    template <typename TNum>
    VectorObj<TNum> subtProj(const VectorObj<TNum>& u, const VectorObj<TNum>& v) {
        if (v.L2norm() == 0) return u;

        TNum factor = (u * v) / (v * v);
        VectorObj<TNum> result = u - (v * factor);
        return result;
    }

    // Gram-Schmidt process for orthogonalization
    template <typename TNum, typename MatrixType>
    void gramSchmidt(const MatrixType& A, std::vector<VectorObj<TNum>>& orthSet) {
        int m = A.getCols();
        orthSet.resize(m);

        for (int i = 0; i < m; ++i) {
            VectorObj<TNum> v = A.getColumn(i);
            for (int j = 0; j < i; ++j) {
                v = subtProj(v, orthSet[j]);
            }
            v.normalize();
            orthSet[i] = v;
        }
    }

    // Generate a unit vector at position i
    template <typename TNum>
    VectorObj<TNum> genUnitVec(int i, int n) {
        VectorObj<TNum> unitVec(n);
        unitVec[i] = static_cast<TNum>(1);
        return unitVec;
    }

    // Generate an identity matrix
    template <typename TNum, typename MatrixType>
    MatrixType genUnitMat(int n) {
        MatrixType identity(n, n);
        for (int i = 0; i < n; ++i) {
            identity(i, i) = static_cast<TNum>(1);
        }
        return identity;
    }

    // Sign function
    template <typename TNum>
    TNum sign(TNum x) {
        return (x >= 0) ? static_cast<TNum>(1) : static_cast<TNum>(-1);
    }

    // Generate a Householder transformation matrix
    template <typename TNum, typename MatrixType>
    void householderTransform(const MatrixType& A, MatrixType& H, int index) {
        int n = A.getRows();
        VectorObj<TNum> x = A.getColumn(index);

        VectorObj<TNum> e = genUnitVec<TNum>(index, n);
        TNum factor = x.L2norm() * sign(x[index]);
        e *= factor;

        VectorObj<TNum> v = x + e;
        v.normalize(); // Normalize for numerical stability

        MatrixType vMat(v, n, 1);
        H = genUnitMat<TNum, MatrixType>(n) - (vMat * vMat.Transpose()) * static_cast<TNum>(2);
    }

    // QR factorization using Householder reflections
    template <typename TNum, typename MatrixType>
    void qrFactorization(const MatrixType& A, MatrixType& Q, MatrixType& R) {
        int n = A.getRows(), m = A.getCols();
        R = A;
        Q = genUnitMat<TNum, MatrixType>(n);

        for (int i = 0; i < std::min(n, m); ++i) {
            MatrixType H;
            householderTransform<TNum, MatrixType>(R, H, i);
            R = H * R;
            Q = Q * H.Transpose();
        }
    }

    // Perform forward or backward substitution
    template <typename TNum, typename MatrixType>
    VectorObj<TNum> Substitution(const VectorObj<TNum>& b, const MatrixType& L, bool forward) {
        const int n = b.size();
        VectorObj<TNum> x(n);

        for (int i = forward ? 0 : n - 1; forward ? i < n : i >= 0; forward ? ++i : --i) {
            TNum sum = TNum(0);

            for (int j = forward ? 0 : n - 1; forward ? j < i : j > i; forward ? ++j : --j) {
                sum += L(i, j) * x[j]; // Using more readable row-major accessor
            }

            if (L(i, i) == static_cast<TNum>(0)) {
                throw std::runtime_error("Division by zero during substitution.");
            }

            x[i] = (b[i] - sum) / L(i, i);
        }

        return x;
    }

} // namespace basic

#endif // BASIC_HPP
