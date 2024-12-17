#ifndef BASIC_HPP
#define BASIC_HPP

#include <iostream>
#include <vector>
#include <cstring> // For memset
#include "MatrixObj.hpp"
#include "VectorObj.hpp"

namespace basic {

    // Power Iteration for finding the dominant eigenvector
    template <typename TNum>
    void powerIter(MatrixObj<TNum>& A, VectorObj<TNum>& b, int max_iter_num) {
        if (max_iter_num <= 0) return;

        for (int i = 0; i < max_iter_num; ++i) {
            b.normalize(); // Ensure the vector remains normalized
            b = A * b;     // Compute A * b
        }
        b.normalize(); // Final normalization for stability
    }

    // Rayleigh Quotient for estimating the dominant eigenvalue
    template <typename TNum>
    double rayleighQuotient(const MatrixObj<TNum>& A, const VectorObj<TNum>& b) {
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
    template <typename TNum>
    void gramSchmidt(const MatrixObj<TNum>& A, std::vector<VectorObj<TNum>>& orthSet) {
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
    template <typename TNum>
    MatrixObj<TNum> genUnitMat(int n) {
        MatrixObj<TNum> identity(n, n);
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
    template <typename TNum>
    void householderTransform(const MatrixObj<TNum>& A, MatrixObj<TNum>& H, int index) {
        int n = A.getRows();
        VectorObj<TNum> x = A.getColumn(index);

        VectorObj<TNum> e = genUnitVec<TNum>(index, n);
        TNum factor = x.L2norm() * sign(x[index]);
        e *= factor;

        VectorObj<TNum> v = x + e;
        v.normalize(); // Normalize for numerical stability

        MatrixObj<TNum> vMat(v, n, 1);
        H = genUnitMat<TNum>(n) - (vMat * vMat.Transpose()) * static_cast<TNum>(2);
    }

    // QR factorization using Householder reflections
    template <typename TNum>
    void qrFactorization(const MatrixObj<TNum>& A, MatrixObj<TNum>& Q, MatrixObj<TNum>& R) {
        int n = A.getRows(), m = A.getCols();
        R = A;
        Q = genUnitMat<TNum>(n);

        for (int i = 0; i < std::min(n, m); ++i) {
            MatrixObj<TNum> H;
            householderTransform(R, H, i);
            R = H * R;
            Q = Q * H.Transpose();
        }
    }

} // namespace basic

#endif // BASIC_HPP
