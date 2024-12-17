#ifndef KRYLOVSUBSPACE_HPP
#define KRYLOVSUBSPACE_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include "MatrixObj.hpp"
#include "VectorObj.hpp"

namespace Krylov {
    template<typename TNum>
    void Arnoldi(MatrixObj<TNum>& A, std::vector<VectorObj<TNum>>& Q, MatrixObj<TNum>& H, TNum tol) {
        size_t m = Q.size();
        size_t n = A.get_row();  // Assuming A is square, use rows as size
        assert(H.get_row() == m && H.get_col() == m - 1);
        assert(Q[0].size() == n);  // Ensure the vectors in Q are of the same size as the matrix rows

        // Iterate over the Krylov subspace dimensions
        for (size_t i = 1; i < m; ++i) {
            VectorObj<TNum> Av = A * Q[i - 1];  // A * q_(i-1)

            // Orthogonalization
            for (size_t j = 0; j < i; ++j) {
                TNum h = Q[j] * Av;  // Inner product
                H(j, i - 1) = h;
                Av = Av - Q[j] * h;  // Update Av to be orthogonal to Q[j]
            }

            // Check the norm to ensure it's above tolerance
            TNum h = Av.L2norm();
            if (h < tol) {
                std::cout << "Breakdown: Norm below tolerance at iteration " << i << "\n";
                break;  // Terminate if the norm is below the tolerance
            }

            H(i, i - 1) = h;
            Q[i] = Av / h;  // Normalize the vector
        }
    }
}

#endif // KRYLOVSUBSPACE_HPP
