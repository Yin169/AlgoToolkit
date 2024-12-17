#ifndef LU_HPP
#define LU_HPP

#include <vector>
#include <stdexcept> // For std::invalid_argument
#include <cmath>     // For std::abs
#include <algorithm> // For std::swap
#include "MatrixObj.hpp"
#include "VectorObj.hpp"

namespace LU {

    template <typename TNum>
    void PivotLU(MatrixObj<TNum>& A, std::vector<int>& P) {
        int n = A.getRows();

        if (n != A.getCols()) {
            // LU decomposition is only defined for square matrices.
            throw std::invalid_argument("Matrix must be square for LU decomposition.");
        }

        P.resize(n);
        // Initialize the permutation vector with row indices
        for (int i = 0; i < n; ++i) {
            P[i] = i;
        }

        for (int j = 0; j < n; ++j) {
            // Find the pivot row
            TNum maxVal = std::abs(A(j, j));
            int maxIndex = j;

            for (int i = j + 1; i < n; ++i) {
                if (std::abs(A(i, j)) > maxVal) {
                    maxVal = std::abs(A(i, j));
                    maxIndex = i;
                }
            }

            // Swap rows in A and update the permutation vector
            if (maxIndex != j) {
                A.swapRows(j, maxIndex);
                std::swap(P[j], P[maxIndex]);
            }

            // Perform the elimination
            for (int i = j + 1; i < n; ++i) {
                if (A(j, j) == static_cast<TNum>(0)) {
                    throw std::runtime_error("Matrix is singular and cannot be decomposed.");
                }

                TNum factor = A(i, j) / A(j, j);
                A(i, j) = factor; // Store the factor in place

                for (int k = j + 1; k < n; ++k) {
                    A(i, k) -= factor * A(j, k);
                }
            }
        }
    }

} // namespace LU

#endif // LU_HPP
