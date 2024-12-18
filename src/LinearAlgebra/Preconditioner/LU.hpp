#ifndef LU_HPP
#define LU_HPP

#include <vector>
#include <stdexcept> // For std::invalid_argument
#include <cmath>     // For std::abs
#include <algorithm> // For std::swap

namespace LU {

    template <typename TNum, typename MatrixType>
    void PivotLU(MatrixType& A, std::vector<int>& P) {
        int n = A.getRows();

        if (n != A.getCols()) {
            throw std::invalid_argument("Matrix must be square for LU decomposition.");
        }

        P.resize(n);

        // Initialize the permutation vector with row indices
        for (int i = 0; i < n; ++i) {
            P[i] = i;
        }

        // LU Decomposition with partial pivoting
        for (int j = 0; j < n; ++j) {
            TNum maxVal = std::abs(A(j, j));
            int maxIndex = j;
            for (int i = j + 1; i < n; ++i) {
                TNum currentVal = std::abs(A(i, j));
                if (currentVal > maxVal) {
                    maxVal = currentVal;
                    maxIndex = i;
                }
            }

            if (maxIndex != j) {
                A.swapRows(j, maxIndex);
                std::swap(P[j], P[maxIndex]);
            }

            // Check for singular matrix using a small tolerance
            const TNum epsilon = static_cast<TNum>(1e-12);
            if (std::abs(A(j, j)) < epsilon) {
                throw std::runtime_error("Matrix is singular or nearly singular and cannot be decomposed.");
            }

            for (int i = j + 1; i < n; ++i) {
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
