#ifndef LU_HPP
#define LU_HPP

#include <vector>
#include <stdexcept> // For std::invalid_argument
#include <cmath>     // For std::abs
#include <algorithm> // For std::swap

#include "../Factorized/basic.hpp"
/**
 * @namespace LU
 * @brief Namespace containing functions for LU decomposition.
 */

/**
 * @brief Performs LU decomposition with partial pivoting on a given matrix.
 *
 * This function decomposes a given square matrix A into a lower triangular matrix L and an upper triangular matrix U
 * such that A = P * L * U, where P is a permutation matrix. The decomposition is done in-place, modifying the input
 * matrix A and generating a permutation vector P.
 *
 * @tparam TNum The numeric type of the matrix elements (e.g., float, double).
 * @tparam MatrixType The type of the matrix (must support getRows(), getCols(), operator(), and swapRows()).
 * @param A The input matrix to be decomposed. It must be square (i.e., the number of rows must equal the number of columns).
 * @param P The permutation vector that will store the row permutations applied during the decomposition.
 * @throws std::invalid_argument If the input matrix A is not square.
 * @throws std::runtime_error If the matrix is singular or nearly singular and cannot be decomposed.
 */
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
