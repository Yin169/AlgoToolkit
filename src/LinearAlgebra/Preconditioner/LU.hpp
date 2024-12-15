#ifndef LU_HPP
#define LU_HPP

#include <vector>
#include <stdexcept> // For std::invalid_argument
#include "MatrixObj.hpp"
#include "VectorObj.hpp"

namespace LU {
    template <typename TNum>
    void PivotLU(MatrixObj<TNum> &A, std::vector<TNum> &P) {
        int n = A.get_row();
        if (n != A.get_col()) {
            // LU decomposition is only defined for square matrices.
            throw std::invalid_argument("Matrix must be square for LU decomposition.");
        }

        P.resize(n);
        for (int i = 0; i < n; i++) {
            P[i] = i;
        }

        for (int j = 0; j < n; j++) {
            TNum maxVal = std::abs(A[j + j * n]);
            int maxIndex = j;

            for (int i = j + 1; i < n; i++) {
                if (maxVal < std::abs(A[i + j * n])) {
                    maxVal = std::abs(A[i + j * n]);
                    maxIndex = i;
                }
            }

            if (maxIndex != j) {
                for (int k = 0; k < n; k++) {
                    std::swap(A[j + k * n], A[maxIndex + k * n]);
                }
                std::swap(P[j], P[maxIndex]);
            }

            for (int i = j + 1; i < n; i++) {
                TNum factor = A[i + j * n] / A[j + j * n];
                for (int k = j; k < n; k++) {
                    A[i + k * n] -= factor * A[j + k * n];
                }
            }
        }
    }
}

#endif