#ifndef ILU_PRECONDITIONER_HPP
#define ILU_PRECONDITIONER_HPP

#include <vector>
#include <stdexcept>
#include "../../Obj/SparseObj.hpp"
#include "../../Obj/DenseObj.hpp"
#include "../../Obj/VectorObj.hpp"

// Forward declaration of the primary template
template<typename TNum, typename MatrixType>
class ILUPreconditioner;

// Specialization for SparseMatrixCSC
template<typename TNum>
class ILUPreconditioner<TNum, SparseMatrixCSC<TNum>> {
private:
    SparseMatrixCSC<TNum> L;  // Lower triangular part
    SparseMatrixCSC<TNum> U;  // Upper triangular part
    int n;                    // Matrix dimension

public:
    ILUPreconditioner() : n(0) {}

    void compute(const SparseMatrixCSC<TNum>& A) {
        n = A.getRows();
        if (n != A.getCols()) {
            throw std::invalid_argument("Matrix must be square for ILU factorization");
        }

        // Initialize L and U matrices
        L = SparseMatrixCSC<TNum>(n, n);
        U = SparseMatrixCSC<TNum>(n, n);

        // Temporary storage for each column
        std::vector<TNum> work(n, 0.0);
        std::vector<bool> marker(n, false);

        // Process each column
        for (int k = 0; k < n; k++) {
            std::fill(work.begin(), work.end(), 0.0);
            std::fill(marker.begin(), marker.end(), false);

            // Copy column k of A into work array
            for (int i = A.col_ptr[k]; i < A.col_ptr[k + 1]; i++) {
                work[A.row_indices[i]] = A.values[i];
                marker[A.row_indices[i]] = true;
            }

            // Apply previous transformations
            for (int i = 0; i < k; i++) {
                if (!marker[i] || work[i] == 0.0) continue;

                TNum multiplier = work[i] / U(i, i);
                L.addValue(k, i, multiplier);

                for (int j = U.col_ptr[i]; j < U.col_ptr[i + 1]; j++) {
                    int row = U.row_indices[j];
                    if (row > i) {
                        work[row] -= multiplier * U.values[j];
                        marker[row] = true;
                    }
                }
            }

            // Store the results in L and U
            for (int i = 0; i < n; i++) {
                if (!marker[i] || work[i] == 0.0) continue;
                if (i < k) {
                    L.addValue(k, i, work[i]);
                } else {
                    U.addValue(i, k, work[i]);
                }
            }
        }

        L.finalize();
        U.finalize();

        // Add ones on the diagonal of L
        for (int i = 0; i < n; i++) {
            L.addValue(i, i, 1.0);
        }
        L.finalize();
    }

    VectorObj<TNum> solve(const VectorObj<TNum>& b) const {
        return solveUpper(solveLower(b));
    }

private:
    VectorObj<TNum> solveLower(const VectorObj<TNum>& b) const {
        VectorObj<TNum> y(n);
        for (int i = 0; i < n; i++) {
            TNum sum = 0.0;
            for (int j = L.col_ptr[i]; j < L.col_ptr[i + 1]; j++) {
                if (L.row_indices[j] < i) {
                    sum += L.values[j] * y[L.row_indices[j]];
                }
            }
            y[i] = (b[i] - sum) / L(i, i);
        }
        return y;
    }

    VectorObj<TNum> solveUpper(const VectorObj<TNum>& y) const {
        VectorObj<TNum> x(n);
        for (int i = n - 1; i >= 0; i--) {
            TNum sum = 0.0;
            for (int j = U.col_ptr[i]; j < U.col_ptr[i + 1]; j++) {
                if (U.row_indices[j] > i) {
                    sum += U.values[j] * x[U.row_indices[j]];
                }
            }
            x[i] = (y[i] - sum) / U(i, i);
        }
        return x;
    }
};

// Specialization for DenseObj
template<typename TNum>
class ILUPreconditioner<TNum, DenseObj<TNum>> {
private:
    DenseObj<TNum> L;  // Lower triangular part
    DenseObj<TNum> U;  // Upper triangular part
    int n;            // Matrix dimension

public:
    ILUPreconditioner() : n(0) {}

    void compute(const DenseObj<TNum>& A) {
        n = A.getRows();
        if (n != A.getCols()) {
            throw std::invalid_argument("Matrix must be square for ILU factorization");
        }

        // Initialize L and U matrices
        L = DenseObj<TNum>(n, n);
        U = DenseObj<TNum>(n, n);

        // Copy A into U
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                U(i, j) = A(i, j);
            }
        }

        // Set L's diagonal to 1
        for (int i = 0; i < n; i++) {
            L(i, i) = 1.0;
        }

        // Perform ILU factorization
        for (int k = 0; k < n - 1; k++) {
            for (int i = k + 1; i < n; i++) {
                if (U(i, k) != 0.0) {
                    TNum multiplier = U(i, k) / U(k, k);
                    L(i, k) = multiplier;
                    for (int j = k; j < n; j++) {
                        U(i, j) -= multiplier * U(k, j);
                    }
                }
            }
        }
    }

    VectorObj<TNum> solve(const VectorObj<TNum>& b) const {
        return solveUpper(solveLower(b));
    }

private:
    VectorObj<TNum> solveLower(const VectorObj<TNum>& b) const {
        VectorObj<TNum> y(n);
        for (int i = 0; i < n; i++) {
            TNum sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L(i, j) * y[j];
            }
            y[i] = (b[i] - sum) / L(i, i);
        }
        return y;
    }

    VectorObj<TNum> solveUpper(const VectorObj<TNum>& y) const {
        VectorObj<TNum> x(n);
        for (int i = n - 1; i >= 0; i--) {
            TNum sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += U(i, j) * x[j];
            }
            x[i] = (y[i] - sum) / U(i, i);
        }
        return x;
    }
};

#endif // ILU_PRECONDITIONER_HPP