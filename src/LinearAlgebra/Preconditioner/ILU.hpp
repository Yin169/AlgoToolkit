#ifndef ILU_HPP
#define ILU_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include "../../Obj/SparseObj.hpp"
#include "../../Obj/VectorObj.hpp"
#include "../../utils.hpp"

template <typename TNum, typename MatrixType = SparseMatrixCSC<TNum>>
class ILUPreconditioner {
private:
    MatrixType L, U;
    int n;
    bool isComputed;

public:
    ILUPreconditioner() : isComputed(false), n(0) {}

    void compute(const MatrixType& Matrix) {
        MatrixType A = Matrix;
        n = A.getRows();
        if (n != A.getCols()) {
            throw std::invalid_argument("Matrix must be square for ILU factorization");
        }

        // Initialize L and U matrices
        L = MatrixType(n, n);
        U = MatrixType(n, n);
        
        // Temporary storage for column values
        std::vector<TNum> work(n);
        const TNum epsilon = static_cast<TNum>(1e-12);
        
        // Perform ILU factorization
        for (int j = 0; j < n; ++j) {
            // Check for numerical stability
            if (std::abs(A(j, j)) < epsilon) {
                throw std::runtime_error("Matrix is numerically singular");
            }

            // Cache column j for better memory access
            for (int i = j; i < n; ++i) {
                work[i] = A(i, j);
            }

            // Compute multipliers and update matrix
            for (int i = j + 1; i < n; ++i) {
                TNum factor = work[i] / work[j];
                if (std::abs(factor) > epsilon) {
                    A.addValue(i, j, factor);
                    
                    // Update remaining elements using Gaussian elimination
                    for (int k = j + 1; k < n; ++k) {
                        TNum update = A(i, k) - factor * A(j, k);
                        if (std::abs(update) > epsilon) {
                            A.addValue(i, k, update);
                        }
                    }
                }
            }
            A.finalize();
        }

        // Extract L and U factors
        for (int i = 0; i < n; ++i) {
            L.addValue(i, i, TNum(1.0)); // Diagonal of L is always 1
            for (int j = 0; j < n; ++j) {
                TNum value = A(i, j);
                if (std::abs(value) > epsilon) {
                    if (i > j) {
                        L.addValue(i, j, value);
                    } else if (i <= j) {
                        U.addValue(i, j, value);
                    }
                }
            }
        }

        L.finalize();
        U.finalize();
        isComputed = true;
    }

    VectorObj<TNum> solve(const VectorObj<TNum>& b) const {
        if (!isComputed) {
            throw std::runtime_error("ILU factorization not computed");
        }
        if (static_cast<int>(b.size()) != n) {
            throw std::invalid_argument("Vector size does not match matrix size");
        }

        // Forward substitution (Ly = b)
        VectorObj<TNum> y(n);
        for (int i = 0; i < n; ++i) {
            TNum sum = TNum(0);
            for (int j = 0; j < i; ++j) {
                sum += L(i,j) * y[j];
            }
            y[i] = b[i] - sum;
        }

        // Back substitution (Ux = y)
        VectorObj<TNum> x(n);
        for (int i = n - 1; i >= 0; --i) {
            TNum sum = TNum(0);
            for (int j = i + 1; j < n; ++j) {
                sum += U(i,j) * x[j];
            }
            x[i] = (y[i] - sum) / U(i,i);
        }

        return x;
    }

    const MatrixType& getLFactor() const { return L; }
    const MatrixType& getUFactor() const { return U; }
};

#endif // ILU_HPP