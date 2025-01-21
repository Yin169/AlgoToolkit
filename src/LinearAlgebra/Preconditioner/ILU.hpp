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

        // Initialize L and U with the same sparsity pattern as A
        L = MatrixType(n, n);
        U = MatrixType(n, n);
        
         // LU Decomposition 
        for (int j = 0; j < n; ++j) {
            // Check for singular matrix using a small tolerance
            const TNum epsilon = static_cast<TNum>(1e-12);
            if (std::abs(A(j, j)) < epsilon) {
                throw std::runtime_error("Matrix is singular or nearly singular and cannot be decomposed.");
            }

            for (int i = j + 1; i < n; ++i) {
                TNum factor = A(i, j) / A(j, j);
                // A(i, j) = factor; // Store the factor in place
                A.addValue(i, j, factor);
                A.finalize();
                
                for (int k = j + 1; k < n; ++k) {
                    // A(i, k) = A(i, k) - factor * A(j, k);
                    A.addValue(i, k, A(i, k) - factor * A(j, k));
                    A.finalize();
                }
            }
        }


        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i > j) {
                    // L(i, j) = A(i, j); // Lower triangular part
                    L.addValue(i, j, A(i, j));
                } else {
                    // U(i, j) = A(i, j); // Upper triangular part
                    U.addValue(i, j, A(i, j));
                }
            }
            L.addValue(i, i, TNum(1.0)); // Set diagonal elements of L to 1
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
            y[i] = b[i] - sum;  // L has 1's on diagonal
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

    // Get the L and U factors for debugging or analysis
    const MatrixType& getLFactor() const { return L; }
    const MatrixType& getUFactor() const { return U; }
};

#endif // ILU_HPP