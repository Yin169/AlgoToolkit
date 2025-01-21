#ifndef ILU_HPP
#define ILU_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include "../../Obj/SparseObj.hpp"
#include "../../Obj/VectorObj.hpp"

template <typename TNum, typename MatrixType = SparseMatrixCSC<TNum>>
class ILUPreconditioner {
private:
    MatrixType L, U;
    int n;
    bool isComputed;

public:
    ILUPreconditioner() : isComputed(false), n(0) {}

    void compute(const MatrixType& A) {
        n = A.getRows();
        if (n != A.getCols()) {
            throw std::invalid_argument("Matrix must be square for ILU factorization");
        }

        // Initialize L and U with the same sparsity pattern as A
        L = MatrixType(n, n);
        U = MatrixType(n, n);
        
        // Copy A's structure to U initially
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                TNum val = A(i, j);
                if (val != TNum(0)) {
                    if (i > j) {
                        L.addValue(i, j, val);
                    } else {
                        U.addValue(i, j, val);
                    }
                }
            }
        }
        
        L.finalize();
        U.finalize();

        // Perform the incomplete LU factorization
        for (int k = 0; k < n - 1; ++k) {
            // Check for zero pivot
            if (std::abs(U(k,k)) < 1e-12) {
                throw std::runtime_error("Zero pivot encountered in ILU factorization");
            }

            for (int i = k + 1; i < n; ++i) {
                if (L(i,k) != TNum(0)) {
                    TNum factor = L(i,k) / U(k,k);
                    L.addValue(i, k, factor);
                    
                    for (int j = k + 1; j < n; ++j) {
                        if (U(k,j) != TNum(0)) {
                            TNum existing = U(i,j);
                            if (existing != TNum(0)) {
                                U.addValue(i, j, existing - factor * U(k,j));
                            }
                        }
                    }
                }
            }
        }

        // Add ones to L's diagonal
        for (int i = 0; i < n; ++i) {
            L.addValue(i, i, TNum(1));
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