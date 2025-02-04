#ifndef ITER_SOLVER_HPP
#define ITER_SOLVER_HPP

#include "../../Obj/SparseObj.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <optional>

// Gradient Descent Solver Template with Preconditioner and Convergence Check
template <typename TNum, typename SparseMatrixType, typename VectorType>
class GradientDescent {
private:
    const SparseMatrixType& A;
    const VectorType& b;
    int maxIter;
    TNum tol;

public:
    GradientDescent(const SparseMatrixType& matrix, const VectorType& rhs, int iterations, TNum tolerance)
        : A(matrix), b(rhs), maxIter(iterations), tol(tolerance) {}

    void solve(VectorType& x) {
        VectorType r = b - (const_cast<SparseMatrixType&>(A) * x); // Initial residual
        TNum residualNorm = r.L2norm();
        VectorType xOld = x;

        for (int iter = 0; iter < maxIter && residualNorm > tol; ++iter) {
            VectorType Ar = const_cast<SparseMatrixType&>(A) * r;
            TNum alpha = (r * r) / (r * Ar);
            // std::cout << (r * Ar) << std::endl;
            x = x + r * alpha;
            r = r - Ar * alpha;

            residualNorm = r.L2norm();

            // Early convergence check based on the change in solution
            TNum changeNorm = (x - xOld).L2norm();
            if (changeNorm < tol) {
                break;  // Converged early
            }
            xOld = x;
        }
    }
};

// Static Iterative Method (e.g., Jacobi or Gauss-Seidel) with Relaxation and Convergence Check
template <typename TNum, typename SparseMatrixType, typename VectorType>
class SOR {
private:
    const SparseMatrixType& A;
    const VectorType& b;
    int maxIter;
    TNum omega; // Relaxation factor for methods like SOR

public:
    SOR(const SparseMatrixType& matrix, const VectorType& rhs, int iterations, TNum relaxation = 1.0)
        : A(matrix), b(rhs), maxIter(iterations), omega(relaxation) {}

    void solve(VectorType& x) {
        VectorType xOld(x.size(), 0.0);
        TNum changeNorm = 0.0;

        for (int iter = 0; iter < maxIter; ++iter) {
            for (int i = 0; i < A.getRows(); ++i) {
                TNum diag = A(i, i);
                if (std::abs(diag) < 1e-10) {
                    throw std::runtime_error("Matrix has near-zero diagonal element.");
                }

                TNum sum = b[i];
                for (int k = A.col_ptr[i]; k < A.col_ptr[i + 1]; ++k) {
                    int j = A.row_indices[k];
                    if (j != i) {
                        sum -= A.values[k] * x[j];
                    }
                }

                xOld[i] = x[i];
                x[i] = (1 - omega) * xOld[i] + omega * (sum / diag);
            }

            // Check for convergence based on the change in solution
            changeNorm = (x - xOld).L2norm();
            if (changeNorm < 1e-6) {
                break;  // Converged early
            }
        }

        if (changeNorm >= 1e-6) {
            throw std::runtime_error("Solution did not converge within the specified iterations.");
        }
    }
};
#endif // ITER_SOLVER_HPP
