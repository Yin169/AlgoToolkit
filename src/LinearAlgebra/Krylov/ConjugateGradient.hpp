#ifndef CONJUGATEGRADIENT_HPP
#define CONJUGATEGRADIENT_HPP

#include <VectorObj.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>

template <typename TNum, typename MatrixType, typename VectorType = VectorObj<TNum>>
class ConjugateGrad {
public:
    VectorType x, r, p;
    double _tol = 1e-6;
    int _maxIter;           // Maximum number of iterations
    MatrixType A;          // Coefficient matrix
    VectorType b;          // Right-hand side vector

    // Default constructor
    ConjugateGrad() = default;

    // Parameterized constructor
    explicit ConjugateGrad(const MatrixType& A, const VectorType& b, int maxIter, double tol) :A(std::move(A)), b(std::move(b)), _tol(tol), _maxIter(maxIter){
        // Initialize x to zero vector
        x = VectorType(b.size(), 0.0);

        // Compute initial residual and search direction
        r = b - (const_cast<MatrixType&>(A) * x);
        p = r;
    }

    // Virtual destructor
    virtual ~ConjugateGrad() = default;

    void solve(VectorType& x_out) {
        TNum rs_old = r * r;
        TNum b_norm = std::sqrt(b * b);
        if (b_norm < 1e-12) {
            x_out = x; // Solution is zero for zero RHS
            return;
        }

        if (std::sqrt(rs_old) / b_norm < _tol) {
            x_out = x; // Already converged
            return;
        }

        for (int i = 0; i < _maxIter; ++i) {
            VectorType Ap = A * p;
            TNum pAp = p * Ap;

            if (std::abs(pAp) < 1e-12) {
                throw std::runtime_error("Breakdown: Division by near-zero in CG.");
            }

            TNum alpha = rs_old / pAp;
            x = x + p * alpha;
            r = r - Ap * alpha;

            TNum rs_new = r * r;

            if (std::sqrt(rs_new) / b_norm < _tol) {
                break;
            }

            p = r + p * (rs_new / rs_old);
            rs_old = rs_new;
        }

        x_out = x;
    }
};

#endif // CONJUGATEGRADIENT_HPP
