#ifndef CONJUGATEGRADIENT_HPP
#define CONJUGATEGRADIENT_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include "SolverBase.hpp"

template <typename TNum, typename MatrixType, typename VectorType>
class ConjugateGrad : public IterSolverBase<TNum, MatrixType, VectorType> {
public:
    VectorType x, r, p;
    double _tol = 1e-6;

    // Default constructor
    ConjugateGrad() = default;

    // Parameterized constructor
    explicit ConjugateGrad(MatrixType P, MatrixType A, VectorType b, int maxIter, double tol)
        : _tol(tol), IterSolverBase<TNum, MatrixType, VectorType>(std::move(P), std::move(A), std::move(b), maxIter) {
        // Initialize x to zero vector
        x = VectorType(this->b.size());

        // Compute initial residual and search direction
        r = this->b - (this->A * x);
        p = r;
    }

    // Virtual destructor
    virtual ~ConjugateGrad() = default;

    // Dummy function (not used in this solver)
    VectorType calGrad(const VectorType& x) const override {
        return VectorType();
    };

    // Perform the conjugate gradient updates
    void callUpdate() {
        int maxIter = this->getMaxIter();
        double residualNorm = r.L2norm();

        // Iterative loop
        for (int iter = 0; iter < maxIter; ++iter) {
            // Compute A*p and cache it
            VectorType Ap = this->A * p;

            // Compute alpha (step size)
            double rDotR = r * r;
            double pDotAp = p * Ap;
            if (std::abs(pDotAp) < 1e-12) {
                std::cerr << "Warning: Division by near-zero in alpha calculation.\n";
                break; // Prevent division by zero or near-zero values
            }
            double alpha = rDotR / pDotAp;

            // Update x and r
            x = x + (p * alpha);
            r = r - (Ap * alpha);

            // Check for convergence
            residualNorm = r.L2norm();
            if (residualNorm < _tol) {
                std::cout << "Conjugate Gradient converged in " << iter + 1 << " iterations.\n";
                return; // Converged
            }

            // Compute beta (for new search direction)
            double rDotR_new = r * r;
            if (std::abs(rDotR) < 1e-12) {
                std::cerr << "Warning: Division by near-zero in beta calculation.\n";
                break; // Prevent division by zero or near-zero values
            }
            double beta = rDotR_new / rDotR;

            // Update search direction p
            p = r + (p * beta);
        }

        if (residualNorm > _tol) {
            std::cerr << "Warning: Conjugate Gradient did not converge within the maximum number of iterations.\n";
        }
    }
};

#endif // CONJUGATEGRADIENT_HPP
