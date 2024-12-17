#ifndef CONJUGATEGRADIENT_HPP
#define CONJUGATEGRADIENT_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include "MatrixObj.hpp"
#include "VectorObj.hpp"
#include "SolverBase.hpp"

template <typename TNum>
class ConjugateGrad : public IterSolverBase<TNum> {
public:
    VectorObj<TNum> x, r, p;
    double _tol = 1e-6;

    ConjugateGrad() = default;

    explicit ConjugateGrad(MatrixObj<TNum> P, MatrixObj<TNum> A, VectorObj<TNum> b, int maxIter, double tol)
        : _tol(tol), IterSolverBase<TNum>(std::move(P), std::move(A), std::move(b), maxIter) {
        // Initialize x to zero vector
        x = VectorObj<TNum>(b.size());

        // Compute initial residual and search direction
        r = this->b - (this->A * x);
        p = r;
    }

    virtual ~ConjugateGrad() = default;

    // dummy function
    VectorObj<TNum> calGrad(const VectorObj<TNum>& x) const {return VectorObj<TNum>();}; 

    void callUpdate() {
        int maxIter = this->getMaxIter();
        double residualNorm = r.L2norm();

        // Iterative loop
        for (int iter = 0; iter < maxIter; ++iter) {
            // Compute A*p and cache it
            VectorObj<TNum> Ap = this->A * p;

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
