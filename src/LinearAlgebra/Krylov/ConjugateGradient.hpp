#ifndef CONJUGATEGRADIENT_HPP
#define CONJUGATEGRADIENT_HPP

#include <vector>
#include <cmath>
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
        x = VectorObj<TNum>(b.get_row());

        // Compute initial residual and search direction
        r = this->b - (this->A * x);
        p = r;
    }

    virtual ~ConjugateGrad() = default;
	VectorObj<TNum> calGrad(VectorObj<TNum> &x){return x;}

    void callUpdate() {
        int maxIter = this->getIter();
        double residualNorm = r.L2norm();

        // Iterative loop
        while (maxIter--) {
            // Compute A*p and cache it
            VectorObj<TNum> Ap = this->A * p;

            // Compute alpha
            double rDotR = r * r;
            double pDotAp = p * Ap;
            if (std::abs(pDotAp) < 1e-12) {
                break;
                // Prevent division by zero or near-zero values
                throw std::runtime_error("Division by near-zero in alpha calculation.");
            }
            double alpha = rDotR / pDotAp;

            // Update x and r
            x = x + (p * alpha);
            r = r - (Ap * alpha);

            // Check for convergence
            residualNorm = r.L2norm();
            if (residualNorm < _tol) break;

            // Compute beta
            double rDotR_new = r * r;
            if (std::abs(rDotR) < 1e-12) {
                break;
                // Prevent division by zero or near-zero values
                throw std::runtime_error("Division by near-zero in beta calculation.");
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