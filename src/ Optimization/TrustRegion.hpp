#ifndef TRUSTREGION_HPP
#define TRUSTREGION_HPP

#include <cmath>
#include <limits>
#include <stdexcept>
#include "../Obj/SparseObj.hpp"
#include "../Obj/DenseObj.hpp"
#include "../Obj/VectorObj.hpp"
#include "../LinearAlgebra/Krylov/ConjugateGradient.hpp"

template <typename TObj>
class TrustRegion {
private:
    double delta; // Trust region radius
    double eta;   // Acceptance threshold

public:
    // Constructor
    TrustRegion(double initial_delta, double acceptance_threshold)
        : delta(initial_delta), eta(acceptance_threshold) {}

    // Virtual destructor
    virtual ~TrustRegion() = default;

    // Solve the nonlinear system using Trust Region method
    VectorObj<TObj> solve(const VectorObj<TObj>& x0, 
                          const SparseMatrixCSC<TObj>& A, 
                          const VectorObj<TObj>& b, 
                          int maxIter, 
                          double tol) 
    {
        VectorObj<TObj> x = x0;
        VectorObj<TObj> r = b - A * x; // Residual
        if (r.L2norm() < tol) {
            return x; // Initial guess is a good enough solution
        }

        for (int iter = 0; iter < maxIter; ++iter) {
            // Solve the linearized subproblem using Conjugate Gradient method
            ConjugateGrad<TObj, SparseMatrixCSC<TObj>, VectorObj<TObj>> cg(A, r, maxIter, tol);
            VectorObj<TObj> p;
            cg.solve(p);

            // Check the step size and update trust region radius
            if (p.L2norm() > delta) {
                p = p * (delta / p.L2norm());
            }

            // Evaluate the actual reduction
            TObj actual_reduction = r * (A * p);

            // Predicted reduction
            TObj predicted_reduction = -0.5 * (p * (A * p));

            // Ratio of actual to predicted reduction
            double rho = actual_reduction / predicted_reduction;

            // Update the solution
            if (rho > eta) {
                x = x + p;
                r = b - A * x;
            }

            // Adjust the trust region radius
            if (rho > 0.75) {
                delta = std::min(delta * 2.0, 1.0);
            } else if (rho < 0.25) {
                delta = std::max(delta * 0.25, tol);
            }

            // Check for convergence
            if (r.L2norm() < tol) {
                return x; // Converged to the solution
            }
        }

        throw std::runtime_error("Trust Region method did not converge within the maximum number of iterations.");
    }
};

#endif // TRUSTREGION_HPP