#ifndef NEWTONMETHOD_HPP
#define NEWTONMETHOD_HPP

#include <iostream>
#include <cmath>
#include <functional>
#include "VectorObj.hpp"
#include "GMRES.hpp"

template<typename T, typename MatrixType, typename VectorType = VectorObj<T>>
class NewtonMethod {
private:
    T tol; // Tolerance for convergence
    size_t max_iter; // Maximum number of iterations
    T alpha; // Step size for line search (optional)
    GMRES<T, MatrixType, VectorType> linear_solver; // Linear solver for the Newton step

public:
    // Constructor with default values
    NewtonMethod(T tol_ = 1e-6, size_t max_iter_ = 1000, T alpha_ = 1.0)
        : tol(tol_), max_iter(max_iter_), alpha(alpha_) {}

    // Destructor
    ~NewtonMethod() = default;

    // Solve the nonlinear system F(x) = 0 using Newton's method
    void solve(VectorType &x, 
               const std::function<VectorType(const VectorType &)> &F, // Nonlinear function F(x)
               const std::function<MatrixType(const VectorType &)> &J) // Jacobian J(x)
    {
        size_t iter = 0;
        VectorType residual;
        VectorType delta_x;
        T residual_norm;
        // linear_solver.enablePreconditioner();

        while (iter < max_iter) {
            // Evaluate the nonlinear function F(x)
            residual = F(x);

            // Check for convergence: stop if ||F(x)|| < tol
            residual_norm = residual.L2norm();
            if (residual_norm < tol) {
                std::cout << "Converged after " << iter << " iterations." << std::endl;
                return;
            }

            // Evaluate the Jacobian matrix J(x)
            MatrixType J_x = J(x);

            // Solve the linear system J(x) * delta_x = -F(x) for the update delta_x
            VectorType rhs = residual * T(-1.0); // Right-hand side of the linear system
            delta_x = VectorType(x.size(), T(0.0)); // Initialize delta_x to zero

            linear_solver.solve(J_x, rhs, delta_x, max_iter, std::min(J_x.getRows(), J_x.getCols()), tol);

            // Update the solution: x = x + alpha * delta_x
            x = x + delta_x * alpha;

            iter++;
        }

        // If the loop ends without convergence
        std::cerr << "NewtonMethod did not converge within " << max_iter << " iterations." << std::endl;
    }
};

#endif