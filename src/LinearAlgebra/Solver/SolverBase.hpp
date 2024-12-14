#ifndef SOLVERBASE_HPP
#define SOLVERBASE_HPP

#include <vector>
#include "Obj/MatrixObj.hpp"
#include "LinearAlgebra/Factorized/basic.hpp"

// Template base class for iterative solvers
template <typename TNum>
class IterSolverBase {
private:
    int maxIter; // Maximum number of iterations

public:
    TNum alpha = static_cast<TNum>(1.0); // Learning rate or step size
    MatrixObj<TNum> P; // Preconditioner matrix
    MatrixObj<TNum> A; // Coefficient matrix
    VectorObj<TNum> b; // Right-hand side vector

    // Default constructor
    IterSolverBase() : maxIter(1000) {}

    // Constructor with parameters
    IterSolverBase(MatrixObj<TNum> P, MatrixObj<TNum> A, VectorObj<TNum> b, int maxIter)
        : P(std::move(P)), A(std::move(A)), b(std::move(b)), maxIter(maxIter) {
        if (maxIter < 0) {
            throw std::invalid_argument("Number of iterations must be greater than 0");
        }
    }

    // Virtual destructor
    virtual ~IterSolverBase() {}

    // Pure virtual function to calculate the gradient
    virtual VectorObj<TNum> calGrad(VectorObj<TNum> &x) = 0;

    // Update the solution vector based on the gradient
    void Update(VectorObj<TNum> &x, VectorObj<TNum> &grad) {
        grad *= alpha;
        x = x + grad; // Return a new vector instead of modifying x
    }

    // Get the maximum number of iterations
    int getIter() { return maxIter; }
};

#endif // SOLVERBASE_HPP