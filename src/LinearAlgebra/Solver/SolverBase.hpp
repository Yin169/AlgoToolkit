#ifndef SOLVERBASE_HPP
#define SOLVERBASE_HPP

#include <vector>
#include <stdexcept> // For std::invalid_argument
#include "MatrixObj.hpp"
#include "VectorObj.hpp"

// Template base class for iterative solvers
template <typename TNum>
class IterSolverBase {
protected:
    int maxIter; // Maximum number of iterations
    TNum alpha;  // Learning rate or step size

    MatrixObj<TNum> P; // Preconditioner matrix
    MatrixObj<TNum> A; // Coefficient matrix
    VectorObj<TNum> b; // Right-hand side vector

public:
    // Default constructor
    IterSolverBase()
        : maxIter(1000), alpha(static_cast<TNum>(1.0)) {}

    // Constructor with parameters
    IterSolverBase(MatrixObj<TNum> P_, MatrixObj<TNum> A_, VectorObj<TNum> b_, int maxIter_, TNum alpha_ = static_cast<TNum>(1.0))
        : P(std::move(P_)), A(std::move(A_)), b(std::move(b_)), maxIter(maxIter_), alpha(alpha_) {
        if (maxIter_ <= 0) {
            throw std::invalid_argument("Maximum number of iterations must be greater than 0.");
        }
        if (alpha_ <= static_cast<TNum>(0)) {
            throw std::invalid_argument("Alpha (learning rate) must be positive.");
        }
    }

    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~IterSolverBase() = default;

    // Pure virtual function to calculate the gradient (must be implemented by derived classes)
    virtual VectorObj<TNum> calGrad(const VectorObj<TNum>& x) const = 0;

    // Update the solution vector based on the gradient
    void Update(VectorObj<TNum>& x, const VectorObj<TNum>& grad) const {
        x = x + (grad * alpha); // Update x using the scaled gradient
    }

    // Get the maximum number of iterations
    inline int getMaxIter() const { return maxIter; }

    // Set the learning rate
    void setAlpha(TNum newAlpha) {
        if (newAlpha <= static_cast<TNum>(0)) {
            throw std::invalid_argument("Alpha (learning rate) must be positive.");
        }
        alpha = newAlpha;
    }

    // Get the learning rate
    TNum getAlpha() const { return alpha; }
};

#endif // SOLVERBASE_HPP
