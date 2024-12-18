#ifndef ITERSOLVER_HPP
#define ITERSOLVER_HPP

#include "basic.hpp"
#include "SolverBase.hpp"

// GradientDescent Solver
template <typename TNum, typename MatrixType, typename VectorType>
class GradientDescent : public IterSolverBase<TNum, MatrixType, VectorType> {
public:
    // Default constructor
    GradientDescent() = default;

    // Parameterized constructor
    GradientDescent(MatrixType P, MatrixType A, VectorType b, int max_iter)
        : IterSolverBase<TNum, MatrixType, VectorType>(std::move(P), std::move(A), std::move(b), max_iter) {}

    // Virtual destructor
    virtual ~GradientDescent() = default;

    // Calculate the gradient (negative residual direction)
    VectorType calGrad(const VectorType& x) const override {
        return this->b - (this->A * x); // Gradient: b - Ax
    }

    // Perform the iterative gradient descent updates
    void solve(VectorType& x) {
        for (int i = 0; i < this->getMaxIter(); ++i) {
            VectorType grad = calGrad(x);
            this->Update(x, grad);
        }
    }
};

// Static Iterative Method Solver
template <typename TNum, typename MatrixType, typename VectorType>
class StaticIterMethod : public IterSolverBase<TNum, MatrixType, VectorType> {
public:
    // Default constructor
    StaticIterMethod() = default;

    // Parameterized constructor
    StaticIterMethod(MatrixType P, MatrixType A, VectorType b, int max_iter)
        : IterSolverBase<TNum, MatrixType, VectorType>(std::move(P), std::move(A), std::move(b), max_iter) {}

    // Virtual destructor
    virtual ~StaticIterMethod() = default;

    // Perform forward or backward substitution
    VectorType Substitution(const VectorType& b, const MatrixType& L, bool forward) const {
        const int n = b.size();
        VectorType x(n);

        for (int i = forward ? 0 : n - 1; forward ? i < n : i >= 0; forward ? ++i : --i) {
            TNum sum = static_cast<TNum>(0);

            for (int j = forward ? 0 : n - 1; forward ? j < i : j > i; forward ? ++j : --j) {
                sum += L(i, j) * x[j]; // Using more readable row-major accessor
            }

            if (L(i, i) == static_cast<TNum>(0)) {
                throw std::runtime_error("Division by zero during substitution.");
            }

            x[i] = (b[i] - sum) / L(i, i);
        }

        return x;
    }

    // Calculate the gradient using preconditioned residual
    VectorType calGrad(const VectorType& x) const override {
        VectorType residual = this->b - (this->A * x);
        return Substitution(residual, this->P, true); // Forward substitution for preconditioning
    }

    // Perform the iterative updates using the static iterative method
    void solve(VectorType& x) {
        for (int i = 0; i < this->getMaxIter(); ++i) {
            VectorType grad = calGrad(x);
            this->Update(x, grad);
        }
    }
};

#endif // ITERSOLVER_HPP
