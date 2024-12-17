#ifndef ITERSOLVER_HPP
#define ITERSOLVER_HPP

#include "MatrixObj.hpp"
#include "VectorObj.hpp"
#include "basic.hpp"
#include "SolverBase.hpp"

// GradientDescent Solver
template <typename TNum>
class GradientDescent : public IterSolverBase<TNum> {
public:
    // Default constructor
    GradientDescent() = default;

    // Parameterized constructor
    GradientDescent(MatrixObj<TNum> P, MatrixObj<TNum> A, VectorObj<TNum> b, int max_iter)
        : IterSolverBase<TNum>(std::move(P), std::move(A), std::move(b), max_iter) {}

    // Virtual destructor
    virtual ~GradientDescent() = default;

    // Calculate the gradient (negative residual direction)
    VectorObj<TNum> calGrad(const VectorObj<TNum>& x) const override {
        return this->b - (this->A * x); // Gradient: b - Ax
    }

    // Perform the iterative gradient descent updates
    void solve(VectorObj<TNum>& x) {
        for (int i = 0; i < this->getMaxIter(); ++i) {
            VectorObj<TNum> grad = calGrad(x);
            this->Update(x, grad);
        }
    }
};

// Static Iterative Method Solver
template <typename TNum>
class StaticIterMethod : public IterSolverBase<TNum> {
public:
    // Default constructor
    StaticIterMethod() = default;

    // Parameterized constructor
    StaticIterMethod(MatrixObj<TNum> P, MatrixObj<TNum> A, VectorObj<TNum> b, int max_iter)
        : IterSolverBase<TNum>(std::move(P), std::move(A), std::move(b), max_iter) {}

    // Virtual destructor
    virtual ~StaticIterMethod() = default;

    // Perform forward or backward substitution
    VectorObj<TNum> Substitution(const VectorObj<TNum>& b, const MatrixObj<TNum>& L, bool forward) const {
        const int n = b.size();
        VectorObj<TNum> x(n);

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

public:
    // Calculate the gradient using preconditioned residual
    VectorObj<TNum> calGrad(const VectorObj<TNum>& x) const override {
        VectorObj<TNum> residual = this->b - (this->A * x);
        return Substitution(residual, this->P, true); // Forward substitution for preconditioning
    }

    // Perform the iterative updates using the static iterative method
    void solve(VectorObj<TNum>& x) {
        for (int i = 0; i < this->getMaxIter(); ++i) {
            VectorObj<TNum> grad = calGrad(x);
            this->Update(x, grad);
        }
    }
};

#endif // ITERSOLVER_HPP
