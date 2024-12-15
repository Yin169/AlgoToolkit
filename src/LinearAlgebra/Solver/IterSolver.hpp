#ifndef ITERSOLVER_HPP
#define ITERSOLVER_HPP

#include "MatrixObj.hpp"
#include "VectorObj.hpp" 
#include "basic.hpp"
#include "SolverBase.hpp"

template <typename TNum>
class GradientDesent : public IterSolverBase<TNum> {
public:
    GradientDesent() = default;
    explicit GradientDesent(MatrixObj<TNum> P, MatrixObj<TNum> A, VectorObj<TNum> b, int max_iter)
        : IterSolverBase<TNum>(std::move(P), std::move(A), std::move(b), max_iter) {}

    virtual ~GradientDesent() = default;

    VectorObj<TNum> calGrad(VectorObj<TNum>& x) override {
        return this->b - (this->A * x);
    }

    void callUpdate(VectorObj<TNum>& x) {
        int max_iter = this->getIter();
        for (int i = 0; i < max_iter; ++i) {
            VectorObj<TNum> grad = this->calGrad(x);
            this->Update(x, grad);
        }
    }
};

template <typename TNum>
class Jacobi : public IterSolverBase<TNum> {
public:
    Jacobi() = default;
    explicit Jacobi(MatrixObj<TNum> P, MatrixObj<TNum> A, VectorObj<TNum> b, int max_iter)
        : IterSolverBase<TNum>(std::move(P), std::move(A), std::move(b), max_iter) {}

    virtual ~Jacobi() = default;

    VectorObj<TNum> Substitution(VectorObj<TNum>& b, MatrixObj<TNum>& L, bool forward) {
        int n = b.get_row();
        VectorObj<TNum> x(n);

        for (int i = forward ? 0 : n - 1; forward ? i < n : i >= 0; forward ? i++ : i--) {
            TNum sum = TNum(0);
            for (int j = forward ? 0 : n - 1; forward ? j < i : j > i; forward ? j++ : j--) {
                sum += L[j * L.get_row() + i] * x[j];
            }
            if (L[i * L.get_row() + i] == TNum(0)) {
                throw std::runtime_error(forward ? "Division by zero in forward substitution." : "Division by zero in backward substitution.");
            }
            x[i] = (b[i] - sum) / L[i * L.get_row() + i];
        }
        return x;
    }

    VectorObj<TNum> calGrad(VectorObj<TNum>& x) override {
        VectorObj<TNum> rhs = this->b - (this->A * x);
        return Substitution(rhs, this->P, true);
    }

    void callUpdate(VectorObj<TNum>& x) {
        int max_iter = this->getIter();
        for (int i = 0; i < max_iter; ++i) {
            VectorObj<TNum> grad = this->calGrad(x);
            this->Update(x, grad);
        }
    }
};

#endif // ITERSOLVER_HPP