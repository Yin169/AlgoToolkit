#ifndef ITERSOLVER_HPP
#define ITERSOLVER_HPP

#include "Obj/MatrixObj.hpp"
#include "Factorized/basic.hpp"
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
private:
    MatrixObj<TNum> Pinv;

    void computePinv(const MatrixObj<TNum>& P) {
        basic::genUnitMatx(P.get_row(), P.get_col(), Pinv);
        for (int i = 0; i < P.get_row(); i++) {
            TNum diag_elem = P[i * P.get_row() + i];
            if (diag_elem == 0) {
                throw std::runtime_error("Diagonal element of P is zero, cannot compute inverse.");
            }
            Pinv[i * P.get_row() + i] = 1 / diag_elem;
        }
    }

public:
    Jacobi() = default;
    explicit Jacobi(MatrixObj<TNum> P, MatrixObj<TNum> A, VectorObj<TNum> b, int max_iter)
        : IterSolverBase<TNum>(std::move(P), std::move(A), std::move(b), max_iter) {
        computePinv(this->P);
    }

    virtual ~Jacobi() = default;

    VectorObj<TNum> calGrad(VectorObj<TNum>& x) override {
        return Pinv * (this->b - (this->A * x));
    }

    void callUpdate(VectorObj<TNum>& x) {
        int max_iter = this->getIter();
        for (int i = 0; i < max_iter; ++i) {
            VectorObj<TNum> grad = this->calGrad(x);
            this->Update(x, grad);
        }
    }

    MatrixObj<TNum> getPinv() {return Pinv;} 
};

#endif // ITERSOLVER_HPP