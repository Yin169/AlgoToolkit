#ifndef ITERSOLVER_HPP
#define ITERSOLVER_HPP

#include "Obj/MatrixObj.hpp"
#include "SolverBase.hpp"

template <typename TNum>
class GradientDesent : public IterSolverBase<TNum> {
public:
    GradientDesent() {}
    GradientDesent(MatrixObj<TNum> P, MatrixObj<TNum> A, VectorObj<TNum> b, int max_iter) :
        IterSolverBase<TNum>(P, A, b, max_iter) {}

    virtual ~GradientDesent() {}

    VectorObj<TNum> calGrad(VectorObj<TNum> &x) override {
        return this->b - (this->A * x);
    }

    void callUpdate(VectorObj<TNum> &x) {
        for (int i = 0; i < this->getIter(); ++i) {
            VectorObj<TNum> grad = this->calGrad(x);
            this->Update(x, grad); // Assign the result of Update to x
        }
    }
};

#endif // ITERSOLVER_HPP