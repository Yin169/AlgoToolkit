#ifndef ITERSOLVER_HPP
#define ITERSOLVER_HPP

#include "Obj/MatrixObj.hpp"
#include "LinearAlgebra/Factorized/basic.hpp"
#include "SolverBase.hpp" 

template <typename TNum>
class GradientDesent : public IterSolverBase<TNum> {
public:
    GradientDesent() {}

    GradientDesent(const MatrixObj<TNum> P, const MatrixObj<TNum> &A, const VectorObj<TNum> &b, VectorObj<TNum> x, int max_iter) :
        IterSolverBase<TNum>(P, A, b, x, max_iter) {}

    virtual ~GradientDesent() {}

    void calGrad() override {
        MatrixObj<TNum> xt(this->_x);
		MatrixObj<TNum> gradt;
        gradt = this->b - this->A * this->xt;
    }

    TNum calAlp(TNum hype) override { return hype; }

    void Update() override {
        int iterNum = this->getIter();
        while(iterNum--){
            calGrad();
            this->Update();
        }
    }

    TNum getAns() { return this->Ans(); }
};

#endif 