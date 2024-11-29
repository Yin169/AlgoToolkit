#ifndef SOLVERBASE_HPP
#define SOLVERBASE_HPP

#include "Obj/MatrixObj.hpp"
#include "LinearAlgebra/Factorized/basic.hpp"
#include "IterSolver.hpp" 

template <typename TNum>
class GradientDesent : public IterSolverBase<TNum> {
public:
    GradientDesent() {}

    GradientDesent(const MatrixObj<TNum> P, const MatrixObj<TNum> &A, const VectorObj<TNum> &b, VectorObj<TNum> x, int max_iter) :
        IterSolverBase<TNum>(P, A, b, x, max_iter) {}

    virtual ~GradientDesent() {}

    void calGrad() override {
        MatrixObj<TNum> xt(_x);
		MatrixObj<TNum> gradt;
        gradt = A * xt - b;
		grad = static_cast<TNum>(-1) * gradt.get_Col(0)

    }

    TNum calAlp(TNum hype) override { return hype; }

    void Update() override {
        IterSolverBase<TNum>::Update();
    }

    TNum getAns() { return IterSolverBase<TNum>::Ans(); }
};

#endif 