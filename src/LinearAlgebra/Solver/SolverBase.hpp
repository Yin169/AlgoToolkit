#ifndef SOLVERBASE_HPP
#define SOLVERBASE_HPP

#include <vector>
#include "Obj/MatrixObj.hpp"
#include "LinearAlgebra/Factorized/basic.hpp"

template <typename TNum>
class IterSolverBase {
private:
    int _n, _m, _max_iter=1000;
    TNum _x;

public:
    TNum alp;
    const MatrixObj<TNum> P;
    const MatrixObj<TNum> A;
    const VectorObj<TNum> b;
    VectorObj<TNum> grad;

    IterSolverBase() : _n(0), _m(0), _max_iter(0), _x(0), alp(0) {}

    IterSolverBase(const MatrixObj<TNum> P, const MatrixObj<TNum> &A, const VectorObj<TNum> &b, VectorObj<TNum> x, int max_iter) :
        P(P), A(A), b(b), grad(x), _x(x), _n(A.get_row()), _m(A.get_col()), _max_iter(max_iter), alp(0) {
            if (max_iter < 0) {
                throw std::invalid_argument("Number of Iteration must GT 0");
            }
        }

    ~IterSolverBase();

    virtual void calGrad() = 0;
    virtual TNum calAlp(TNum hype) = 0;

    virtual void Update() {
		grad *= calAlp(alp);
        _x = _x + grad; 
    }
    int getIter(){return _max_iter;}
    TNum Ans() { return _x; }
};

#endif