#ifndef SOLVERBASE_HPP
#define SOLVERBASE_HPP

#include <vector>
#include "Obj/MatrixObj.hpp"
#include "LinearAlgebra/Factorized/basic.hpp"

template <typename TNum>
class IterSolverBase{
	private:
	int _n, _m, _max_iter, alp;
	MatrixObj<TNum> _P;
	MatrixObj<TNum> _A;
	VectorObj<TNum> _x;
	VectorObj<TNum> _b;
	vector<VectorObj<TNum>> _grad;

	public:
	SolverBase(){}
	SolverBase(const MatrixObj<TNum> P, const MatrixObj<TNum> &A, const VectorObj<TNum> &b, const VectorObj<TNum> &x, int max_iter) : \
		_P(P), _A(A), _b(b), _x(x), _n(A.get_row()), _m(A.get_col()), \
		_max_iter(max_iter){}

	~SolverBase(){}

	virtual void calGrad()=0;
	virtual TNum calAlp()=0;
	void Update(){ _x = _x + calAlp() * _grad.back();}

};

#endif
