#ifndef BASIC_HPP
#define BASIC_HPP

#include "../../Obj/VectorObj.hpp"

namespace PowerMethod {

	template <typename TNum>
	void powerIter(const MatrixObj<TNum> &A, VectorObj<TNum> &b, int max_iter_num) {
		while(max_iter_num--){
			b.normalized();	
			A *= b;	
		}
	}
	template <typename TNum>	
	double rayleighQuotient(const MatrixObj<TNum> &A, const VectorObj<TNum> &b){ return b.Tranpose() * A * b / b.Tranpose() * b;}
}

#endif
