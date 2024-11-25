#ifndef BASIC_HPP
#define BASIC_HPP

#include <cmath>
#include "../../Obj/VectorObj.hpp"

namespace basic {

	template <typename TNum>
	void powerIter(const MatrixObj<TNum> &A, VectorObj<TNum> &b, int max_iter_num) {
		while(max_iter_num--){
			b.normalized();	
			A *= b;	
		}
	}

	template <typename TNum>
	double rayleighQuotient(const MatrixObj<TNum> &A, const VectorObj<TNum> &b){ return b.Tranpose() * A * b / b.Tranpose() * b;}


	template <typename TNum>
	VectorObj<TNum> subtProj(const VectorObj<TNum> &u, const VectorObj<TNum> &v){
		double factor = u.Tranpose() * v / v.Tranpose() * v;
		return u - factor * v;
	}

	template <typename TNum>
	MatrixObj<TNum> gramSmith(const MatrixObj<TNum> &A){
		size_t n = A.get_row(), m = A.get_col();
		TNum* temp = new TNum [n*m];
		for (size_t i=0; i<n; i++){

			TNum* us = new TNum [n];
			std::copy(us + i * n, us + (i+1) * n, temp + i * n);
			A.Slice(us, i * n, (i + 1) * n);
			VectorObj<TNum> u(us, n);
			delete[] us;

			for (size_t j=i-1; j>=0; j--){

				TNum* vs = new TNum* [n];
				A.Slice(vs, j * n, (j + 1) * n);
				VectorObj<TNum> v(vs, n);
				VectorObj<TNum> proj = subtProj(v, u);
				proj.Slice(vs, 0, n);
				std::copy(vs, vs+n, temp+j*n);
				delete[] vs;
			}
		}
		MatrixObj<TNum> orthSet(temp, n, m);
		delete[] temp;
		return orthSet;
	}



}



namespace KrylovSub{
	/*
	 * Ardoni
	 */
}

#endif
