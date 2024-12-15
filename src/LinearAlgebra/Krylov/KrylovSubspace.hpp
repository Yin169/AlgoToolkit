#ifndef KRYLOVSUBSPACE_HPP
#define KRYLOVSUBSPACE_HPP

#include <vector>
#include <cmath> 

#include "MatrixObj.hpp" 
#include "VectorObj.hpp" 

namespace Krylov {
	template <typename TNum>
	void Arnoldi(MatrixObj<TNum> &A, VectorObj<TNum> &r, std::vector<VectorObj<TNum>> &Q, MatrixObj<TNum> &H, int maxIter, TNum tol){
		int n = A.get_col();
		int iternum = std::min(n, maxIter);
		
		if (H.get_row() != iternum + 1 || H.get_col() != iternum) {
			H.resize(iternum + 1 , iternum);
		}

		r.normalized();	
		Q.resize(iternum + 1);
		Q[0] = r;

		for (int i = 1; i <= iternum; ++i){
			VectorObj<TNum> Av = A * Q[i-1];
			for (int j = 0; j < i; ++j){
				TNum h = Q[j] * Av;
				H[j + (i-1) * (iternum + 1) ] = h;
				Av = Av - Q[j] * h;
			}
			TNum h = Av.L2norm();
			if (h < tol) break;
			H[i + (i-1) * (iternum + 1)] = h;
			Q[i] = Av / h;
		}
	}
}
#endif // KRYLOVSUBSPACE_HPP