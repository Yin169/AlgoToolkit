#ifndef KRYLOVSUBSPACE_HPP
#define KRYLOVSUBSPACE_HPP

#include <iostream>
#include <vector>
#include <cmath> 
#include <assert.h>

#include "MatrixObj.hpp" 
#include "VectorObj.hpp" 

namespace Krylov {
	template<typename TNum>
	void Arnoldi(MatrixObj<TNum>& A, std::vector<VectorObj<TNum>>& Q, MatrixObj<TNum>& H, TNum tol) {
    	size_t m = Q.size();
    	assert(H.get_row() == m + 1 && H.get_col() == m);

    	for (size_t i = 1; i <= m; ++i) {
        	VectorObj<TNum> Av = A * Q[i - 1];

        	// Orthogonalization
        	for (size_t j = 0; j < i; ++j) {
            	TNum h = Q[j] * Av;
            	H(j, i - 1) = h;
            	Av = Av - Q[j] * h;
        	}

        	// Reorthogonalization
        	for (size_t j = 0; j < i; ++j) {
            	TNum h = Q[j] * Av;
            	Av = Av - Q[j] * h;
        	}

        	// Normalize the vector
        	TNum h = Av.L2norm();
        	if (h < tol) {
            	std::cout << "Breakdown: Norm below tolerance at iteration " << i << "\n";
            	break;
        	}
        	H(i, i - 1) = h;
        	Q[i] = Av / h;
    	}
	}
}
#endif // KRYLOVSUBSPACE_HPP