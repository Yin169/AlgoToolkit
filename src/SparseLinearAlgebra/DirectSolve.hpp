#ifndef DIRECTSOLVE_HPP
#define DIRECTSOLVE_HPP

#include "../../Obj/VectorObj.hpp"
#include "../Obj/SparseObj.hpp"
#include "SparseBasic.hpp"

template<typename T = double >
class LUSolver{
	private:
	SparseMatrixCSC<T> L, U;
public:
	LUSolver() = default;
	~LUSolver() = default;

	LUSolver(const SparseMatrixCSC<T> &A){
		int n = A.getRows(), m = A.getCols();
        if (n != m) {
            throw std::invalid_argument("Matrix must be square for LU decomposition");
        }
		L = SparseMatrixCSC<T>(n, m);
		U = SparseMatrixCSC<T>(n, m);

		for(size_t i=0; i<n; i ++){
			L.addValue(i, i, T(1));
		}

		for (size_t j = 0; j<m; j++){
			VectorObj<T> x(n, T(0));
			for(size_t p = A.col_ptr[j]; p < A.col_ptr[j+1]; p++){
				x[A.row_indices[p]] = A.values[p];
			}
			if (j>0) {
				SparseLA::LSolve(L, x);
			}
			
			for(size_t i=0; i<=j; i++){
				if (x[i] != T(0)){
					U.addValue(i, j, x[i]);
				}
			}

			if (x[j] == T(0)) {
                throw std::runtime_error("Zero pivot encountered. Matrix may be singular or require pivoting.");
            }

            for (int i = j + 1; i < n; i++) {
                if (x[i] != T(0)) {
                    T lij = x[i] / x[j];
                    L.addValue(i, j, lij);
				}
			}
		}
        
        L.finalize();
        U.finalize();
	}
    void solve(VectorObj<T> &b){
        int n = L.getRows();
        
        // Forward substitution: Solve Ly = b
        VectorObj<T> y(n);
        for (int i = 0; i < n; i++) {
            y[i] = b[i];
            for (int j = L.col_ptr[i]; j < L.col_ptr[i+1]; j++) {
                int row = L.row_indices[j];
                if (row > i) { // Skip diagonal (which is 1)
                    y[row] -= L.values[j] * y[i];
                }
            }
        }
        
        // Backward substitution: Solve Ux = y
        for (int i = n - 1; i >= 0; i--) {
            T sum = T(0);
            for (int j = U.col_ptr[i]; j < U.col_ptr[i+1]; j++) {
                int row = U.row_indices[j];
                if (row < i) {
                    sum += U.values[j] * b[row];
                }
            }
            
            // Find diagonal element
            T diag = T(0);
            for (int j = U.col_ptr[i]; j < U.col_ptr[i+1]; j++) {
                if (U.row_indices[j] == i) {
                    diag = U.values[j];
                    break;
                }
            }
            
            if (diag == T(0)) {
                throw std::runtime_error("Matrix is singular");
            }
            
            b[i] = (y[i] - sum) / diag;
        }
    }
    
    const SparseMatrixCSC<T>& getL() const { return L; }
    const SparseMatrixCSC<T>& getU() const { return U; }
};


#endif