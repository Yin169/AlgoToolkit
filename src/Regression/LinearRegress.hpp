#ifndef LINEARREGRESS_HPP
#define LINEARREGRESS_HPP

#include "../Obj/VectorObj.hpp"
#include "../Obj/DenseObj.hpp"
#include "../../LinearAlgebra/Krylov/GMRES.hpp"

template <typename T ,typename MatrixType>
class LinearRegress{
	private:
	size_t n_features;
	VectorObj<T> coef_;	

	public:
	LinearRegress(size_t n) : n_features(n), coef_(VectorObj<T>(n+1)){};
	~LinearRegress() = default;
	
	void fit(const MatrixType& X, const VectorObj<T>& y, size_t max_iter = 1000, T tol = 1e-8){
		if (X.getRows() != y.size()){
			throw std::invalid_argument("Matrix rows and vector size do not match.");
		}
		MatrixType Xs = X;
		Xs.appendColumn(VectorObj<T>(X.getRows(), T(1)));
		MatrixType X_trans = Xs.Transpose();
		MatrixType A = X_trans * Xs;
		VectorObj<T> b = X_trans * y;

		GMRES<T, MatrixType> gmres;
		gmres.enablePreconditioner();
		gmres.solve(A, b, coef_, max_iter, n_features, tol);
	}

	T predict(const VectorObj<T>& x) const{
		if (x.size() != n_features){
			throw std::invalid_argument("Input vector size does not match the number of features.");
		}
		T result = coef_[n_features];
		for (size_t i = 0; i < n_features; i++){
			result += coef_[i] * x[i];
		}
		return result;
	}

};

#endif // "LINEARREGRESS_HPP"