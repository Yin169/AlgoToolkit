#include <iostream>
#include "utils.hpp"
#include "DenseObj.hpp"
#include "SparseObj.hpp"
#include "VectorObj.hpp"
#include "GMRES.hpp"

template<typename MatrixType>
void solve(){
	MatrixType A;
	MatrixType bm;
	utils::readfile<double, MatrixType>("../data/bcsstk01/bcsstk01.mtx", A);
	std::cout << " A Row: " << A.getRows() << " A Col" << A.getCols() << std::endl;

	VectorObj<double> b(A.getCols());
	for(int i = 0; i < A.getCols(); i++){
		double rand = utils::GenRandom();
		b = b + A.getColumn(i) * rand;
		printf("%0.4f ", rand);	
	}
	printf("\n\n");

	VectorObj<double> x(b.size());
	GMRES<double, MatrixType, VectorObj<double>> gmres;
	// gmres.enablePreconditioner();
	gmres.solve(A, b, x, 1000, std::min(A.getRows(), A.getCols()), 1e-6);

	for(size_t i=0; i<x.size(); i++){
		printf("%0.4f ", x[i]);
	}
	printf("\n");
}

int main() {
	solve<DenseObj<double>>();
	solve<SparseMatrixCSC<double>>();
	return 0;
}