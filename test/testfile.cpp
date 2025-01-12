#include <iostream>
#include "utils.hpp"
#include "DenseObj.hpp"
#include "SparseObj.hpp"
#include "VectorObj.hpp"
#include "GMRES.hpp"

int main() {
	// Declare a matrix:
	SparseMatrixCSC<double> A;
	SparseMatrixCSC<double> bm;
	
	utils::readfile<double, SparseMatrixCSC<double>>("../data/poisson3Db/poisson3Db.mtx", A);
	// readfile("../data/poisson3Db/poisson3Db_b.mtx", bm);

	std::cout << " A Row: " << A.getRows() << " A Col" << A.getCols() << std::endl;
	// std::cout << "b Row: " << bm.getRows() << "b Col" << bm.getCols() << std::endl;

	// VectorObj<double> b = bm.getColumn(0);
	// // Declare a vector:
	// VectorObj<double> x(b.size(), 0.0);
	// GMRES<double, DenseObj<double>, VectorObj<double>> gmres;
	// gmres.solve(A, b, x, 1000, std::min(A.getRows(), A.getCols()), 1e-6);

	return 0;
}