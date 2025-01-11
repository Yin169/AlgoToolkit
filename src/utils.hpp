#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <algorithm>
#include "DenseObj.hpp"
#include "SparseObj.hpp"

void readfile(std::string filename, DenseObj<double>& matrix) {
	// Open the file:
	std::ifstream fin(filename);

	// Declare variables:
	int M, N, L;

	// Ignore headers and comments:
	while (fin.peek() == '%') fin.ignore(2048, '\n');

	// Read defining parameters:
	fin >> M >> N >> L;

	// Create your matrix:

	matrix = DenseObj<double>(M, N);

	// Read the data
	for (int l = 0; l < L; l++)
	{
		int m, n;
		double data;
		fin >> m >> n >> data;
		matrix(m-1, n-1) = data;
	}

	fin.close();
}


#endif // UTILS_HPP