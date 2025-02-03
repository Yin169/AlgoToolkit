#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include "Obj/DenseObj.hpp"
#include "Obj/SparseObj.hpp"

namespace utils{

double GenRandom(){
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double random_number = dis(gen);
	return random_number;
}

template <typename T = double, typename MatrixType = DenseObj<T>>
void setMatrixValue(MatrixType& H, int i, int j, T value) {
    H.addValue(i, j, value);
    H.finalize();
}


template <typename T, typename MatrixType>
void readMatrixMarket(std::string filename, MatrixType& matrix) {
    // Open the file:
    std::ifstream fin(filename, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Unable to open" << std::endl;
        return;
    }

    // Declare variables:
    int M, N, L;

    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');

    // Read defining parameters:
    fin >> M >> N >> L;

    std::cout << "M: " << M << " N: " << N << " L: " << L << std::endl;    
    
    // Create your matrix:
    matrix = MatrixType(M, N);
    std::cout << "Matrix created " << matrix.getRows() << " " << matrix.getCols() << std::endl;

    // Pre-allocate vectors for batch processing
    std::vector<int> rows, cols;
    std::vector<T> values;
    rows.reserve(L);
    cols.reserve(L);
    values.reserve(L);

    // Read all data at once
    for (int l = 0; l < L; l++) {
        int m, n;
        T data;
        fin >> m >> n >> data;
        matrix.addValue(m-1, n-1, data);
    }
    matrix.finalize();
    fin.close();

}

} // namespace utils
#endif // UTILS_HPP