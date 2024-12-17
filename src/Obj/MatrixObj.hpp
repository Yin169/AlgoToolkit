#ifndef MATRIXOBJ_HPP
#define MATRIXOBJ_HPP

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <iostream>
#include <openblas/cblas.h> // Ensure BLAS is installed or use a wrapper library like Eigen

template<typename TObj>
class MatrixObj {
private:
    int _n; // Number of rows
    int _m; // Number of columns
    std::vector<TObj> arr; // Stores matrix elements in column-major order

public:
    // Constructors
    MatrixObj() : _n(0), _m(0), arr() {}

    MatrixObj(int n, int m) : _n(n), _m(m), arr(n * m, TObj(0)) {}

    MatrixObj(const std::vector<TObj>& data, int n, int m) 
        : _n(n), _m(m), arr(data) {
        if (data.size() != n * m) {
            throw std::invalid_argument("Data size does not match matrix dimensions.");
        }
    }

    // Copy Constructor
    MatrixObj(const MatrixObj& other) = default;

    // Move Constructor
    MatrixObj(MatrixObj&& other) noexcept = default;

    // Copy Assignment
    MatrixObj& operator=(const MatrixObj& other) = default;

    // Move Assignment
    MatrixObj& operator=(MatrixObj&& other) noexcept = default;

    // Destructor
    ~MatrixObj() = default;

    // Accessors
    TObj& operator()(size_t row, size_t col) {
        if (row >= _n || col >= _m) {
            throw std::out_of_range("Matrix indices are out of range.");
        }
        return arr[row + col * _n];
    }

    const TObj& operator()(size_t row, size_t col) const {
        if (row >= _n || col >= _m) {
            throw std::out_of_range("Matrix indices are out of range.");
        }
        return arr[row + col * _n];
    }

    TObj* data() { return arr.data(); }
    const TObj* data() const { return arr.data(); }

    inline int getRows() const { return _n; }
    inline int getCols() const { return _m; }

    // Scalar multiplication
    MatrixObj& operator*=(TObj scalar) {
        for (TObj& value : arr) {
            value *= scalar;
        }
        return *this;
    }

    MatrixObj operator*(TObj scalar) const {
        MatrixObj result = *this;
        result *= scalar;
        return result;
    }

    // Matrix addition and subtraction
    MatrixObj operator+(const MatrixObj& other) const {
        return elementWiseOperation(other, std::plus<TObj>());
    }

    MatrixObj operator-(const MatrixObj& other) const {
        return elementWiseOperation(other, std::minus<TObj>());
    }

    // Element-wise operations
    template <typename Op>
    MatrixObj elementWiseOperation(const MatrixObj& other, Op operation) const {
        if (_n != other._n || _m != other._m) {
            throw std::invalid_argument("Matrix dimensions do not match for operation.");
        }
        MatrixObj result(_n, _m);
        std::transform(arr.begin(), arr.end(), other.arr.begin(), result.arr.begin(), operation);
        return result;
    }

    // Matrix multiplication
    MatrixObj operator*(const MatrixObj& other) const {
        if (_m != other._n) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }
        MatrixObj result(_n, other._m);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    _n, other._m, _m, 1.0,
                    arr.data(), _n,
                    other.arr.data(), other._n,
                    0.0, result.arr.data(), _n);
        return result;
    }

    // Matrix-vector multiplication
    std::vector<TObj> operator*(const std::vector<TObj>& vec) const {
        if (_m != static_cast<int>(vec.size())) {
            throw std::invalid_argument("Vector size does not match matrix columns.");
        }
        std::vector<TObj> result(_n, TObj(0));
        cblas_dgemv(CblasColMajor, CblasNoTrans,
                    _n, _m, 1.0,
                    arr.data(), _n,
                    vec.data(), 1,
                    0.0, result.data(), 1);
        return result;
    }

    // Transpose
    MatrixObj Transpose() const {
        MatrixObj result(_m, _n);
        #pragma omp parallel for
        for (int i = 0; i < _n; ++i) {
            for (int j = 0; j < _m; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // Get a specific column
    std::vector<TObj> getColumn(int index) const {
        if (index < 0 || index >= _m) {
            throw std::out_of_range("Column index is out of range.");
        }
        std::vector<TObj> column(_n);
        for (int i = 0; i < _n; ++i) {
            column[i] = (*this)(i, index);
        }
        return column;
    }

    // Resize matrix
    void resize(int n, int m) {
        _n = n;
        _m = m;
        arr.resize(n * m);
    }

};

#endif // MATRIXOBJ_HPP
