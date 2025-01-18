#ifndef DENSEOBJ_HPP
#define DENSEOBJ_HPP

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <iostream>
// #include <openblas/cblas.h>
#include "cblas.h"

#include "MatrixObj.hpp"
#include "VectorObj.hpp"

template<typename TObj> 
class VectorObj;

template<typename TObj> 
class MatrixObj;

template<typename TObj>
class DenseObj : public MatrixObj<TObj> {
private:
    int _n; // Number of rows
    int _m; // Number of columns
    std::vector<TObj> arr; // Stores matrix elements in column-major order

public:
    // Constructors
    DenseObj() : _n(0), _m(0), arr() {}

    DenseObj(int n, int m) : _n(n), _m(m), arr(n * m, TObj(0)) {}

    DenseObj(const VectorObj<TObj>& data, int n, int m) 
        : _n(n), _m(m), arr(std::vector<TObj>(data.element(), data.element() + n * m)) {
        if (data.size() != n * m) {
            throw std::invalid_argument("Data size does not match matrix dimensions.");
        }
    }

    DenseObj(const std::vector<VectorObj<TObj>>& data, int n, int m) 
        : _n(n), _m(m) {
        if (data.size() != m) {
            throw std::invalid_argument("Data size does not match matrix dimensions.");
        }
        for (int i = 0; i < data.size(); ++i){
            arr.insert(arr.end(), data[i].element(), data[i].element() + n);
        }
    }

    DenseObj(const std::vector<TObj>& data, int n, int m) 
        : _n(n), _m(m), arr(data) {
        if (data.size() != n * m) {
            throw std::invalid_argument("Data size does not match matrix dimensions.");
        }
    }

    // Copy Constructor
    DenseObj(const DenseObj& other) = default;

    // Move Constructor
    DenseObj(DenseObj&& other) noexcept = default;

    // Copy Assignment
    DenseObj& operator=(const DenseObj& other) = default;

    // Move Assignment
    DenseObj& operator=(DenseObj&& other) noexcept = default;

    // Destructor
    ~DenseObj() = default;

    // Accessors
    TObj& operator[](int index) {
        if ( index < 0 || index >= _n * _m ) {
            throw std::out_of_range("Matrix indices are out of range.");
        }
        return arr[index];
    }
    const TObj& operator[](int index) const {
        if ( index < 0 || index >= _n * _m ) {
            throw std::out_of_range("Matrix indices are out of range.");
        }
        return arr[index];
    }

    TObj& operator()(int row, int col) {
        if (row < 0 || col < 0 || row >= _n || col >= _m) {
            throw std::out_of_range("Matrix indices are out of range.");
        }
        return arr[row + col * _n];
    }

    const TObj& operator()(int row, int col) const {
        if (row < 0 || col < 0 || row >= _n || col >= _m) {
            throw std::out_of_range("Matrix indices are out of range.");
        }
        return arr[row + col * _n];
    }

    TObj* data() { return arr.data(); }
    const TObj* data() const { return arr.data(); }

    inline int getRows() const override { return _n; }
    inline int getCols() const override { return _m; }

    // Scalar multiplication
    DenseObj& operator*=(TObj scalar) {
        for (TObj& value : arr) {
            value *= scalar;
        }
        return *this;
    }

    DenseObj operator*(TObj scalar) const {
        DenseObj result = *this;
        result *= scalar;
        return result;
    }

    // Matrix addition and subtraction
    DenseObj operator+(const DenseObj& other) const {
        return elementWiseOperation(other, std::plus<TObj>());
    }

    DenseObj operator-(const DenseObj& other) const {
        return elementWiseOperation(other, std::minus<TObj>());
    }

    // Element-wise operations
    template <typename Op>
    DenseObj elementWiseOperation(const DenseObj& other, Op operation) const {
        if (_n != other._n || _m != other._m) {
            throw std::invalid_argument("Matrix dimensions do not match for operation.");
        }
        DenseObj result(_n, _m);
        std::transform(arr.begin(), arr.end(), other.arr.begin(), result.arr.begin(), operation);
        return result;
    }

    // Matrix multiplication
    DenseObj operator*(const DenseObj& other) const {
        if (_m != other._n) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }
        DenseObj result(_n, other._m);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    _n, other._m, _m, 1.0,
                    arr.data(), _n,
                    other.arr.data(), other._n,
                    0.0, result.arr.data(), _n);
        return result;
    }

    // Matrix-vector multiplication
    VectorObj<TObj> operator*(const VectorObj<TObj>& vec) const {
        if (_m != static_cast<int>(vec.size())) {
            throw std::invalid_argument("Vector size does not match matrix columns.");
        }
        VectorObj<TObj> result(_n);
        cblas_dgemv(CblasColMajor, CblasNoTrans,
                    _n, _m, 1.0,
                    arr.data(), _n,
                    vec.element(), 1,
                    0.0, const_cast<TObj*>(result.element()), 1);
        return result;
    }

    void swapRows(int row1, int row2){
        for (int j = 0; j < _m; ++j){
            std::swap((*this)(row1, j), (*this)(row2, j));
        }
    }

    // Transpose
    DenseObj Transpose() const {
        DenseObj result(_m, _n);
        #pragma omp parallel for
        for (int i = 0; i < _n; ++i) {
            for (int j = 0; j < _m; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // Get a specific column
    VectorObj<TObj> getColumn(int index) const override {
        if (index < 0 || index >= _m) {
            throw std::out_of_range("Column index is out of range.");
        }
        VectorObj<TObj> column(_n);
        for (int i = 0; i < _n; ++i) {
            column[i] = (*this)(i, index);
        }
        return column;
    }

    inline int size(){ return arr.size(); }

    void zero(){
        arr.clear();
        arr.resize(_n*_m);
    }    

    // Resize matrix
    void resize(int n, int m) {
        _n = n;
        _m = m;
        arr.resize(n * m);
    }

};

#endif // MATRIXOBJ_HPP
