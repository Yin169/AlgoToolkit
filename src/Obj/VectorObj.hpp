#ifndef VECTOROBJ_HPP
#define VECTOROBJ_HPP

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric> // For std::inner_product
#include <vector>

#include "MatrixObj.hpp"

template <typename TObj>
class VectorObj {
private:
    int _size;
    std::vector<TObj> data;

public:
    // Constructors
    VectorObj() : _size(0), data() {}

    explicit VectorObj(int n) : _size(n), data(n, TObj(0)) {}

    VectorObj(const TObj* other, int n) : _size(n), data(other, other + n) {
        if (!other) throw std::invalid_argument("Null pointer provided to VectorObj constructor.");
    }

    // Copy and Move Semantics
    VectorObj(const VectorObj& other) = default;
    VectorObj(VectorObj&& other) noexcept = default;

    VectorObj& operator=(const VectorObj& other) = default;
    VectorObj& operator=(VectorObj&& other) noexcept = default;

    ~VectorObj() = default;

    // Element Access
    TObj& operator[](int index) {
        if (index < 0 || index >= _size) throw std::out_of_range("Index out of range.");
        return data[index];
    }

    const TObj& operator[](int index) const {
        if (index < 0 || index >= _size) throw std::out_of_range("Index out of range.");
        return data[index];
    }

    int size() const { return _size; }

    // L2 Norm
    double L2norm() const {
        return std::sqrt(std::inner_product(data.begin(), data.end(), data.begin(), 0.0));
    }

    // Normalize
    void normalize() {
        double norm = L2norm();
        if (norm == 0.0) throw std::runtime_error("Cannot normalize a zero vector.");
        for (TObj& val : data) val /= norm;
    }

    // Scalar Operations
    VectorObj operator*(double scalar) const {
        VectorObj result(*this);
        for (TObj& val : result.data) val *= scalar;
        return result;
    }

    VectorObj& operator*=(double scalar) {
        for (TObj& val : data) val *= scalar;
        return *this;
    }

    VectorObj operator/(double scalar) const {
        if (scalar == 0.0) throw std::runtime_error("Division by zero.");
        VectorObj result(*this);
        for (TObj& val : result.data) val /= scalar;
        return result;
    }

    VectorObj& operator/=(double scalar) {
        if (scalar == 0.0) throw std::runtime_error("Division by zero.");
        for (TObj& val : data) val /= scalar;
        return *this;
    }

    // Vector Addition
    VectorObj operator+(const VectorObj& other) const {
        if (_size != other._size) throw std::invalid_argument("Vector dimensions do not match.");
        VectorObj result(_size);
        for (int i = 0; i < _size; ++i) result[i] = data[i] + other[i];
        return result;
    }

    // Vector Subtraction
    VectorObj operator-(const VectorObj& other) const {
        if (_size != other._size) throw std::invalid_argument("Vector dimensions do not match.");
        VectorObj result(_size);
        for (int i = 0; i < _size; ++i) result[i] = data[i] - other[i];
        return result;
    }

    // Dot Product
    TObj operator*(const VectorObj& other) const {
        if (_size != other._size) throw std::invalid_argument("Vector dimensions do not match for dot product.");
        return std::inner_product(data.begin(), data.end(), other.data.begin(), TObj(0));
    }
};

#endif // VECTOROBJ_HPP
