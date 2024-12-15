#ifndef VECTOROBJ_HPP
#define VECTOROBJ_HPP

#include <cstring>
#include <stdexcept>
#include <vector>

#include "MatrixObj.hpp"


template<typename TObj> class MatrixObj;
template <typename TObj>
class VectorObj : public MatrixObj<TObj> {
    public:
    VectorObj(){}
    VectorObj(int n) : MatrixObj<TObj>(n, 1) {}
    VectorObj(const TObj *other, int n) : MatrixObj<TObj>(other, n, 1) {
        if (other == nullptr) {
            throw std::invalid_argument("Null pointer provided to VectorObj constructor.");
        }
    }
    
    ~VectorObj() {}
    
    double L2norm() {
        double sum = 0;
        for (int i = 0; i < MatrixObj<TObj>::get_row() * MatrixObj<TObj>::get_col(); i++) {
            sum += std::pow(this->data()[i], 2);
        }
        return std::sqrt(sum);
    }
    
    void normalized() {
        double l2norm = L2norm();
        if (l2norm == 0) {
            throw std::runtime_error("Cannot normalize a zero vector.");
        }
        for (int i = 0; i < MatrixObj<TObj>::get_row() * MatrixObj<TObj>::get_col(); i++) {
            this->data()[i] = this->data()[i] / l2norm;
        }
    }
    
    VectorObj operator+(const VectorObj<TObj> &other) {
        int n = other.get_row(), m = other.get_col();
        if (this->get_row() != n || this->get_col() != m) {
            throw std::invalid_argument("Vector dimensions do not match for addition.");
        }
        VectorObj<TObj> result(n);
        for (int i = 0; i < n * m ; ++i) {
            result.data()[i] = this->data()[i] + other.data()[i];
        }
        return result;
    }

    VectorObj operator-(const VectorObj<TObj> &other) {
        int n = other.get_row(), m = other.get_col();
        if (this->get_row() != n || this->get_col() != m) {
            throw std::invalid_argument("Vector dimensions do not match for subtraction.");
        }
        VectorObj<TObj> result(n);
        for (int i = 0; i < n * m ; ++i) {
            result.data()[i] = this->data()[i] - other.data()[i];
        }
        return result;
    }
    
    VectorObj &operator*=(double scalar) {
        MatrixObj<TObj>::scalarMultiple(&(MatrixObj<TObj>::data()[0]), static_cast<TObj>(scalar));
        return *this;
    }

    VectorObj &operator*(double scalar) {
        MatrixObj<TObj>::scalarMultiple(&(MatrixObj<TObj>::data()[0]), static_cast<TObj>(scalar));
        return *this;
    }

    VectorObj &operator/(double scalar) {
        MatrixObj<TObj>::scalarMultiple(&(MatrixObj<TObj>::data()[0]), static_cast<TObj>(1/scalar));
        return *this;
    }

    VectorObj &operator=(VectorObj other) {
        this->swap(other);
        return *this;
    }
    
    TObj &operator[](int index) {
        if (index < 0 || index >= this->get_row() * this->get_col()) {
            throw std::out_of_range("Index out of range for vector element access.");
        }
        return MatrixObj<TObj>::data()[index];
    }
    

    TObj operator*(const VectorObj<TObj>& other) const {
        if (this->get_row() != other.get_row()) {
            throw std::invalid_argument("Vector dimensions do not match for dot product.");
        }
        TObj sum = static_cast<TObj>(0);
        for (int i = 0; i < this->get_row(); ++i) {
            sum += this->data()[i] * other.data()[i];
        }
        return sum;
    }
    
    int get_col() const {
        return MatrixObj<TObj>::get_col();
    }
    
    int get_row() const {
        return MatrixObj<TObj>::get_row();
    }
};

#endif