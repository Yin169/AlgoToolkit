#ifndef MATRIXOBJ_HPP
#define MATRIXOBJ_HPP

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

template<typename TObj> class VectorObj;

template<typename TObj>
class MatrixObj {
private:
    int _n, _m;
    TObj *arr;

public:
    MatrixObj(){}
    MatrixObj(int n, int m) : _n(n), _m(m), arr(new TObj[n * m]()) {}
    MatrixObj(const TObj *other, int n, int m){
        arr = new TObj[n * m]();
        _n = n;
        _m = m;
        std::copy(other, other + n * m, arr);
    }
    ~MatrixObj() {
        delete[] arr;
    }

    int get_row() const { return _n; }
    int get_col() const { return _m; }

    void setDim(int n, int m) {
        if (n != _n || m != _m) {
            throw std::invalid_argument("Cannot change the dimensions of a MatrixObj instance.");
        }
        _n = n;
        _m = m;
    }

    void Slice(TObj* slice, int n, int m) const {
        if (n < 0 || m < 0 || n + m > _n * _m) {
            throw std::out_of_range("Index out of range in Slice function.");
        }
        std::copy(arr + n, arr + n + m, slice);
    }
    
    MatrixObj &operator=(MatrixObj other) {
        std::swap(this->arr, other.arr);
        std::swap(this->_n, other._n);
        std::swap(this->_m, other._m);
        return *this;
        }

    MatrixObj operator+(const MatrixObj &other) {
        if (_n != other.get_row() || _m != other.get_col()) {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }
        MatrixObj result(arr, _n, _m);
        for (int i = 0; i < _n * _m; ++i) {
            result.arr[i] += other.arr[i];
        }
        
        return result;
    }

    MatrixObj operator-(const MatrixObj &other) {
        if (_n != other.get_row() || _m != other.get_col()) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
        }
        MatrixObj result(arr, _n, _m);
        for (int i = 0; i < _n * _m; ++i) {
            result.arr[i] -= other.arr[i];
        }
        return result;
    }

    inline void scalarMultiple(TObj *arrObj, TObj scalar){
        for (int i = 0; i < _n * _m; ++i) {
            arrObj[i] *= scalar;
        }
    }

    MatrixObj &operator*=(double scalar) {
        scalarMultiple(&(arr[0]), scalar);
        return *this;
    }

    MatrixObj &operator*(double factor) {
        scalarMultiple(&(arr[0]), factor);
        return *this;
    }

    MatrixObj operator*(const MatrixObj &other) {
        if (_m != other.get_row()) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }
        MatrixObj result(_n, other.get_col());
        for (int i = 0; i < _n; ++i) {
            for (int j = 0; j < other.get_col(); ++j) {
                for (int k = 0; k < _m; ++k) {
                    result.arr[i * other.get_col() + j] += arr[i * _n + k] * other.arr[k * other.get_row() + j];
                }
            }
        }
        return result;
    }

    MatrixObj Transpose() {
        MatrixObj result(_m, _n);
        for (int i = 0; i < _n; ++i) {
            for (int j = 0; j < _m; ++j) {
                result.arr[j * _n + i] = arr[i * _m + j];
            }
        }
        return result;
    }

    const TObj &operator[](int index) const {
        if (index < 0 || index >= _n * _m) {
            throw std::out_of_range("Index out of range for matrix element access.");
        }
        return arr[index];
    }

    TObj &operator[](int index) {
        if (index < 0 || index >= _n * _m) {
            throw std::out_of_range("Index out of range for matrix element access.");
        }
        return arr[index];
    }

    VectorObj<TObj> get_Col(int index) const {
        if (index < 0 || index >= _m) {
            throw std::out_of_range("Index out of range for column access.");
        }
        return VectorObj<TObj>(&arr[index * _n], _n);
    }

    explicit MatrixObj(const std::vector<VectorObj<TObj>>& vectors) : _n(vectors.size()), _m(vectors.empty() ? 0 : vectors[0].get_row()) {
        arr = new TObj[_n * _m]();
        for (int i = 0; i < _n; ++i) {
            std::copy(vectors[i].arr, vectors[i].arr + _m, arr + i * _m);
        }
    }

    TObj* data() { return arr; }
    const TObj* data() const { return arr; }

};

template <typename TObj>
class VectorObj : public MatrixObj<TObj> {
 public:
  VectorObj(){}
  VectorObj(int n) : MatrixObj<TObj>(1, n) {}
  VectorObj(const TObj *other, int n) : MatrixObj<TObj>(other, 1, n) {
    if (other == nullptr) {
      throw std::invalid_argument("Null pointer provided to VectorObj constructor.");
    }
  }

  ~VectorObj() {}

  double L2norm() {
    double sum = 0;
    for (int i = 0; i < MatrixObj<TObj>::get_row() * MatrixObj<TObj>::get_col(); i++) {
      sum += std::pow((*this)[i], 2);
    }
    return std::sqrt(sum);
  }

  void normalized() {
    double l2norm = L2norm();
    if (l2norm == 0) {
      throw std::runtime_error("Cannot normalize a zero vector.");
    }
    for (int i = 0; i < MatrixObj<TObj>::get_row() * MatrixObj<TObj>::get_col(); i++) {
      (*this)[i] /= l2norm;
    }
  }

  TObj &operator[](int index) {
    if (index < 0 || index >= this->get_row() * this->get_col()) {
      throw std::out_of_range("Index out of range for vector element access.");
    }
    return MatrixObj<TObj>::data()[index];
  }
  
  TObj operator*(const VectorObj<TObj>& other) const {
    if (this->get_row() * this->get_col() != other.get_row() * other.get_col()) {
        throw std::invalid_argument("Vector dimensions do not match for dot product.");
    }
    TObj sum = static_cast<TObj>(0);
    for (size_t i = 0; i < this->get_row() * this->get_col(); ++i) {
      sum += this->data()[i] * other.data()[i];
    }
    return sum;
  }

  int get_row() const {
      return MatrixObj<TObj>::get_row();
  }
};

#endif // MATRIXOBJ_HPP