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
    MatrixObj(int n, int m) : _n(n), _m(m) {
        arr = new TObj[_n * _m]();
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
    }

    void Slice(TObj* slice, int n, int m) const {
        if (n < 0 || m < 0 || n + m > _n * _m) {
            throw std::out_of_range("Index out of range in Slice function.");
        }
        std::copy(arr + n, arr + n + m, slice);
    }

    MatrixObj(const MatrixObj &other) : _n(other.get_row()), _m(other.get_col()) {
        arr = new TObj[_n * _m];
        std::copy(&other[0], &other[_n * _m], arr);
    }

    MatrixObj &operator=(const MatrixObj &other) {
        if (this != &other) {
            if (_n != other.get_row() || _m != other.get_col()) {
                throw std::invalid_argument("Matrix dimensions do not match for assignment.");
            }
            std::copy(&other[0], &other[_n * _m], arr);
        }
        return *this;
    }

    MatrixObj operator+(const MatrixObj &other) {
        if (_n != other.get_row() || _m != other.get_col()) {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }
        MatrixObj result(*this);
        for (int i = 0; i < _n * _m; ++i) {
            result.arr[i] += other.arr[i];
        }
        return result;
    }

    MatrixObj operator-(const MatrixObj &other) {
        if (_n != other.get_row() || _m != other.get_col()) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
        }
        MatrixObj result(*this);
        for (int i = 0; i < _n * _m; ++i) {
            result.arr[i] -= other.arr[i];
        }
        return result;
    }

    MatrixObj &operator*(double factor) {
        for (size_t i = 0; i < _n * _m; ++i) {
            arr[i] *= factor;
        }
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
                    result.arr[i * other.get_col() + j] += arr[i * _m + k] * other.arr[k * other.get_col() + j];
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
        return VectorObj<TObj>(&arr[index * _n], 1, _n);
    }

    explicit MatrixObj(const std::vector<VectorObj<TObj>>& vectors) : _n(vectors.size()), _m(vectors.empty() ? 0 : vectors[0].get_row()) {
        arr = new TObj[_n * _m]();
        for (int i = 0; i < _n; ++i) {
            std::copy(vectors[i].arr, vectors[i].arr + _m, arr + i * _m);
        }
    }
};

template <typename TObj>
class VectorObj : public MatrixObj<TObj> {
 public:
  VectorObj(const TObj *other, int n) : MatrixObj<TObj>(other, 1, n) {
    if (other == nullptr) {
      throw std::invalid_argument("Null pointer provided to VectorObj constructor.");
    }
  }

  ~VectorObj() {}

  double L2norm() {
    double sum = 0;
    for (int i = 0; i < MatrixObj<TObj>::get_row() * MatrixObj<TObj>::get_col(); i++) {
      sum += std::pow(MatrixObj<TObj>::arr[i], 2);
    }
    return std::sqrt(sum);
  }

  void normalized() {
    double l2norm = L2norm();
    if (l2norm == 0) {
      throw std::runtime_error("Cannot normalize a zero vector.");
    }
    for (int i = 0; i < MatrixObj<TObj>::get_row() * MatrixObj<TObj>::get_col(); i++) {
      MatrixObj<TObj>::arr[i] /= l2norm;
    }
  }

  TObj operator[](int index) const {
    if (index < 0 || index >= MatrixObj<TObj>::get_row() * MatrixObj<TObj>::get_col()) {
      throw std::out_of_range("Index out of range for vector element access.");
    }
    return MatrixObj<TObj>::arr[index];
  }

  int get_row() const {
      return MatrixObj<TObj>::get_row();
  }
};

#endif // MATRIXOBJ_HPP