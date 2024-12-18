#ifndef MATRIXOBJ_HPP
#define MATRIXOBJ_HPP

#include "DenseObj.hpp"
#include "SparseObj.hpp"
#include "VectorObj.hpp"

template <typename TObj>
class VectorObj;

template <typename TObj>
class DenseObj;

template <typename TObj>
class SparseMatrixCSC;

template <typename TObj>
class MatrixObj {
public:
    MatrixObj() = default;
    MatrixObj(const MatrixObj& other) = default;
    virtual ~MatrixObj() = default;

    virtual int getRows() const = 0;
    virtual int getCols() const = 0;
    virtual VectorObj<TObj> getColumn(int index) const = 0;
};

#endif