#ifndef SPARSEOBJ_HPP
#define SPARSEOBJ_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <algorithm> // for std::lower_bound
#include <numeric>   // for std::accumulate

#include "MatrixObj.hpp"
#include "VectorObj.hpp"

template<typename TObj>
class VectorObj;

template <typename TObj>
class SparseMatrixCSC : public MatrixObj<TObj> {
public:
    int _n, _m; // Number of rows and columns
    std::vector<TObj> values;      // Non-zero values
    std::vector<int> row_indices;  // Row indices of non-zeros
    std::vector<int> col_ptr;      // Column pointers

    // Constructor
    SparseMatrixCSC(int rows, int cols) : _n(rows), _m(cols), col_ptr(cols + 1, 0) {}

    ~SparseMatrixCSC() = default;

    // Add a value to the sparse matrix
    void addValue(int row, int col, TObj value) {
        if (col >= _m || row >= _n) throw std::out_of_range("Row or column index out of range.");
        if (value == TObj()) return; // Skip zero values
        values.push_back(value);
        row_indices.push_back(row);
        ++col_ptr[col + 1];
    }

    // Finalize column pointers after all values are added
    void finalize() {
        std::partial_sum(col_ptr.begin(), col_ptr.end(), col_ptr.begin());
    }

    inline int getRows() const { return _n; }
    inline int getCols() const { return _m; }

    // Get a specific column
    VectorObj<TObj> getColumn(int index) const {
        if (index < 0 || index >= _m) {
            throw std::out_of_range("Column index is out of range.");
        }
        VectorObj<TObj> column(_n);
        for (int i = 0; i < _n; ++i) {
            column[i] = (*this)(i, index);
        }
        return column;
    }

    // Element access with binary search
    TObj operator()(int row, int col) const {
        if (row < 0 || row >= _n || col < 0 || col >= _m) throw std::out_of_range("Index out of range.");
        auto start = col_ptr[col];
        auto end = col_ptr[col + 1];
        auto it = std::lower_bound(row_indices.begin() + start, row_indices.begin() + end, row);
        if (it != row_indices.begin() + end && *it == row) {
            return values[it - row_indices.begin()];
        }
        return TObj(); // Default value
    }

    template <typename Op>
    SparseMatrixCSC Operator(const SparseMatrixCSC& other, Op operation) const {
        if (_n != other._n || _m != other._m) throw std::invalid_argument("Matrices must be the same size for addition.");
        SparseMatrixCSC result(_n, _m);

        for (int col = 0; col < _m; ++col) {
            int a_pos = col_ptr[col], b_pos = other.col_ptr[col];
            std::unordered_map<int, TObj> col_data;

            while (a_pos < col_ptr[col + 1]) col_data[row_indices[a_pos++]] = operation(col_data[row_indices[a_pos]], values[a_pos]);
            while (b_pos < other.col_ptr[col + 1]) col_data[other.row_indices[b_pos++]] = operation(col_data[other.row_indices[b_pos]], other.values[b_pos]);

            for (const auto& [row, value] : col_data) {
                if (value != TObj()) result.addValue(row, col, value);
            }
        }
        result.finalize();
        return result;
    }

    // Optimized addition
    SparseMatrixCSC operator+(const SparseMatrixCSC& other) const {
        return Operator(other, std::plus<TObj>());
    }

    SparseMatrixCSC operator-(const SparseMatrixCSC& other) const {
        return Operator(other, std::minus<TObj>());
    }

    SparseMatrixCSC &operator*=(double scalar) {
        for (auto& e : value){ 
            e *= static_cast<TObj>(scalar);
        }
        return *this 
    }

    SparseMatrixCSC operator*(double scalar){
        SparseMatrixCSC result = *this;
        result *= scalar;
        return result 
    }

    VectorObj<TObj> operator*(const VectorObj<TObj>& vector) {
        // Check dimension compatibility
        if (_m != vector.size()) {
            throw std::invalid_argument("Sparse matrix columns must match vector size.");
        }

        // Result vector initialization
        VectorObj<TObj> result(_n, TObj(0));

        // Perform sparse matrix-vector multiplication
        for (int col = 0; col < _m; ++col) {
            TObj vector_value = vector[col];
            if (vector_value == TObj()) continue; // Skip multiplication by zero

            // Iterate over non-zero entries in the column
            for (int idx = col_ptr[col]; idx < col_ptr[col + 1]; ++idx) {
                int row = row_indices[idx];
                result[row] += values[idx] * vector_value;
            }
        }
        return result;
    }


    // Optimized multiplication using CSC * CSR
    SparseMatrixCSC operator*(const SparseMatrixCSC& other) const {
        if (_m != other._n) throw std::invalid_argument("Matrix dimensions incompatible for multiplication.");
        SparseMatrixCSC result(_n, other._m);

        std::vector<std::vector<std::pair<int, TObj>>> rows(_n);
        for (int col = 0; col < other._m; ++col) {
            for (int k = other.col_ptr[col]; k < other.col_ptr[col + 1]; ++k) {
                int row_b = other.row_indices[k];
                TObj value_b = other.values[k];

                for (int p = col_ptr[row_b]; p < col_ptr[row_b + 1]; ++p) {
                    rows[row_indices[p]].emplace_back(col, values[p] * value_b);
                }
            }
        }

        for (int row = 0; row < _n; ++row) {
            std::unordered_map<int, TObj> temp_map;
            for (const auto& [col, value] : rows[row]) {
                temp_map[col] += value;
            }
            for (const auto& [col, value] : temp_map) {
                if (value != TObj()) result.addValue(row, col, value);
            }
        }

        result.finalize();
        return result;
    }
};

#endif // SPARSEOBJ_HPP
