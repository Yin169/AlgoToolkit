#ifndef SPARSEOBJ_HPP
#define SPARSEOBJ_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#include "VectorObj.hpp"

template<typename TObj>
class VectorObj;

template <typename TObj>
class SparseMatrixCSC {
public:
    int _n, _m;
    std::vector<TObj> values;      // Non-zero values
    std::vector<int> row_indices;  // Row indices of non-zeros
    std::vector<int> col_ptr;      // Column pointers
    
    // Helper struct for construction
    struct Entry {
        int row;
        int col;
        TObj value;
        
        bool operator<(const Entry& other) const {
            return col < other.col || (col == other.col && row < other.row);
        }
    };
    
    std::vector<Entry> construction_buffer;

    SparseMatrixCSC() = default;
    SparseMatrixCSC(int rows, int cols) : _n(rows), _m(cols), col_ptr(cols + 1, 0) {}
    ~SparseMatrixCSC() = default;

    // Add value during construction phase
    void addValue(int row, int col, TObj value) {
        if (row < 0 || row >= _n || col < 0 || col >= _m) {
            throw std::out_of_range("Index out of range");
        }
        if (value != TObj()) {  // Only store non-zero values
            construction_buffer.push_back({row, col, value});
        }
    }

    // Finalize the matrix structure
    void finalize() {
        // Sort entries by column, then row
        std::sort(construction_buffer.begin(), construction_buffer.end());
        
        // Count entries per column
        col_ptr.assign(_m + 1, 0);
        for (const auto& entry : construction_buffer) {
            ++col_ptr[entry.col + 1];
        }
        
        // Compute column pointers
        std::partial_sum(col_ptr.begin(), col_ptr.end(), col_ptr.begin());
        
        // Allocate final arrays
        const int nnz = construction_buffer.size();
        values.reserve(nnz);
        row_indices.reserve(nnz);
        
        // Fill arrays
        for (const auto& entry : construction_buffer) {
            values.push_back(entry.value);
            row_indices.push_back(entry.row);
        }
        
        // Clear construction buffer
        std::vector<Entry>().swap(construction_buffer);
    }

    inline int getRows() const { return _n; }
    inline int getCols() const { return _m; }

    // Optimized element access
    TObj operator()(int row, int col) const {
        if (row < 0 || row >= _n || col < 0 || col >= _m) {
            throw std::out_of_range("Index out of range");
        }
        
        const auto start = col_ptr[col];
        const auto end = col_ptr[col + 1];
        
        // Binary search for row index
        auto it = std::lower_bound(
            row_indices.begin() + start,
            row_indices.begin() + end,
            row
        );
        
        if (it != row_indices.begin() + end && *it == row) {
            return values[it - row_indices.begin()];
        }
        return TObj();
    }

    // Optimized sparse matrix-vector multiplication
    VectorObj<TObj> operator*(const VectorObj<TObj>& vec) const {
        if (_m != vec.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        
        VectorObj<TObj> result(_n, TObj());
        
        #pragma omp parallel for schedule(dynamic)
        for (int col = 0; col < _m; ++col) {
            const TObj vec_val = vec[col];
            if (vec_val == TObj()) continue;
            
            for (int idx = col_ptr[col]; idx < col_ptr[col + 1]; ++idx) {
                #pragma omp atomic
                result[row_indices[idx]] += values[idx] * vec_val;
            }
        }
        
        return result;
    }

    // Optimized sparse matrix-matrix multiplication
    SparseMatrixCSC operator*(const SparseMatrixCSC& other) const {
        if (_m != other._n) {
            throw std::invalid_argument("Dimension mismatch");
        }
        
        SparseMatrixCSC result(_n, other._m);
        
        // Use temporary dense accumulator for each column
        std::vector<TObj> accumulator(_n);
        
        for (int col = 0; col < other._m; ++col) {
            std::fill(accumulator.begin(), accumulator.end(), TObj());
            
            // Multiply column of B with all columns of A
            for (int k = other.col_ptr[col]; k < other.col_ptr[col + 1]; ++k) {
                const int row_b = other.row_indices[k];
                const TObj val_b = other.values[k];
                
                for (int i = col_ptr[row_b]; i < col_ptr[row_b + 1]; ++i) {
                    accumulator[row_indices[i]] += values[i] * val_b;
                }
            }
            
            // Add non-zero results to output matrix
            for (int i = 0; i < _n; ++i) {
                if (accumulator[i] != TObj()) {
                    result.addValue(i, col, accumulator[i]);
                }
            }
        }
        
        result.finalize();
        return result;
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
        for (auto& e : values){ 
            e *= static_cast<TObj>(scalar);
        }
        return *this; 
    }

    SparseMatrixCSC operator*(double scalar){
        SparseMatrixCSC result = *this;
        result *= scalar;
        return result;
    }

    // Efficient transpose operation
    SparseMatrixCSC Transpose() const {
        SparseMatrixCSC result(_m, _n);
        
        for (int col = 0; col < _m; ++col) {
            for (int idx = col_ptr[col]; idx < col_ptr[col + 1]; ++idx) {
                result.addValue(col, row_indices[idx], values[idx]);
            }
        }
        
        result.finalize();
        return result;
    }
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
};

#endif // SPARSEOBJ_HPP