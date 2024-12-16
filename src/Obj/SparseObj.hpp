#ifndef SPARSEOBJ_HPP
#define SPARSEOBJ_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <unordered_map>

template<typename TObj>
class VectorObj;

template <typename TObj>
class SparseMatrixCSC {
public:
    int _n, _m; // Number of rows and columns
    std::vector<TObj> values;      // Non-zero values
    std::vector<int> row_indices;  // Row indices of non-zeros
    std::vector<int> col_ptr;      // Column pointers

    // Constructor
    SparseMatrixCSC(int rows, int cols) : _n(rows), _m(cols), col_ptr(cols + 1, 0) {}

    // Destructor
    ~SparseMatrixCSC() = default;

    // Add a value to the sparse matrix
    void addValue(int row, int col, TObj value) {
        if (col >= _m || row >= _n) {
            throw std::out_of_range("Row or column index out of range.");
        }
        if (value == TObj()) return; // Skip zero values
        values.push_back(value);
        row_indices.push_back(row);
        ++col_ptr[col + 1];
    }

    // Finalize column pointers after all values are added
    void finalize() {
        for (int col = 1; col < col_ptr.size(); ++col) {
            col_ptr[col] += col_ptr[col - 1];
        }
    }

    // Get the number of rows
    inline int getRows() const { return _n; }

    // Get the number of columns
    inline int getCols() const { return _m; }

    // Get a column as a VectorObj
    VectorObj<TObj> getColumn(int index) const {
        if (index < 0 || index >= _m) {
            throw std::out_of_range("Index out of range for column access.");
        }
        VectorObj<TObj> colData(_n);
        for (int i = col_ptr[index]; i < col_ptr[index + 1]; ++i) {
            colData[row_indices[i]] = values[i];
        }
        return colData;
    }

    // Access an element
    TObj operator()(int row, int col) const {
        if (row < 0 || row >= _n || col < 0 || col >= _m) {
            throw std::out_of_range("Index out of range for element access.");
        }
        for (int i = col_ptr[col]; i < col_ptr[col + 1]; ++i) {
            if (row_indices[i] == row) {
                return values[i];
            }
        }
        return TObj(); // Default value if not found
    }

    // Copy assignment operator
    SparseMatrixCSC& operator=(const SparseMatrixCSC& other) {
        if (this != &other) {
            _n = other._n;
            _m = other._m;
            values = other.values;
            row_indices = other.row_indices;
            col_ptr = other.col_ptr;
        }
        return *this;
    }

    // Addition operator
    SparseMatrixCSC operator+(const SparseMatrixCSC& other) const {
        if (_n != other._n || _m != other._m) {
            throw std::invalid_argument("Matrices must be the same size for addition.");
        }
        SparseMatrixCSC result(_n, _m);
        result.values.reserve(values.size() + other.values.size());
        result.row_indices.reserve(row_indices.size() + other.row_indices.size());
        result.col_ptr = col_ptr;

        // Merge-like approach
        for (int col = 0; col < _m; ++col) {
            int a_pos = col_ptr[col], b_pos = other.col_ptr[col];
            while (a_pos < col_ptr[col + 1] || b_pos < other.col_ptr[col + 1]) {
                if (a_pos < col_ptr[col + 1] && (b_pos >= other.col_ptr[col + 1] || row_indices[a_pos] < other.row_indices[b_pos])) {
                    result.addValue(row_indices[a_pos], col, values[a_pos]);
                    ++a_pos;
                } else if (b_pos < other.col_ptr[col + 1] && (a_pos >= col_ptr[col + 1] || row_indices[a_pos] > other.row_indices[b_pos])) {
                    result.addValue(other.row_indices[b_pos], col, other.values[b_pos]);
                    ++b_pos;
                } else {
                    result.addValue(row_indices[a_pos], col, values[a_pos] + other.values[b_pos]);
                    ++a_pos;
                    ++b_pos;
                }
            }
        }
        return result;
    }

    // Multiplication operator
    SparseMatrixCSC operator*(const SparseMatrixCSC& other) const {
        if (_m != other._n) {
            throw std::invalid_argument("Number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication.");
        }
        SparseMatrixCSC result(_n, other._m);
        result.values.reserve(_n * other._m);

        // Compressed multiplication logic
        for (int col = 0; col < other._m; ++col) {
            std::vector<TObj> temp(_n, TObj());
            for (int k = other.col_ptr[col]; k < other.col_ptr[col + 1]; ++k) {
                int row_b = other.row_indices[k];
                TObj value_b = other.values[k];
                for (int p = col_ptr[row_b]; p < col_ptr[row_b + 1]; ++p) {
                    temp[row_indices[p]] += values[p] * value_b;
                }
            }
            for (int row = 0; row < _n; ++row) {
                if (temp[row] != TObj()) {
                    result.addValue(row, col, temp[row]);
                }
            }
        }
        return result;
    }
};

#endif // SPARSEOBJ_HPP
