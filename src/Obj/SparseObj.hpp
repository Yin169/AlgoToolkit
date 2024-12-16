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
    SparseMatrixCSC(int rows, int cols) : _n(rows), _m(cols) {
        col_ptr.resize(_m + 1, 0);
    }

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
        ++col_ptr[col + 1]; // Increment column pointer for the next column
    }

    // Get the number of rows
    const int get_row() const { return _n; }

    // Get the number of columns
    const int get_col() const { return _m; }

    // Get a column as a VectorObj
    VectorObj<TObj> get_Col(int index) const {
        if (index < 0 || index >= _m) {
            throw std::out_of_range("Index out of range for column access.");
        }
        TObj* arr = new TObj[_n]();
        for (int i = col_ptr[index]; i < col_ptr[index + 1]; i++) {
            arr[row_indices[i]] = values[i];
        }
        return VectorObj<TObj>(arr, _n);
    }

    // Access an element
    TObj operator()(int row, int col) const {
        if (row < 0 || row >= _n || col < 0 || col >= _m) {
            throw std::out_of_range("Index out of range for element access.");
        }
        for (int i = col_ptr[col]; i < col_ptr[col + 1]; i++) {
            if (row_indices[i] == row) {
                return values[i];
            }
        }
        return TObj(); // Return default value for zero
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
        for (int i = 0; i < _m; ++i) {
            for (int p = col_ptr[i]; p < col_ptr[i + 1]; ++p) {
                int row = row_indices[p];
                TObj val = values[p];
                result.addValue(row, i, val + other(row, i));
            }
        }
        return result;
    }

    // Subtraction operator
    SparseMatrixCSC operator-(const SparseMatrixCSC& other) const {
        if (_n != other._n || _m != other._m) {
            throw std::invalid_argument("Matrices must be the same size for subtraction.");
        }
        SparseMatrixCSC result(_n, _m);
        for (int i = 0; i < _m; ++i) {
            for (int p = col_ptr[i]; p < col_ptr[i + 1]; ++p) {
                int row = row_indices[p];
                TObj val = values[p];
                result.addValue(row, i, val - other(row, i));
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
        for (int i = 0; i < _n; ++i) {
            for (int j = 0; j < other._m; ++j) {
                TObj sum = TObj();
                // Store column entries in a map for quick access
                std::unordered_map<int, TObj> other_row_map;
                for (int k = other.col_ptr[j]; k < other.col_ptr[j + 1]; ++k) {
                    other_row_map[other.row_indices[k]] = other.values[k];
                }
                for (int k = 0; k < _m; ++k) {
                    for (int p = col_ptr[k]; p < col_ptr[k + 1]; ++p) {
                        if (row_indices[p] == i) {
                            auto it = other_row_map.find(k);
                            if (it != other_row_map.end()) {
                                sum += values[p] * it->second;
                            }
                        }
                    }
                }
                if (sum != TObj()) {
                    result.addValue(i, j, sum);
                }
            }
        }
        return result;
    }
};

#endif // SPARSEOBJ_HPP
