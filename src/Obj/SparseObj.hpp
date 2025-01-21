#ifndef SPARSEOBJ_HPP
#define SPARSEOBJ_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include "VectorObj.hpp"

template<typename TObj>
class VectorObj;

template <typename TObj>
class SparseMatrixCSC {
public:
    int _n, _m;
    std::vector<TObj> values;      // Non-zero values
    std::vector<int> row_indices; // Row indices of non-zeros
    std::vector<int> col_ptr;     // Column pointers

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

    SparseMatrixCSC() : _n(0), _m(0) {}
    SparseMatrixCSC(int rows, int cols) : _n(rows), _m(cols), col_ptr(cols + 1, 0) {}
    ~SparseMatrixCSC() = default;

    void addValue(int row, int col, TObj value) {

        const auto end = col_ptr[col + 1];
        auto it = SearchList(row, col);
        if (it != row_indices.begin() + end && *it == row) {
            int index = it - row_indices.begin(); 
            values[index] = value;
            if (construction_buffer[index].row == row_indices[index]){
                construction_buffer[index].value = value;
            }
            return;
        } 

        if (row < 0 || row >= _n || col < 0 || col >= _m) {
            throw std::out_of_range("Index out of range");
        }
        if (value != TObj()) {
            construction_buffer.push_back({row, col, value});
        }
    }

    void finalize() {
        if (construction_buffer.empty()) {
            construction_buffer.clear();
            return;
        }
        std::sort(construction_buffer.begin(), construction_buffer.end());

        col_ptr.assign(_m + 1, 0);
        for (const auto& entry : construction_buffer) {
            ++col_ptr[entry.col + 1];
        }

        std::partial_sum(col_ptr.begin(), col_ptr.end(), col_ptr.begin());

        const int nnz = construction_buffer.size();
        values.resize(nnz);
        row_indices.resize(nnz);

        for (size_t i = 0; i < nnz; ++i) {
            values[i] = construction_buffer[i].value;
            row_indices[i] = construction_buffer[i].row;
        }

        // construction_buffer.clear();
    }

    inline int getRows() const { return _n; }
    inline int getCols() const { return _m; }

    auto SearchList(int row, int col) const{
        if (row < 0 || row >= _n || col < 0 || col >= _m) {
            throw std::out_of_range("Index out of range");
        }

        const auto start = col_ptr[col];
        const auto end = col_ptr[col + 1];

        auto it = std::lower_bound(
            row_indices.begin() + start,
            row_indices.begin() + end,
            row
        );
        return it;   
    }

    TObj operator()(int row, int col) const {
        const auto end = col_ptr[col + 1];
        auto it = SearchList(row, col); 
        if (it != row_indices.begin() + end && *it == row) {
            TObj value = values[it - row_indices.begin()];
            return value;
        }
        return TObj();
    }

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
                #pragma omp critical
                {
                    result[row_indices[idx]] += values[idx] * vec_val;
                }
            }
        }

        return result;
    }

    SparseMatrixCSC operator*(const SparseMatrixCSC& other) const {
        if (_m != other._n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        SparseMatrixCSC result(_n, other._m);

        std::vector<TObj> accumulator(_n);

        for (int col = 0; col < other._m; ++col) {
            std::fill(accumulator.begin(), accumulator.end(), TObj());

            for (int k = other.col_ptr[col]; k < other.col_ptr[col + 1]; ++k) {
                const int row_b = other.row_indices[k];
                const TObj val_b = other.values[k];

                for (int i = col_ptr[row_b]; i < col_ptr[row_b + 1]; ++i) {
                    accumulator[row_indices[i]] += values[i] * val_b;
                }
            }

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
    SparseMatrixCSC applyElementwise(const SparseMatrixCSC& other, Op operation) const {
        if (_n != other._n || _m != other._m) {
            throw std::invalid_argument("Dimension mismatch");
        }

        SparseMatrixCSC result(_n, _m);

        for (int col = 0; col < _m; ++col) {
            int a_pos = col_ptr[col];
            int b_pos = other.col_ptr[col];

            while (a_pos < col_ptr[col + 1] || b_pos < other.col_ptr[col + 1]) {
                int row_a = a_pos < col_ptr[col + 1] ? row_indices[a_pos] : _n;
                int row_b = b_pos < other.col_ptr[col + 1] ? other.row_indices[b_pos] : _n;

                if (row_a == row_b) {
                    TObj value = operation(values[a_pos], other.values[b_pos]);
                    if (value != TObj()) {
                        result.addValue(row_a, col, value);
                    }
                    ++a_pos;
                    ++b_pos;
                } else if (row_a < row_b) {
                    TObj default_b = (b_pos < other.col_ptr[col + 1]) ? TObj() : TObj();
                    result.addValue(row_a, col, operation(values[a_pos], default_b));
                    ++a_pos;
                } else {
                    TObj default_a = (a_pos < col_ptr[col + 1]) ? TObj() : TObj();
                    result.addValue(row_b, col, operation(default_a, other.values[b_pos]));
                    ++b_pos;
                }
            }
        }

        result.finalize();
        return result;
    }

    SparseMatrixCSC operator+(const SparseMatrixCSC& other) const {
        return applyElementwise(other, std::plus<TObj>());
    }

    SparseMatrixCSC operator-(const SparseMatrixCSC& other) const {
        return applyElementwise(other, std::minus<TObj>());
    }

    SparseMatrixCSC& operator*=(double scalar) {
        for (auto& e : values) {
            e *= static_cast<TObj>(scalar);
        }
        return *this;
    }

    SparseMatrixCSC operator*(double scalar) const {
        SparseMatrixCSC result = *this;
        result *= scalar;
        return result;
    }

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
    void swapRows(int row1, int row2){
        if (row1 < 0 || row1 >= _n || row2 < 0 || row2 >= _n) {
            throw std::out_of_range("Index out of range");
        }

        if (row1 == row2) return;

        std::vector<TObj> row1_values;
        std::vector<int> row1_indices;
        for (int i = 0; i < _m; ++i) {
            auto it = SearchList(row1, i);
            if (it != row_indices.begin() + col_ptr[i + 1] && *it == row1) {
                row1_values.push_back(values[it - row_indices.begin()]);
            } else {
                row1_values.push_back(TObj());
            }
            row1_indices.push_back(i);
        }

        std::vector<TObj> row2_values;
        std::vector<int> row2_indices;
        for (int i = 0; i < _m; ++i) {
            auto it = SearchList(row2, i);
            if (it != row_indices.begin() + col_ptr[i + 1] && *it == row2) {
                row2_values.push_back(values[it - row_indices.begin()]);
            } else {
                row2_values.push_back(TObj());
            }
            row2_indices.push_back(i);
        }

        for (int i = 0; i < _m; ++i) {
            if (row1_values[i] != TObj()) {
                addValue(row2, row1_indices[i], row1_values[i]);
            }
            if (row2_values[i] != TObj()) {
                addValue(row1, row2_indices[i], row2_values[i]);
            }
        }

        finalize();
    }
};

#endif // SPARSEOBJ_HPP