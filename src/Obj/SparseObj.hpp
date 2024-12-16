#include <iostream>
#include <vector>

#include "VectorObj.hpp"

template<typename TObj> class VectorObj;
template <typename TObj>
class SparseMatrixCSC {
public:
    int _n, _m;
    std::vector<TObj> values;      // Non-zero values
    std::vector<int> row_indices;    // Row indices of non-zeros
    std::vector<int> col_ptr;        // Column pointers

    SparseMatrixCSC(int rows, int cols) : _n(rows), _m(cols) { col_ptr.push_back(0); }
	~SparseMatrixCSC() = default;

    void addValue(int row, int col, TObj value) {
        values.push_back(value);
        row_indices.push_back(row);
        if (col_ptr.size() <= col + 1) {
            col_ptr.push_back(values.size());
        } else {
            col_ptr[col + 1] = values.size();
        }
    }    
	const int get_row() const { return _n; }
    const int get_col() const { return _m; }

    VectorObj<TObj> get_Col(int index) const {
        if (index < 0 || index >= _m) { throw std::out_of_range("Index out of range for column access."); }
		TObj arr[_n](0);
		int ptr = col_ptr[index];
		
		while(ptr < col_ptr[index + 1]){
			arr[row_indices[ptr]] = values[row_indices[ptr]];
			ptr++;
		}
		return VectorObj<TObj>( arr + 0, _n);
    }

	TObj operator()(int row, int col){
        if (row < 0 || row >= _n || col < 0 || col >= _m) { throw std::out_of_range("Index out of range for element access."); }
		for (int i=col_ptr[col]; i < col_ptr[col + 1]; i++){
			if (row_indices[i] == row) { return values[i];}
		}
		return 0;
	}

	SparseMatrixCSC &operator=() {};
	SparseMatrixCSC &operator+() {};
	SparseMatrixCSC &operator-() {};
	SparseMatrixCSC &operator*() {};
	SparseMatrixCSC &operator/() {};

};
