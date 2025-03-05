#include "../../Obj/SparseObj.hpp"
#include "../../Obj/VectorObj.hpp" 

namespace SparseLA{

	template<typename T = double >
	void dfs(size_t i, const SparseMatrixCSC<T> &G, std::vector<bool> &visited, std::stack<size_t> &reach){
		if (visited[i]) return;
		visited[i] = true;
		for(size_t j=G.col_ptr[i]; j<G.col_ptr[i+1]; j++){
			dfs(G.row_indices[j], G, visited, reach);
		}
		reach.push(i);
	}

	template<typename T = double >
	std::stack<size_t> Reach(const SparseMatrixCSC<T> &G){
		size_t n = G.getRows(), m = G.getCols();
		std::vector<bool> visited(n, false);
		std::stack<size_t> reach;
		for(size_t i=0; i<n; i++){
			if (!visited[i]){
				dfs<T>(i, G, visited, reach);
			}
		}
		return reach;		
	}

	template<typename T = double >
	void LSolve(const SparseMatrixCSC<T> &L, VectorObj<T> &b){
		std::stack<size_t> reach = Reach<T>(L);
		while(!reach.empty()){
			size_t i = reach.top();
			reach.pop();
			for(size_t j = L.col_ptr[i]; j < L.col_ptr[i+1]; j++){
				size_t row = L.row_indices[j];
				if(row > i){
					b[row] -= L.values[j] * b[i];
				}
			}
		}
	}
}