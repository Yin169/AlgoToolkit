#ifndef ALGEBRAIC_DECOMPOSITION_HPP
#define ALGEBRAIC_DECOMPOSITION_HPP

#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "../../Obj/MatrixObj.hpp"
#include "../../Obj/VectorObj.hpp"
#include "../../Obj/DenseObj.hpp"
#include "../../Obj/SparseObj.hpp"
#include "../Factorized/basic.hpp"

namespace decomposition {

template<typename TObj>
class DomainDecomposition {
private:
    struct Graph {
        int num_vertices;
        std::vector<std::vector<std::pair<int, TObj>>> adjacency_list;
        
        explicit Graph(int n) : num_vertices(n), adjacency_list(n) {}
        
        void addEdge(int from, int to, TObj weight) {
            adjacency_list[from].push_back({to, weight});
            adjacency_list[to].push_back({from, weight});
        }
    };

    // Convert matrix to graph representation
    static Graph matrixToGraph(const MatrixObj<TObj>& matrix) {
        int n = matrix.getRows();
        Graph graph(n);
        
        // For sparse matrices, use the existing structure
        if (auto* sparse_matrix = dynamic_cast<const SparseMatrixCSC<TObj>*>(&matrix)) {
            for (int col = 0; col < sparse_matrix->_m; ++col) {
                for (int idx = sparse_matrix->col_ptr[col]; 
                     idx < sparse_matrix->col_ptr[col + 1]; ++idx) {
                    int row = sparse_matrix->row_indices[idx];
                    TObj value = sparse_matrix->values[idx];
                    if (row != col && value != TObj(0)) {
                        graph.addEdge(row, col, std::abs(value));
                    }
                }
            }
        }
        // For dense matrices, check all non-zero elements
        else {
            const auto* dense_matrix = dynamic_cast<const DenseObj<TObj>*>(&matrix);
            if (dense_matrix) {
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        TObj value = (*dense_matrix)(i, j);
                        if (i != j && value != TObj(0)) {
                            graph.addEdge(i, j, std::abs(value));
                        }
                    }
                }
            }
        }
        return graph;
    }

    // Compute the Fiedler vector using power iteration
    static VectorObj<TObj> computeFiedlerVector(const Graph& graph) {
        int n = graph.num_vertices;
        
        // Construct the Laplacian matrix
        DenseObj<TObj> laplacian(n, n);
        for (int i = 0; i < n; ++i) {
            TObj degree = TObj(0);
            for (const auto& edge : graph.adjacency_list[i]) {
                degree += edge.second;
                laplacian(i, edge.first) = -edge.second;
            }
            laplacian(i, i) = degree;
        }

        // Initialize random vector
        VectorObj<TObj> v(n);
        for (int i = 0; i < n; ++i) {
            v[i] = static_cast<TObj>(rand()) / RAND_MAX;
        }
        
        // Power iteration to find second eigenvector
        const int max_iter = 100;
        for (int iter = 0; iter < max_iter; ++iter) {
            v = laplacian * v;
            
            // Orthogonalize against constant vector
            TObj mean = std::accumulate(v.element(), v.element() + n, TObj(0)) / n;
            for (int i = 0; i < n; ++i) {
                v[i] -= mean;
            }
            
            v.normalize();
        }
        
        return v;
    }

    // Recursive bisection
    static void recursiveBisection(
        const Graph& graph,
        std::vector<int>& partition,
        int current_part,
        int depth,
        int max_depth
    ) {
        if (depth >= max_depth) return;

        // Compute Fiedler vector for spectral bisection
        VectorObj<TObj> fiedler = computeFiedlerVector(graph);
        
        // Find median for balanced partition
        std::vector<std::pair<TObj, int>> sorted_vertices;
        for (int i = 0; i < graph.num_vertices; ++i) {
            if (partition[i] == current_part) {
                sorted_vertices.push_back({fiedler[i], i});
            }
        }
        
        std::sort(sorted_vertices.begin(), sorted_vertices.end());
        int middle = sorted_vertices.size() / 2;
        
        // Assign partitions
        for (int i = middle; i < sorted_vertices.size(); ++i) {
            partition[sorted_vertices[i].second] = current_part * 2 + 1;
        }
        
        // Recurse on both halves
        recursiveBisection(graph, partition, current_part * 2, depth + 1, max_depth);
        recursiveBisection(graph, partition, current_part * 2 + 1, depth + 1, max_depth);
    }

public:
    // Main decomposition function
    static std::vector<int> decompose(
        const MatrixObj<TObj>& matrix,
        int num_domains
    ) {
        if (num_domains <= 0) {
            throw std::invalid_argument("Number of domains must be positive");
        }

        int n = matrix.getRows();
        if (n != matrix.getCols()) {
            throw std::invalid_argument("Matrix must be square for domain decomposition");
        }

        // Convert matrix to graph
        Graph graph = matrixToGraph(matrix);
        
        // Initialize partition vector
        std::vector<int> partition(n, 0);
        
        // Calculate required depth for desired number of domains
        int max_depth = static_cast<int>(std::ceil(std::log2(num_domains)));
        
        // Perform recursive bisection
        recursiveBisection(graph, partition, 0, 0, max_depth);
        
        // Compress partition numbers to be consecutive
        std::unordered_map<int, int> compression_map;
        int next_id = 0;
        for (int& p : partition) {
            if (compression_map.find(p) == compression_map.end()) {
                compression_map[p] = next_id++;
            }
            p = compression_map[p];
        }
        
        return partition;
    }

    // Get interface nodes between domains
    static std::vector<std::vector<int>> getInterfaces(
        const MatrixObj<TObj>& matrix,
        const std::vector<int>& partition
    ) {
        int num_domains = *std::max_element(partition.begin(), partition.end()) + 1;
        std::vector<std::vector<int>> interfaces(num_domains);
        
        // For sparse matrices
        if (auto* sparse_matrix = dynamic_cast<const SparseMatrixCSC<TObj>*>(&matrix)) {
            for (int col = 0; col < sparse_matrix->_m; ++col) {
                int domain1 = partition[col];
                for (int idx = sparse_matrix->col_ptr[col]; 
                     idx < sparse_matrix->col_ptr[col + 1]; ++idx) {
                    int row = sparse_matrix->row_indices[idx];
                    int domain2 = partition[row];
                    if (domain1 != domain2) {
                        interfaces[domain1].push_back(col);
                        break;
                    }
                }
            }
        }
        // For dense matrices
        else {
            const auto* dense_matrix = dynamic_cast<const DenseObj<TObj>*>(&matrix);
            if (dense_matrix) {
                int n = dense_matrix->getRows();
                for (int i = 0; i < n; ++i) {
                    int domain1 = partition[i];
                    bool is_interface = false;
                    for (int j = 0; j < n && !is_interface; ++j) {
                        if ((*dense_matrix)(i, j) != TObj(0) && domain1 != partition[j]) {
                            interfaces[domain1].push_back(i);
                            is_interface = true;
                        }
                    }
                }
            }
        }
        
        // Remove duplicates
        for (auto& interface : interfaces) {
            std::sort(interface.begin(), interface.end());
            interface.erase(std::unique(interface.begin(), interface.end()), 
                          interface.end());
        }
        
        return interfaces;
    }
};

} // namespace decomposition

#endif // ALGEBRAIC_DECOMPOSITION_HPP