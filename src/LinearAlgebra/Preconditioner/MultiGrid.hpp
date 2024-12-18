#ifndef AMG_HPP
#define AMG_HPP

#include "SparseObj.hpp"
#include "IterSolver.hpp"
#include <vector>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <numeric>

// Algebraic Multi-Grid Solver Template
template <typename TNum, typename VectorType>
class AlgebraicMultiGrid {
private:
    // Compute strong connections based on threshold theta
    std::vector<std::unordered_set<int>> computeStrongConnections(const SparseMatrixCSC<TNum>& A, TNum theta) {
        std::vector<std::unordered_set<int>> connections(A.getRows());
        for (int col = 0; col < A.getCols(); ++col) {
            TNum diag = std::abs(A(col, col));
            for (int k = A.col_ptr[col]; k < A.col_ptr[col + 1]; ++k) {
                int row = A.row_indices[k];
                TNum value = std::abs(A.values[k]);
                if (row != col && value > theta * diag) {
                    connections[row].insert(col);
                    connections[col].insert(row);
                }
            }
        }
        return connections;
    }

    // Coarse grid selection using greedy C/F splitting
    std::vector<int> coarseGridSelection(const SparseMatrixCSC<TNum>& A, TNum theta) {
        auto strongConnections = computeStrongConnections(A, theta);
        std::vector<int> gridFlag(A.getRows(), -1); // -1: Unassigned, 0: Fine, 1: Coarse
        std::vector<int> coarsePoints;

        for (int i = 0; i < A.getRows(); ++i) {
            if (gridFlag[i] == -1) {
                gridFlag[i] = 1; // Mark as coarse
                coarsePoints.push_back(i);
                for (int neighbor : strongConnections[i]) {
                    if (gridFlag[neighbor] == -1) {
                        gridFlag[neighbor] = 0; // Mark as fine
                    }
                }
            }
        }
        return coarsePoints;
    }

    // Build prolongation matrix P
    SparseMatrixCSC<TNum> buildProlongation(const SparseMatrixCSC<TNum>& A, const std::vector<int>& coarsePoints) {
        SparseMatrixCSC<TNum> P(A.getRows(), coarsePoints.size());
        std::unordered_map<int, int> coarseMap;
        for (size_t i = 0; i < coarsePoints.size(); ++i) {
            coarseMap[coarsePoints[i]] = i;
            P.addValue(coarsePoints[i], i, 1.0);
        }

        for (int i = 0; i < A.getRows(); ++i) {
            if (coarseMap.find(i) == coarseMap.end()) { // Fine points
                TNum sum = 0.0;
                for (int k = A.col_ptr[i]; k < A.col_ptr[i + 1]; ++k) {
                    sum += std::abs(A.values[k]);
                }
                for (int k = A.col_ptr[i]; k < A.col_ptr[i + 1]; ++k) {
                    int coarseIdx = coarseMap[A.row_indices[k]];
                    P.addValue(i, coarseIdx, A.values[k] / sum);
                }
            }
        }
        P.finalize();
        return P;
    }

    // Compute Galerkin coarse matrix A_c = P^T * A * P
    SparseMatrixCSC<TNum> galerkinCoarseMatrix(const SparseMatrixCSC<TNum>& A, const SparseMatrixCSC<TNum>& P) {
        SparseMatrixCSC<TNum> PT = P.Transpose();
        SparseMatrixCSC<TNum> AP = A * P;
        return PT * AP;
    }

public:
    AlgebraicMultiGrid() = default;
    ~AlgebraicMultiGrid() = default;

    // Recursive AMG V-cycle
    void amgVCycle(const SparseMatrixCSC<TNum>& A, const VectorType& b, VectorType& x, int levels, int smoothingSteps, TNum theta) {
        if (levels == 1) {
            StaticIterMethod<TNum, SparseMatrixCSC<TNum>, VectorType>(A, b, smoothingSteps).solve(x);
            return;
        }

        // Pre-smoothing
        StaticIterMethod<TNum, SparseMatrixCSC<TNum>, VectorType>(A, b, smoothingSteps).solve(x);

        // Compute residual r = b - A * x
        VectorType r = b - (const_cast<SparseMatrixCSC<TNum>&>(A) * x);

        // Build coarse grid
        auto coarsePoints = coarseGridSelection(A, theta);
        SparseMatrixCSC<TNum> P = buildProlongation(A, coarsePoints);
        SparseMatrixCSC<TNum> A_c = galerkinCoarseMatrix(A, P);

        // Restrict residual to coarse grid
        VectorType r_c = P.Transpose() * r;

        // Solve on coarse grid
        VectorType x_c(r_c.size(), 0.0);
        amgVCycle(A_c, r_c, x_c, levels - 1, smoothingSteps, theta);

        // Prolong correction to fine grid
        VectorType correction = P * x_c;
        x = x + correction;

        // Post-smoothing
        StaticIterMethod<TNum, SparseMatrixCSC<TNum>, VectorType>(A, b, smoothingSteps).solve(x);
    }
};

#endif // AMG_HPP
