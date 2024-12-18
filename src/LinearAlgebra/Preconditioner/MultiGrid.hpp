#ifndef AMG_HPP
#define AMG_HPP

#include "SparseObj.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <iostream>

// Helper to compute strength of connection
template <typename TObj>
std::vector<std::unordered_set<int>> computeStrongConnections(const SparseMatrixCSC<TObj>& A, TObj theta) {
    std::vector<std::unordered_set<int>> connections(A.getRows());

    for (int col = 0; col < A.getCols(); ++col) {
        TObj diag = A(col, col);
        for (int k = A.col_ptr[col]; k < A.col_ptr[col + 1]; ++k) {
            int row = A.row_indices[k];
            TObj value = A.values[k];
            if (row != col && std::abs(value) > theta * std::abs(diag)) {
                connections[row].insert(col);
                connections[col].insert(row);
            }
        }
    }
    return connections;
}

// Coarse grid selection using C/F splitting
template <typename TObj>
std::vector<int> coarseGridSelection(const SparseMatrixCSC<TObj>& A, TObj theta) {
    auto strongConnections = computeStrongConnections(A, theta);

    std::vector<int> gridFlag(A.getRows(), -1);  // -1: Unassigned, 0: Fine, 1: Coarse
    std::vector<int> coarsePoints;

    for (int i = 0; i < A.getRows(); ++i) {
        if (gridFlag[i] == -1) {
            // Select current as coarse
            gridFlag[i] = 1;
            coarsePoints.push_back(i);

            // Mark neighbors as fine
            for (int neighbor : strongConnections[i]) {
                if (gridFlag[neighbor] == -1) {
                    gridFlag[neighbor] = 0;
                }
            }
        }
    }
    return coarsePoints;
}

// Build the prolongation operator
template <typename TObj>
SparseMatrixCSC<TObj> buildProlongation(const SparseMatrixCSC<TObj>& A, const std::vector<int>& coarsePoints) {
    SparseMatrixCSC<TObj> P(A.getRows(), coarsePoints.size());

    for (size_t i = 0; i < coarsePoints.size(); ++i) {
        P.addValue(coarsePoints[i], i, 1.0);  // Coarse grid points map directly
    }

    for (int i = 0; i < A.getRows(); ++i) {
        if (std::find(coarsePoints.begin(), coarsePoints.end(), i) == coarsePoints.end()) {
            // Fine points: interpolate from neighbors
            TObj sum = 0.0;
            for (int k = A.col_ptr[i]; k < A.col_ptr[i + 1]; ++k) {
                sum += std::abs(A.values[k]);
            }
            for (int k = A.col_ptr[i]; k < A.col_ptr[i + 1]; ++k) {
                if (std::find(coarsePoints.begin(), coarsePoints.end(), A.row_indices[k]) != coarsePoints.end()) {
                    P.addValue(i, std::distance(coarsePoints.begin(),
                                  std::find(coarsePoints.begin(), coarsePoints.end(), A.row_indices[k])),
                               A.values[k] / sum);
                }
            }
        }
    }
    P.finalize();
    return P;
}

// Galerkin coarse matrix: A_c = P^T * A * P
template <typename TObj>
SparseMatrixCSC<TObj> galerkinCoarseMatrix(const SparseMatrixCSC<TObj>& A, const SparseMatrixCSC<TObj>& P) {
    SparseMatrixCSC<TObj> PT = P;  // Transpose of P
    SparseMatrixCSC<TObj> AP = A * P;  // A * P
    return PT * AP;  // P^T * (A * P)
}

// Recursive AMG V-cycle
template <typename TObj>
void amgVCycle(const SparseMatrixCSC<TObj>& A, const std::vector<TObj>& b, std::vector<TObj>& x, int levels, int smoothingSteps, TObj theta) {
    if (levels == 1) {
        gaussSeidelSmooth(A, b, x, 10);  // Solve directly at coarsest level
        return;
    }

    // Pre-smoothing
    gaussSeidelSmooth(A, b, x, smoothingSteps);

    // Compute residual
    std::vector<TObj> r = computeResidual(A, b, x);

    // Build coarse grid
    auto coarsePoints = coarseGridSelection(A, theta);
    SparseMatrixCSC<TObj> P = buildProlongation(A, coarsePoints);
    SparseMatrixCSC<TObj> A_c = galerkinCoarseMatrix(A, P);

    // Restrict residual
    std::vector<TObj> r_c = restrictToCoarse(r);

    // Solve on coarse grid
    std::vector<TObj> x_c(r_c.size(), 0.0);
    amgVCycle(A_c, r_c, x_c, levels - 1, smoothingSteps, theta);

    // Prolong correction
    std::vector<TObj> correction = prolongToFine(x_c, b.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] += correction[i];
    }

    // Post-smoothing
    gaussSeidelSmooth(A, b, x, smoothingSteps);
}

#endif // AMG_HPP
