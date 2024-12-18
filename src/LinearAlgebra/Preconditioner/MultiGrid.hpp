#ifndef AMG_HPP
#define AMG_HPP

#include "SparseObj.hpp"
#include "IterSolver.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <iostream>

template <typename TNum, typename VectorType>
class AlgebraicMultiGrid {
    private:

    std::vector<std::unordered_set<int>> computeStrongConnections(const SparseMatrixCSC<TNum>& A, TNum theta) {
        std::vector<std::unordered_set<int>> connections(A.getRows());

        for (int col = 0; col < A.getCols(); ++col) {
            TNum diag = A(col, col);
            for (int k = A.col_ptr[col]; k < A.col_ptr[col + 1]; ++k) {
                int row = A.row_indices[k];
                TNum value = A.values[k];
                if (row != col && std::abs(value) > theta * std::abs(diag)) {
                    connections[row].insert(col);
                    connections[col].insert(row);
                }
            }
        }
        return connections;
    }

    // Coarse grid selection using C/F splitting
    std::vector<int> coarseGridSelection(const SparseMatrixCSC<TNum>& A, TNum theta) {
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

    SparseMatrixCSC<TNum> buildProlongation(const SparseMatrixCSC<TNum>& A, const std::vector<int>& coarsePoints) {
        SparseMatrixCSC<TNum> P(A.getRows(), coarsePoints.size());

        for (size_t i = 0; i < coarsePoints.size(); ++i) {
            P.addValue(coarsePoints[i], i, 1.0);  // Coarse grid points map directly
        }

        for (int i = 0; i < A.getRows(); ++i) {
            if (std::find(coarsePoints.begin(), coarsePoints.end(), i) == coarsePoints.end()) {
                // Fine points: interpolate from neighbors
                TNum sum = 0.0;
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

    VectorType restrictToCoarse(const VectorType& fineResidual) {
        int coarseSize = fineResidual.size() / 2;  // Simple 2-to-1 coarsening
        VectorType> coarseResidual(coarseSize, 0.0);
        for (int i = 0; i < coarseSize; ++i) {
            coarseResidual[i] = (fineResidual[2 * i] + fineResidual[2 * i + 1]) / 2.0;
        }
        return coarseResidual;
    }

    // Galerkin coarse matrix: A_c = P^T * A * P
    SparseMatrixCSC<TNum> galerkinCoarseMatrix(const SparseMatrixCSC<TNum>& A, const SparseMatrixCSC<TNum>& P) {
        SparseMatrixCSC<TNum> PT = P;  // Transpose of P
        SparseMatrixCSC<TNum> AP = A * P;  // A * P
        return PT * AP;  // P^T * (A * P)
    }

    public:
    AlgebraicMultiGrid() = default;
    ~AlgebraicMultiGrid() = default;

    // Recursive AMG V-cycle
    void amgVCycle(const SparseMatrixCSC<TNum>& A, const VectorType& b, VectorType& x, int levels, int smoothingSteps, TNum theta) {
        if (levels == 1) {
            StaticIterMethod(P, A, b, smoothingSteps).solve(x);  // Solve directly at coarsest level
            return;
        }

        // Pre-smoothing
        StaticIterMethod(P, A, b, smoothingSteps).solve(x);

        // Compute residual
        VectorType r = b - A * x;

        // Build coarse grid
        auto coarsePoints = coarseGridSelection(A, theta);
        SparseMatrixCSC<TNum> P = buildProlongation(A, coarsePoints);
        SparseMatrixCSC<TNum> A_c = galerkinCoarseMatrix(A, P);

        // Restrict residual
        VectorType r_c = restrictToCoarse(r);

        // Solve on coarse grid
        VectorType x_c(r_c.size(), 0.0);
        amgVCycle(A_c, r_c, x_c, levels - 1, smoothingSteps, theta);

        // Prolong correction
        VectorType correction = prolongToFine(x_c, b.size());
        x = x + correction;

        // Post-smoothing
        StaticIterMethod(P, A, b, smoothingSteps).solve(x);
    }
};

#endif // AMG_HPP