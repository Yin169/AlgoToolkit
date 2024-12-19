#ifndef GMRES_HPP
#define GMRES_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream> // For debugging logs
#include "SparseObj.hpp"
#include "VectorObj.hpp"
#include "KrylovSubspace.hpp"

template <typename TNum, typename MatrixType = SparseMatrixCSC<TNum>, typename VectorType = VectorObj<TNum>>
class GMRES {
public:
    // Default constructor
    GMRES() = default;

    // Virtual destructor
    virtual ~GMRES() = default;

    // GMRES solver
    void solve(const MatrixType& A, const VectorType& b, VectorType& x, int maxIter, int restart, double tol) {
        int n = b.size();
        if (x.size() != n) {
            throw std::invalid_argument("Dimension mismatch between x and b.");
        }

        VectorType r = b - A * x;
        double beta = r.L2norm();
        if (beta < tol) {
            return; // Initial guess is good enough
        }

        std::vector<VectorType> V(restart + 1, VectorType(n));
        std::vector<std::vector<TNum>> H(restart + 1, std::vector<TNum>(restart, TNum(0)));
        std::vector<TNum> cs(restart, TNum(0));
        std::vector<TNum> sn(restart, TNum(0));
        std::vector<TNum> e1(restart + 1, TNum(0));
        e1[0] = beta;

        for (int iter = 0; iter < maxIter; iter += restart) {
            V[0] = r / beta;

            for (int j = 0; j < restart; ++j) {
                VectorType w = A * V[j];

                for (int i = 0; i <= j; ++i) {
                    H[i][j] = V[i] * w;
                    w = w - V[i] * H[i][j];
                }
                H[j + 1][j] = w.L2norm();
                if (H[j + 1][j] < tol) {
                    break; // Happy breakdown
                }
                V[j + 1] = w / H[j + 1][j];

                // Apply Givens rotations
                for (int i = 0; i < j; ++i) {
                    applyGivensRotation(H[i][j], H[i + 1][j], cs[i], sn[i]);
                }
                generateGivensRotation(H[j][j], H[j + 1][j], cs[j], sn[j]);
                applyGivensRotation(H[j][j], H[j + 1][j], cs[j], sn[j]);
                applyGivensRotation(e1[j], e1[j + 1], cs[j], sn[j]);

                beta = std::abs(e1[j + 1]);
                if (beta < tol) {
                    updateSolution(x, H, V, e1, j + 1);
                    return; // Converged
                }
            }
            updateSolution(x, H, V, e1, restart);
            r = b - A * x;
            beta = r.L2norm();
            if (beta < tol) {
                return; // Converged
            }
            e1[0] = beta;
            std::fill(e1.begin() + 1, e1.end(), TNum(0));
        }
    }

private:
    // Apply Givens rotation
    void applyGivensRotation(TNum& dx, TNum& dy, TNum cs, TNum sn) {
        TNum temp = cs * dx + sn * dy;
        dy = -sn * dx + cs * dy;
        dx = temp;
    }

    // Generate Givens rotation
    void generateGivensRotation(TNum dx, TNum dy, TNum& cs, TNum& sn) {
        if (dy == TNum(0)) {
            cs = 1;
            sn = 0;
        } else if (std::abs(dy) > std::abs(dx)) {
            TNum temp = dx / dy;
            sn = 1 / std::sqrt(1 + temp * temp);
            cs = temp * sn;
        } else {
            TNum temp = dy / dx;
            cs = 1 / std::sqrt(1 + temp * temp);
            sn = temp * cs;
        }
    }

    // Update solution
    void updateSolution(VectorType& x, const std::vector<std::vector<TNum>>& H, const std::vector<VectorType>& V, const std::vector<TNum>& e1, int k) {
        std::vector<TNum> y(k, TNum(0));
        for (int i = k - 1; i >= 0; --i) {
            y[i] = e1[i];
            for (int j = i + 1; j < k; ++j) {
                y[i] -= H[i][j] * y[j];
            }
            y[i] /= H[i][i];
        }
        for (int i = 0; i < k; ++i) {
            x = x + V[i] * y[i];
        }
    }
};

#endif // GMRES_HPP