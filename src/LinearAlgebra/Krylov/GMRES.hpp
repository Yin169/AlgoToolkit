#ifndef GMRES_HPP
#define GMRES_HPP

#include <vector>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <iostream>
#include "basic.hpp"
#include "SparseObj.hpp"
#include "VectorObj.hpp"
#include "KrylovSubspace.hpp"

template <typename TNum, typename MatrixType = SparseMatrixCSC<TNum>, typename VectorType = VectorObj<TNum>>
class GMRES {
public:
    GMRES() = default;
    virtual ~GMRES() = default;

    void solve(const MatrixType& A, const VectorType& b, VectorType& x, int maxIter, int KrylovDim, double tol) {
        const int n = b.size();
        if (A.getRows() != n || A.getCols() != n) {
            throw std::invalid_argument("Matrix dimensions must match vector size");
        }
        if (x.size() != n) {
            throw std::invalid_argument("Initial guess vector must match system size");
        }
        if (KrylovDim <= 0 || KrylovDim > n) {
            throw std::invalid_argument("Invalid Krylov subspace dimension");
        }

        VectorType r = b - const_cast<MatrixType&>(A) * x;
        double beta = r.L2norm();
        std::cout << "Initial residual norm: " << beta << std::endl;
        if (beta < tol) {
            std::cout << "Initial guess is good enough." << std::endl;
            return; // Initial guess is good enough
        }

        std::vector<VectorType> V(KrylovDim + 1, VectorType(n));
        MatrixType H(KrylovDim + 1, KrylovDim + 1);
        std::vector<TNum> cs(KrylovDim, TNum(0));
        std::vector<TNum> sn(KrylovDim, TNum(0));
        VectorType e1(KrylovDim + 1, TNum(0));
        int update = KrylovDim;
        e1[0] = beta;

        for (int iter = 0; iter < maxIter; iter++) {
            V[0] = r / beta;

            for (int j=0; j < KrylovDim; ++j) {
                update = KrylovDim;
                VectorType w = const_cast<MatrixType&>(A) * V[j];
                for (int i = 0; i <= j; ++i) {
                    setMatrixValue(H, i, j, V[i] * w);
                    w = w - V[i] * H(i, j);
                }
                setMatrixValue(H, j + 1, j, w.L2norm());
               if (H(j + 1, j) < tol) {
                    update = j+1;
                    std::cout << "Happy breakdown at iteration " << iter * j << std::endl;
                    break;
                }
                V[j + 1] = w / H(j + 1, j);

                // Apply Givens rotations
                for (int i = 0; i < j; ++i) {
                    applyMatrixGivensRotation(H, i, j, i + 1, j, cs[i], sn[i]);
                    applyGivensRotation(e1[i], e1[i + 1], cs[i], sn[i]);
                }
                generateGivensRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
                applyMatrixGivensRotation(H, j, j, j + 1, j, cs[j], sn[j]);
                applyGivensRotation(e1[j], e1[j + 1], cs[j], sn[j]);

                beta = std::abs(e1[j + 1]);
            }

            updateSolution(x, H, V, e1, update);
            r = b - const_cast<MatrixType&>(A) * x;
            beta = r.L2norm();
            std::cout << "Residual norm after restart: " << beta << " " << std::endl;

            for(int ii=0; ii < x.size(); ++ii){
                std::cout << x[ii] << " ";
            }
            std::cout << std::endl;

            if (beta < tol) {
                std::cout << "Converged after restart at iteration " << iter << std::endl;
                return; // Converged
            }
            e1 = VectorType(KrylovDim + 1, TNum(0));
            e1[0] = beta;
        }
        std::cout << "Reached maximum iterations without convergence." << std::endl;
    }

private:
    void setMatrixValue(MatrixType& H, int i, int j, TNum value) {
        if constexpr (std::is_same_v<MatrixType, SparseMatrixCSC<TNum>>) {
            H.addValue(i, j, value);
            H.finalize();
        } else {
            H(i, j) = value;
        }
    }

    void applyGivensRotation(TNum& dx, TNum& dy, TNum cs, TNum sn) {
        TNum temp = cs * dx + sn * dy;
        dy = -sn * dx + cs * dy;
        dx = temp;
    }

    void applyMatrixGivensRotation(MatrixType& H, int i, int j, int ii, int jj, TNum cs, TNum sn) {
        setMatrixValue(H, i, j,  cs * H(i, j) + sn * H(ii, jj));
        setMatrixValue(H, ii, jj, -sn * H(i, j) + cs * H(ii, jj));
    }

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

    void updateSolution(VectorType& x, const MatrixType& H, const std::vector<VectorType>& V, const VectorType& e1, int k) {
        VectorType y(k, TNum(0));
        for (int i = k - 1; i >= 0; --i) {
            y[i] = e1[i];
            for (int j = i + 1; j < k; ++j) {
                y[i] = y[i] - H(i, j) * y[j];
            }
            y[i] = y[i] / H(i, i);
        }
        for (int i = 0; i < k; ++i) {
            x = x + (V[i] * y[i]);
        }
    }
};

#endif // GMRES_HPP