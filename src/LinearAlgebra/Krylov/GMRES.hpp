#ifndef GMRES_HPP
#define GMRES_HPP

#include <vector>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <iostream>
#include "../Factorized/basic.hpp"
#include "../../Obj/SparseObj.hpp"
#include "../../Obj/VectorObj.hpp"
#include "../Preconditioner/ILU.hpp"

template <typename TNum, typename MatrixType = SparseMatrixCSC<TNum>, typename VectorType = VectorObj<TNum>>
class GMRES {
private:
    ILUPreconditioner<TNum, MatrixType> preconditioner;
    bool usePreconditioner = false;
public:
    GMRES() = default;
    virtual ~GMRES() = default;

    void enablePreconditioner() {
        usePreconditioner = true;
    }

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

        if (usePreconditioner) {
            preconditioner.compute(A);  // Compute ILU factorization once
        }
        VectorType r = b - A * x;
        if (usePreconditioner) {
            r = preconditioner.solve(r);  // Apply M^(-1)(b - Ax)
        }

        double beta = r.L2norm();
        std::cout << "Initial residual norm: " << beta << std::endl;
        if (beta < tol) {
            std::cout << "Initial guess is good enough." << std::endl;
            return; // Initial guess is good enough
        }

        std::vector<VectorType> V(KrylovDim + 1, VectorType(n));
        MatrixType H(KrylovDim + 1, KrylovDim + 1);
        std::vector<TNum> cs(KrylovDim, TNum(1));
        std::vector<TNum> sn(KrylovDim, TNum(0));
        std::vector<TNum> rho(KrylovDim, TNum(0));
        VectorType e1(KrylovDim + 1, TNum(0));
        e1[0] = beta;

        for (int iter = 0; iter < maxIter; iter++) {
            V[0] = r / beta;
            int KryUpdate = KrylovDim;
            for (int j = 0; j < KrylovDim; ++j) {
                VectorType w;
                if (usePreconditioner) {
                    w = preconditioner.solve(A * V[j]);  // Apply M^(-1)A to v_j
                } else {
                    w = A * V[j];
                }

                for (int i = 0; i < j; ++i) {
                    TNum w_dot_v = V[i] * w;
                    setMatrixValue(H, i, j, w_dot_v);
                    w = w - V[i] * w_dot_v;
                }
                
                TNum w_norm = w.L2norm();
                setMatrixValue(H, j + 1, j, w_norm);
                if (w_norm < tol) {
                    KryUpdate = j;
                    std::cout << "KrylovUpdate : " << KryUpdate << std::endl;
                    break;
                }
                V[j + 1] = w / H(j + 1, j);

                // Apply Givens rotations
                for (int i = 0; i < j; ++i) {
                    applyMatrixGivensRotation(H, i, j, i + 1, j, cs[i], sn[i]);
                }
                generateGivensRotation(H(j, j), H(j + 1, j), cs[j], sn[j], rho[j]);
                setMatrixValue(H, j, j, rho[j]);
                setMatrixValue(H, j + 1, j, TNum(0));
                applyGivensRotation(e1[j], e1[j + 1], cs[j], sn[j]);
            }

            updateSolution(x, H, V, e1, KryUpdate);
            r = b - A * x;
            if (usePreconditioner) {
                r = preconditioner.solve(r);  // Apply M^(-1) to the residual
            }

            beta = r.L2norm();
            std::cout << "Residual norm after restart: " << beta << " " << std::endl;

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
    void setMatrixValue(DenseObj<TNum>& H, int i, int j, TNum value) {
        H(i, j) = value;
    }

    void setMatrixValue(SparseMatrixCSC<TNum>& H, int i, int j, TNum value) {
        H.addValue(i, j, value);
        H.finalize();
        // printf(" i : %d j : %d H: %f value : %f \n",i, j, H(i, j), value);
    }

    void applyGivensRotation(TNum& dx, TNum& dy, TNum cs, TNum sn) {
        TNum temp = dx;
        dx = cs * temp + sn * dy;
        dy = -sn * temp + cs * dy;
    }

    void applyMatrixGivensRotation(MatrixType& H, int i, int j, int ii, int jj, TNum cs, TNum sn) {
        TNum temp = H(i, j);
        setMatrixValue(H, i, j,  cs * H(i, j) + sn * H(ii, jj));
        setMatrixValue(H, ii, jj, -sn * temp + cs * H(ii, jj));
    }

    void generateGivensRotation(TNum dx, TNum dy, TNum& cs, TNum& sn, TNum& rho) {
        rho = std::sqrt(dx * dx + dy * dy);
        // Protect against division by zero
        if (std::abs(rho) < std::numeric_limits<TNum>::epsilon()) {
            cs = TNum(1);
            sn = TNum(0);
            rho = TNum(0);
        } else {
            cs = dx / rho;
            sn = dy / rho;
        }
    }

    void updateSolution(VectorType& x, const MatrixType& H, const std::vector<VectorType>& V, const VectorType& e1, int k) {
        VectorType y(k, TNum(0));
        // Back substitution for solving upper triangular system Hy = e1
        for (int i = k - 1; i >= 0; --i) {
            y[i] = e1[i];
            for (int j = i + 1; j < k; ++j) {
                y[i] = y[i] - H(i, j) * y[j];
            }
            // Check for zero diagonal element to avoid division by zero
            if (std::abs(H(i, i)) < std::numeric_limits<TNum>::epsilon()) {
                throw std::runtime_error("GMRES updateSolution: Zero diagonal element encountered in H.");
            }
            y[i] = y[i] / H(i, i);
        }
        // Update solution: x = x + sum_i {V[i] * y[i]}
        for (int i = 0; i < k; ++i) {
            x = x + (V[i] * y[i]);
        }
    }
};

#endif // GMRES_HPP