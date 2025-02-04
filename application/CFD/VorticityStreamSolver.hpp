#ifndef VORTICITYSTREAMFUNCTION_HPP
#define VORTICITYSTREAMFUNCTION_HPP

#include "../../Obj/SparseObj.hpp"
#include "../../Obj/VectorObj.hpp"
#include "../Preconditioner/MultiGrid.hpp"
#include "../../ODE/RungeKutta.hpp"
#include <cmath>
#include <functional>

template<typename TNum>
class VorticityStreamSolver {
private:
    int nx, ny;                          // 网格点数
    TNum dx, dy;                         // 网格间距
    TNum Re;                             // 雷诺数
    TNum U;                              // 顶盖速度

    SparseMatrixCSC<TNum> laplacian;     // 拉普拉斯算子
    VectorObj<TNum> vorticity;           // 涡量场
    VectorObj<TNum> streamFunction;       // 流函数

    // 构建拉普拉斯算子矩阵
    void buildLaplacianMatrix() {
        const int n = nx * ny;
        laplacian = SparseMatrixCSC<TNum>(n, n);
        
        TNum dx2i = 1.0 / (dx * dx);
        TNum dy2i = 1.0 / (dy * dy);
        
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx;
                
                // 设置边界条件
                if (i == 0 || i == nx-1 || j == 0 || j == ny-1) {
                    laplacian.addValue(idx, idx, 1.0);
                    continue;
                }
                
                // 内部点的五点差分格式
                laplacian.addValue(idx, idx, -2.0 * (dx2i + dy2i));
                laplacian.addValue(idx, idx-1, dx2i);
                laplacian.addValue(idx, idx+1, dx2i);
                laplacian.addValue(idx, idx-nx, dy2i);
                laplacian.addValue(idx, idx+nx, dy2i);
            }
        }
        laplacian.finalize();
    }

    // 计算涡量的时间导数
    VectorObj<TNum> computeVorticityDerivative(const VectorObj<TNum>& w) const {
        VectorObj<TNum> dwdt(w.size());
        
        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                int idx = i + j * nx;
                
                // 计算流函数的空间导数
                TNum dpsidx = (streamFunction[idx+1] - streamFunction[idx-1]) / (2.0 * dx);
                TNum dpsidy = (streamFunction[idx+nx] - streamFunction[idx-nx]) / (2.0 * dy);
                
                // 计算涡量的空间导数
                TNum dwdx = (w[idx+1] - w[idx-1]) / (2.0 * dx);
                TNum dwdy = (w[idx+nx] - w[idx-nx]) / (2.0 * dy);
                
                // 计算扩散项
                TNum d2wdx2 = (w[idx+1] - 2.0*w[idx] + w[idx-1]) / (dx * dx);
                TNum d2wdy2 = (w[idx+nx] - 2.0*w[idx] + w[idx-nx]) / (dy * dy);
                
                // 合并对流项和扩散项
                dwdt[idx] = -(dpsidy * dwdx - dpsidx * dwdy) + (d2wdx2 + d2wdy2) / Re;
            }
        }
        
        return dwdt;
    }

public:
    VorticityStreamSolver(int nx_, int ny_, TNum Re_, TNum U_ = 1.0) 
        : nx(nx_), ny(ny_), Re(Re_), U(U_) {
        dx = 1.0 / (nx - 1);
        dy = 1.0 / (ny - 1);
        
        vorticity = VectorObj<TNum>(nx * ny, 0.0);
        streamFunction = VectorObj<TNum>(nx * ny, 0.0);
        
        buildLaplacianMatrix();
    }

    void solve(TNum dt, int nsteps) {
        // 设置初始条件
        // 顶盖驱动流的边界条件
        for (int i = 0; i < nx; ++i) {
            int topIdx = i + (ny-1) * nx;
            vorticity[topIdx] = -2.0 * U * streamFunction[topIdx-nx] / (dy * dy);
        }

        // 创建多重网格求解器
        AlgebraicMultiGrid<TNum, VectorObj<TNum>> amg;
        
        // 创建RK4求解器
        RungeKutta<TNum, VectorObj<TNum>> rk4;
        
        // 时间推进
        for (int step = 0; step < nsteps; ++step) {
            // 1. 求解流函数方程
            amg.amgVCycle(laplacian, vorticity, streamFunction, 3, 2, 0.25);
            
            // 2. 更新边界条件
            for (int i = 0; i < nx; ++i) {
                int topIdx = i + (ny-1) * nx;
                vorticity[topIdx] = -2.0 * U * streamFunction[topIdx-nx] / (dy * dy);
            }
            
            // 3. 时间推进涡量方程
            auto vorticityDerivative = [this](const VectorObj<TNum>& w) {
                return this->computeVorticityDerivative(w);
            };
            
            rk4.solve(vorticity, vorticityDerivative, dt, 1);
        }
    }

    // 获取结果
    const VectorObj<TNum>& getVorticity() const { return vorticity; }
    const VectorObj<TNum>& getStreamFunction() const { return streamFunction; }
};

#endif // VORTICITYSTREAMFUNCTION_HPP