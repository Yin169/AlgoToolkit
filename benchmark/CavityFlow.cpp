#include "../src/LinearAlgebra/Solver/VorticityStreamSolver.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>	

template<typename T>
void writeVTK(const std::string& filename,
              const VectorObj<T>& vorticity,
              const VectorObj<T>& streamFunction,
              int nx, int ny,
              double dx, double dy) {
    std::ofstream file(filename);
    file << std::scientific << std::setprecision(6);  // 添加精度控制
    
    // 写入 VTK 文件头
    file << "# vtk DataFile Version 3.0\n";
    file << "Cavity Flow Solution\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    
    // 写入网格尺寸
    file << "\nDIMENSIONS " << nx << " " << ny << " 1\n";
    
    // 写入点坐标
    file << "\nPOINTS " << nx * ny << " double\n";  // 使用double而不是float
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << i * dx << " " << j * dy << " 0.0\n";
        }
    }
    
    // 写入点数据
    file << "\nPOINT_DATA " << nx * ny << "\n";
    
    // 写入涡量场
    file << "SCALARS vorticity double 1\n";  // 使用double而不是float
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nx * ny; ++i) {
        file << vorticity[i] << "\n";
    }
    
    // 写入流函数
    file << "\nSCALARS streamFunction double 1\n";  // 使用double而不是float
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nx * ny; ++i) {
        file << streamFunction[i] << "\n";
    }
    
    file.close();
}


int main() {
    // 设置参数
    const int nx = 128;
    const int ny = 128;
    const double Re = 1000.0;  // 雷诺数
    const double dt = 0.001;   // 时间步长
    const int nsteps = 1000;   // 总时间步数
    const int output_interval = 100;  // 输出间隔

    // 创建求解器
    VorticityStreamSolver<double> solver(nx, ny, Re);

    // 求解并定期输出结果
    for (int step = 0; step < nsteps; ++step) {
        solver.solve(dt, 1);

        if (step % output_interval == 0) {
            std::stringstream ss;
			std::string filename = "cavity_flow_" + std::to_string(step) + ".vtk";
            writeVTK<double>(filename, solver.getVorticity(), solver.getStreamFunction(), nx, ny, 1.0/nx, 1.0/ny);         

            std::cout << "Step " << step << " completed." << std::endl;
        }
    }

    return 0;
}