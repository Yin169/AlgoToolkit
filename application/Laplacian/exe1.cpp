#include "../../src/PDEs/FDM/Laplacian.hpp"

int main() {
    // Define domain and grid
    const int nx = 50, ny = 50, nz = 50;
    const double xmin = 0.0, xmax = 1.0;
    const double ymin = 0.0, ymax = 1.0;
    const double zmin = 0.0, zmax = 1.0;
    
    // Create solver
    Laplacian3DFDM<double> solver(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax);
    
    // Define source term function (e.g., f(x,y,z) = 6)
    auto sourceFunc = [](double x, double y, double z) -> double {
        return 6.0;  // This would give u = x²+y²+z² as the solution
    };

    // Define boundary condition function
    auto boundaryFunc = [](double x, double y, double z) -> double {
        return x*x + y*y + z*z;  // Exact solution on boundary
    };
    
    // Solve using SOR method with relaxation parameter 1.5
    auto solution = solver.solve(boundaryFunc, sourceFunc, xmin, ymin, zmin, 1.5);
    
    // Export solution for visualization
    solver.exportToVTK(solution, "laplacian_solution.vtk", xmin, ymin, zmin);
    
    return 0;
}