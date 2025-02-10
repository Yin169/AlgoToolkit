#include "../application/CFD/VorticityStreamSolver.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <filesystem>

template<typename T>
void writeVTK(const std::string& filename,
              const VectorObj<T>& vorticity,
              const VectorObj<T>& streamFunction,
              const VectorObj<T>& u_velocity,
              const VectorObj<T>& v_velocity,
              int nx, int ny,
              double dx, double dy) {
    std::ofstream file(filename);
    file << std::scientific << std::setprecision(6);
    
    // Write VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Cavity Flow Solution\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    
    // Write dimensions
    file << "DIMENSIONS " << nx << " " << ny << " 1\n";
    
    // Write points
    file << "POINTS " << nx * ny << " double\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << i * dx << " " << j * dy << " 0.0\n";
        }
    }
    
    // Write point data
    file << "\nPOINT_DATA " << nx * ny << "\n";
    
    // Write vorticity field
    file << "SCALARS vorticity double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nx * ny; ++i) {
        file << vorticity[i] << "\n";
    }
    
    // Write stream function
    file << "\nSCALARS streamFunction double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nx * ny; ++i) {
        file << streamFunction[i] << "\n";
    }
    
    // Write velocity field
    file << "\nVECTORS velocity double\n";
    for (int i = 0; i < nx * ny; ++i) {
        file << u_velocity[i] << " " << v_velocity[i] << " 0.0\n";
    }
    
    file.close();
}

int main() {
    // Parameters
    const int nx = 32;
    const int ny = 32;
    const double Re = 100.0;    // Reynolds number
    const double dt = 1/32;     // Time step increased for faster convergence
    const int nsteps = 100000;    // Reduced total time steps
    const int output_interval = 100;  // Output interval
    const double U = 1.0;        // Lid velocity
    const double tolerance = 1e-5;// Relaxed convergence tolerance
    const double cfl_limit = 0.5; // Reduced CFL limit for better stability

    // Create output directory
    std::filesystem::create_directories("output");

    // Create solver
    VorticityStreamSolver<double> solver(nx, ny, Re, U, tolerance, cfl_limit);

    // Solve and output results periodically
    bool converged = false;
    for (int step = 0; step < nsteps && !converged; ++step) {
        converged = solver.solve(dt, 1000);

        if (step % output_interval == 0) {
            std::string filename = "output/cavity_flow_" + std::to_string(step) + ".vtk";
            
            writeVTK<double>(filename, 
                          solver.getVorticity(),
                          solver.getStreamFunction(),
                          solver.getUVelocity(),
                          solver.getVVelocity(),
                          nx, ny,
                          solver.getDx(),
                          solver.getDy());
            
            std::cout << "Step " << step << " completed." << std::endl;
        }
    }

    if (converged) {
        std::cout << "Simulation converged successfully!" << std::endl;
    } else {
        std::cout << "Maximum iterations reached without convergence." << std::endl;
    }

    return 0;
}