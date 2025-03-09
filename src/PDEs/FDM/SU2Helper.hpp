#include "NavierStoke.hpp"
#include "../SU2Mesh/readMesh.hpp"

// Helper class to integrate SU2Mesh with NavierStokes solver
template <typename TNum>
class SU2NavierStokesSolver {
private:
    SU2Mesh mesh;
    NavierStokesSolver3D<TNum> solver;
    
public:
    SU2NavierStokesSolver(const std::string& mesh_file, TNum dt, TNum Re) {
        // Read mesh
        if (!mesh.readMeshFile(mesh_file)) {
            throw std::runtime_error("Failed to read SU2 mesh file");
        }
        
        if (!mesh.isStructured()) {
            throw std::runtime_error("NavierStokes solver requires structured grid");
        }
        
        // Create solver with mesh dimensions
        int nx = mesh.getNx();
        int ny = mesh.getNy();
        int nz = mesh.getDimension() == 3 ? mesh.getNz() : 1;
        TNum dx = mesh.getDx();
        TNum dy = mesh.getDy();
        TNum dz = mesh.getDimension() == 3 ? mesh.getDz() : 1.0;
        
        solver = NavierStokesSolver3D<TNum>(nx, ny, nz, dx, dy, dz, dt, Re);
        
        // Setup solver with mesh
        mesh.setupNavierStokesSolver(solver);
    }
    
    // Set boundary conditions for specific markers
    void setBoundaryConditions(
        const std::unordered_map<std::string, 
        std::function<void(TNum, TNum, TNum, TNum, TNum&, TNum&, TNum&)>>& bc_funcs) {
        mesh.applyBoundaryConditions(solver, bc_funcs);
    }
    
    // Set initial conditions
    void setInitialConditions(
        std::function<TNum(TNum, TNum, TNum)> u_init,
        std::function<TNum(TNum, TNum, TNum)> v_init,
        std::function<TNum(TNum, TNum, TNum)> w_init) {
        solver.setInitialConditions(u_init, v_init, w_init);
    }
    
    // Run simulation
    void simulate(TNum duration, int output_freq = 1) {
        solver.simulate(duration, output_freq);
    }
    
    // Export solution to VTK
    bool exportSolution(const std::string& filename) {
        return mesh.exportToVTK(filename, 
                               solver.getU(), 
                               solver.getV(), 
                               solver.getW(), 
                               solver.getP());
    }
    
    // Access the solver and mesh
    NavierStokesSolver3D<TNum>& getSolver() { return solver; }
    const SU2Mesh& getMesh() const { return mesh; }
};
