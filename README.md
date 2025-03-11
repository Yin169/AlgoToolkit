# FASTSolver: Advanced Scientific Computing Framework
## Overview
FASTSolver is a high-performance scientific computing framework designed for numerical computations, PDE solving, and scientific data visualization. It combines modern C++ techniques with efficient algorithms for maximum performance.

## Core Components

### Linear Algebra Components
- **Krylov Subspace Methods**
  - <mcsymbol name="GMRES" filename="GMRES.hpp" path="src/LinearAlgebra/Krylov/GMRES.hpp" startline="153" type="function">GMRES Solver</mcsymbol> with Arnoldi iteration and ILU preconditioning
  - Conjugate Gradient method for symmetric positive-definite systems
  - Flexible preconditioning support
- **Matrix Factorizations**
  - LU Decomposition with partial pivoting
  - Cholesky Decomposition
  - ILU Preconditioning for sparse systems
- **Matrix Operations**
  - Dense and sparse matrix operations
  - Matrix-vector multiplication
  - Matrix transpose and element-wise operations

  ![GMRES](https://github.com/Yin169/FASTSolver/blob/dev/doc/pic_2.png)

### Object Types
- **Dense Matrix Object** <mcsymbol name="DenseObj" filename="DenseObj.hpp" path="src/Obj/DenseObj.hpp" startline="62" type="class"></mcsymbol>
  - Row/column operations
  - Matrix arithmetic
  - BLAS-level operations
- **Sparse Matrix Object** <mcsymbol name="SparseMatrixCSC" filename="SparseObj.hpp" path="src/Obj/SparseObj.hpp" startline="76" type="class"></mcsymbol>
  - Compressed Sparse Column (CSC) format
  - Efficient storage and operations
  - Matrix Market file I/O
- **Vector Object** <mcsymbol name="VectorObj" filename="VectorObj.hpp" path="src/Obj/VectorObj.hpp" startline="1" type="class"></mcsymbol>
  - Basic linear algebra operations
  - Element-wise arithmetic
  - BLAS-compatible memory layout

### PDE Solvers
- **Finite Difference Method (FDM)**
  - <mcsymbol name="NavierStoke" filename="NavierStoke.hpp" path="src/PDEs/FDM/NavierStoke.hpp" startline="33" type="class">Navier-Stokes solver</mcsymbol>
  - 3D incompressible flow simulation
  - Adaptive time stepping
  - Multiple advection schemes (Upwind, Central, QUICK)

<img src="https://github.com/Yin169/FASTSolver/blob/dev/doc/pic_3.png" width="400" alt="JetFlow Simulation">

### ODE Solvers
- **Runge-Kutta Methods** <mcsymbol name="RungeKutta" filename="RungeKutta.hpp" path="src/ODE/RungeKutta.hpp" startline="1" type="class"></mcsymbol>
  - Explicit time integration
  - Adaptive time stepping
  - High-order accuracy

### Utilities
- Matrix Market file reader
- Performance tracking tools
- VTK export for visualization
- Structured grid handlingn

### Advanced Lattice Boltzmann Implementation
  - Full D2Q9 and D3Q19 support
  - Comprehensive boundary condition library
  - Integrated IBM capabilities
- Interactive flow visualization
- Real-time performance analytics
### Development Tools
- Streamlined VTK export
- Dynamic flow visualization
- Comprehensive performance tracking
## System Prerequisites
- Modern C++ (17+)
- CMake 3.15 or newer
- Python 3.9+ (for optional features)
## Quick Start

## Installation

Clone the repository and run the setup script:

```bash
git clone https://github.com/Yin169/FASTSolver.git
cd FASTSolver
bash script_test.sh
```

For Python installation:

```bash
cd FASTSolver
python setup.py install
```

## Usage

Here's an example of how to use the `FASTSolver` module in Python:

```python
import fastsolver as fs
import numpy as np


matrix = fs.SparseMatrix(1, 1)
fs.read_matrix_market("../data/bcsstk08/bcsstk08.mtx", matrix)

# Generate a random exact solution

x_ext = fs.Vector(np.random.rand(matrix.cols()))

# # Generate a random right-hand side
b = matrix * x_ext
x = fs.Vector(matrix.cols())
# # Solve the system using GMRES
gmres = fs.GMRES()

max_iter = 100
krylov_dim = 300
tol = 1e-6
gmres.solve(matrix, b, x, max_iter, krylov_dim, tol)
```

```python
import fastsolver as fs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create output directory if it doesn't exist
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Testing Laplacian Solver...")

# =============================================
# Test 2: Navier-Stokes Solver - Lid-Driven Cavity
# =============================================
print("\nTesting Navier-Stokes Solver...")

# Domain parameters
nx, ny, nz = 41, 41, 5  # More resolution in x-y plane, less in z
dx = dy = dz = 0.025    # Grid spacing
dt = 0.001              # Time step
Re = 100                # Reynolds number

# Create Navier-Stokes solver
ns_solver = fs.NavierStokesSolver3D(nx, ny, nz, dx, dy, dz, dt, Re)

# Set time integration and advection schemes
ns_solver.setTimeIntegrationMethod(fs.TimeIntegration.RK4)
ns_solver.setAdvectionScheme(fs.AdvectionScheme.CENTRAL)

# Initial conditions (fluid at rest)
def u_init(x, y, z):
    return 0.0

def v_init(x, y, z):
    return 0.0

def w_init(x, y, z):
    return 0.0

# Set initial conditions
ns_solver.setInitialConditions(u_init, v_init, w_init)

# Boundary conditions (lid-driven cavity)
def u_bc(x, y, z, t):
    if abs(y - (ny-1)*dy) < 1e-10:  # Top boundary moving with u=1
        return 1.0
    else:  # No-slip on other boundaries
        return 0.0

def v_bc(x, y, z, t):
    return 0.0  # No-slip on all boundaries

def w_bc(x, y, z, t):
    return 0.0  # No-slip on all boundaries

# Set boundary conditions
ns_solver.setBoundaryConditions(u_bc, v_bc, w_bc)

# Enable adaptive time stepping for stability
ns_solver.enableAdaptiveTimeStep(0.5)  # CFL target of 0.5

# Simulation parameters
start_time = 0.0
end_time = 1.0  # Run for 1 time unit
output_frequency = 10

# Store velocity fields at different time steps
u_fields = []
v_fields = []
times = []

# Callback function to store results
def store_results(u, v, w, p, time):
    print(f"Time: {time:.3f}, Max velocity: {max(abs(u.to_numpy())):.3f}")
    u_fields.append(u.to_numpy().reshape(nz, ny, nx))
    v_fields.append(v.to_numpy().reshape(nz, ny, nx))
    times.append(time)

# Run the simulation
print(f"Running Navier-Stokes simulation from t={start_time} to t={end_time}...")
ns_solver.solve(start_time, end_time, output_frequency, store_results)

# Plot the final velocity field (mid-z slice)
if len(u_fields) > 0:
    z_mid = nz // 2
    u_final = u_fields[-1][z_mid]
    v_final = v_fields[-1][z_mid]
    
    # Create a grid for plotting
    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dy, ny)
    X, Y = np.meshgrid(x, y)
    
    # Plot velocity magnitude
    plt.figure(figsize=(10, 8))
    speed = np.sqrt(u_final**2 + v_final**2)
    plt.contourf(X, Y, speed, 20, cmap='viridis')
    plt.colorbar(label='Velocity Magnitude')
    
    # Add velocity vectors (subsample for clarity)
    skip = 2
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               u_final[::skip, ::skip], v_final[::skip, ::skip], 
               scale=10, color='white', alpha=0.7)
    
    plt.title(f'Lid-Driven Cavity Flow at t={times[-1]:.3f}, Re={Re}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(output_dir, "navier_stokes_solution.png"))
    print("Navier-Stokes solution saved to", os.path.join(output_dir, "navier_stokes_solution.png"))

```

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

1. Fork the project
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Pybind11 for seamless C++ and Python integration
- Community contributions to numerical methods and solvers

## Citation

If you use FASTSolver in your research, please cite it as follows:

```markdown
@software{FASTSolver2024,
  author = {NG YIN CHEANG},
  title = {FASTSolver: High-Performance Scientific Computing Framework},
  year = {2024},
  url = {https://github.com/Yin169/FASTSolver}
}
```

