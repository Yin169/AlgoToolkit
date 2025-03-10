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

 Create output directory if it doesn't exist
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Testing Laplacian Solver...")
# =============================================
# Test 1: Laplacian Solver
# =============================================
# Define domain parameters
nx, ny, nz = 30, 30, 30  # Number of grid points
xmin, xmax = 0.0, 1.0    # Domain boundaries in x
ymin, ymax = 0.0, 1.0    # Domain boundaries in y
zmin, zmax = 0.0, 1.0    # Domain boundaries in z

# Create Laplacian solver
laplacian_solver = fs.LaplacianSolver(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)

# Define boundary conditions (a simple example with a heated wall)
def boundary_condition(x, y, z):
    if abs(x - xmin) < 1e-10:  # Left wall is hot (temperature = 1)
        return 1.0
    elif abs(x - xmax) < 1e-10:  # Right wall is cold (temperature = 0)
        return 0.0
    else:  # Other walls are insulated (zero gradient)
        return 0.0

# Define source term (zero for Laplace equation)
def source_term(x, y, z):
    return 0.0

# Solve the Laplace equation
print("Solving Laplace equation...")
solution = laplacian_solver.solve(boundary_condition, source_term, xmin, ymin, zmin, omega=1.5)

# Convert solution to numpy array for visualization
solution_np = np.array(solution.to_numpy()).reshape(nz, ny, nx)

# Visualize a 2D slice of the solution at the middle of the z-axis
z_slice = nz // 2
solution_2d = solution_np[z_slice, :, :]

# Create a grid for plotting
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

# Plot the solution
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, solution_2d, 20, cmap='hot')
plt.colorbar(label='Temperature')
plt.title('Solution of Laplace Equation (z-slice at z={:.2f})'.format(zmin + z_slice * (zmax - zmin) / (nz - 1)))
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(os.path.join(output_dir, "laplacian_solution.png"))
print("Laplacian solution saved to", os.path.join(output_dir, "laplacian_solution.png"))

# Export solution to VTK for 3D visualization
vtk_filename = os.path.join(output_dir, "laplacian_solution.vtk")
laplacian_solver.exportToVTK(solution, vtk_filename, xmin, ymin, zmin)
print("Laplacian solution exported to VTK:", vtk_filename)
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

