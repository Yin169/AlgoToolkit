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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import os

# Add the path to the fastsolver module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath('')), 'build'))

try:
    import fastsolver
except ImportError:
    print("Error: fastsolver module not found. Make sure it's built and in the correct path.")
    sys.exit(1)

# Define an analytic solution to the Poisson equation
# We'll use u(x,y,z) = sin(πx)sin(πy)sin(πz)
# For this function, ∇²u = -3π²sin(πx)sin(πy)sin(πz)

def analytic_solution(x, y, z):
    return np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)

def source_function(x, y, z):
    # The right-hand side of the Poisson equation: ∇²u = f
    # For our analytic solution, f = -3π²sin(πx)sin(πy)sin(πz)
    return -3 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)

def boundary_function(x, y, z):
    # Dirichlet boundary conditions matching the analytic solution
    return analytic_solution(x, y, z)

# Set up the domain and grid
nx, ny, nz = 32, 32, 32  # Number of grid points
lx, ly, lz = 1.0, 1.0, 1.0  # Domain size

# Create the Poisson solver using the new Poisson3DSolver class
poisson_solver = fastsolver.Poisson3DSolver(nx, ny, nz, lx, ly, lz)

# Set the right-hand side function and boundary conditions
poisson_solver.setRHS(source_function)
poisson_solver.setDirichletBC(boundary_function)

# Solve the Poisson equation using Conjugate Gradient
max_iter = 2000
tol = 1e-6
poisson_solver.solve(max_iter, tol)

# Extract the numerical solution
numerical_solution = np.zeros((nx, ny, nz))
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            numerical_solution[i, j, k] = poisson_solver.getSolution(i, j, k)

# Calculate the analytic solution on the grid
hx, hy, hz = lx/(nx-1), ly/(ny-1), lz/(nz-1)
analytic_grid = np.zeros((nx, ny, nz))
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            x, y, z = i*hx, j*hy, k*hz
            analytic_grid[i, j, k] = analytic_solution(x, y, z)

# Calculate error
error = np.abs(numerical_solution - analytic_grid)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Maximum absolute error: {max_error:.6e}")
print(f"L2 error: {l2_error:.6e}")

# Visualize the results
# Let's create a slice through the middle of the domain
mid_z = nz // 2

# Create coordinate grids for plotting
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
X, Y = np.meshgrid(x, y)

# Plot the numerical solution
plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.contourf(X, Y, numerical_solution[:, :, mid_z].T, 20, cmap=cm.viridis)
plt.colorbar(label='Numerical Solution')
plt.title('Numerical Solution (z=0.5)')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(132)
plt.contourf(X, Y, analytic_grid[:, :, mid_z].T, 20, cmap=cm.viridis)
plt.colorbar(label='Analytic Solution')
plt.title('Analytic Solution (z=0.5)')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(133)
plt.contourf(X, Y, error[:, :, mid_z].T, 20, cmap=cm.hot)
plt.colorbar(label='Absolute Error')
plt.title(f'Error (z=0.5), Max: {max_error:.2e}, L2: {l2_error:.2e}')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# 3D visualization of the solution
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a coarser grid for 3D visualization
stride = 4
x_coarse = x[::stride]
y_coarse = y[::stride]
z_coarse = np.linspace(0, lz, nz)[::stride]
X_coarse, Y_coarse, Z_coarse = np.meshgrid(x_coarse, y_coarse, z_coarse, indexing='ij')

# Extract values at the coarser grid
values = numerical_solution[::stride, ::stride, ::stride]

# Create a 3D scatter plot with colors representing the solution values
scatter = ax.scatter(X_coarse.flatten(), Y_coarse.flatten(), Z_coarse.flatten(), 
                    c=values.flatten(), cmap=cm.viridis, s=50, alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of Poisson Solution')
plt.colorbar(scatter, ax=ax, label='Solution Value')

plt.tight_layout()
plt.show()

# Convergence study
grid_sizes = [8, 16, 32, 64]
errors_max = []
errors_l2 = []

for n in grid_sizes:
    print(f"Running convergence test with grid size {n}x{n}x{n}")
    # Use the new Poisson3DSolver for the convergence study
    solver = fastsolver.Poisson3DSolver(n, n, n, lx, ly, lz)
    solver.setRHS(source_function)
    solver.setDirichletBC(boundary_function)
    solver.solve(max_iter, tol)
    
    # Extract solution and calculate error
    h = lx/(n-1)
    numerical = np.zeros((n, n, n))
    analytic = np.zeros((n, n, n))
    
    for k in range(n):
        for j in range(n):
            for i in range(n):
                numerical[i, j, k] = solver.getSolution(i, j, k)
                x, y, z = i*h, j*h, k*h
                analytic[i, j, k] = analytic_solution(x, y, z)
    
    error = np.abs(numerical - analytic)
    errors_max.append(np.max(error))
    errors_l2.append(np.sqrt(np.mean(error**2)))

# Plot convergence
plt.figure(figsize=(10, 6))
h_values = [lx/(n-1) for n in grid_sizes]

plt.loglog(h_values, errors_max, 'o-', label='Max Error')
plt.loglog(h_values, errors_l2, 's-', label='L2 Error')

# Add reference lines for 2nd order convergence
ref_h = np.array([min(h_values), max(h_values)])
ref_error = 0.1 * ref_h**2
plt.loglog(ref_h, ref_error, 'k--', label='O(h²) Reference')

plt.xlabel('Grid Spacing (h)')
plt.ylabel('Error')
plt.title('Convergence Study for 3D Poisson Solver')
plt.legend()
plt.grid(True)
plt.show()

# Print convergence rates
for i in range(1, len(grid_sizes)):
    rate_max = np.log(errors_max[i-1]/errors_max[i]) / np.log(grid_sizes[i]/grid_sizes[i-1])
    rate_l2 = np.log(errors_l2[i-1]/errors_l2[i]) / np.log(grid_sizes[i]/grid_sizes[i-1])
    print(f"Grid refinement {grid_sizes[i-1]} -> {grid_sizes[i]}:")
    print(f"  Max error convergence rate: {rate_max:.2f}")
    print(f"  L2 error convergence rate: {rate_l2:.2f}")

```
<img src="https://github.com/Yin169/FASTSolver/blob/dev/doc/poisson1.png" width="600" alt="Poissson">
<img src="https://github.com/Yin169/FASTSolver/blob/dev/doc/poisson2.png" width="400" alt="Poissson2">

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

