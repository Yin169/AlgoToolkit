 # MyAlgorithmicToolkit

A comprehensive C++ library for numerical algorithms, linear algebra operations, and computational fluid dynamics simulations.

## Overview

MyAlgorithmicToolkit is a collection of efficient, template-based implementations of common algorithms and data structures used in scientific computing and numerical simulations. The toolkit is designed to be modular, extensible, and performance-oriented.

## Key Components

### Linear Algebra

- **Matrix and Vector Operations**: Template-based implementations for dense and sparse matrices/vectors
- **Iterative Solvers**: Conjugate Gradient, Jacobi, and other methods for solving linear systems
- **Eigenvalue Algorithms**: Power iteration, Rayleigh quotient methods
- **Matrix Decompositions**: QR, LU decompositions

### Partial Differential Equations (PDEs)

- **Finite Difference Methods**: Implementation for Navier-Stokes equations
- **Mesh Handling**: Adaptors for transforming between physical and computational spaces
- **Boundary Conditions**: Various implementations for different boundary types

### Computational Fluid Dynamics

- **Lattice Boltzmann Methods**: Framework for LBM simulations
- **Cavity Flow**: Benchmark implementations
- **Cylinder Flow**: 2D and 3D implementations

### Utilities

- **Benchmarking Tools**: Performance measurement for various algorithms
- **I/O Utilities**: VTK output for visualization
- **Testing Framework**: Comprehensive tests for all components

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

```
<img src="https://github.com/Yin169/FASTSolver/blob/dev/doc/poisson1.png" width="600" alt="Poissson">
<img src="https://github.com/Yin169/FASTSolver/blob/dev/doc/poisson2.png" width="400" alt="Poissson2">

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

