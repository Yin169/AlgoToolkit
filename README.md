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

