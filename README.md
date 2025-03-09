# FASTSolver: Advanced Scientific Computing Framework
## Overview
FASTSolver is a state-of-the-art scientific computing framework that delivers exceptional performance for numerical computations, PDEs computation, matrix operations, and scientific data visualization.

## Core Components

### Linear Algebra Components
- **Krylov Subspace Methods**
  - <mcsymbol name="GMRES" filename="GMRES.hpp" path="src/LinearAlgebra/Krylov/GMRES.hpp" startline="153" type="function">GMRES Solver</mcsymbol> with Arnoldi iteration and upper Hessenberg matrix handling
  - Back substitution solution update (Hessenberg system resolution)
- **Matrix Factorizations**
  - LU Decomposition with partial pivoting (row swapping)
  - Cholesky Decomposition for symmetric positive-definite matrices
  - ILU Preconditioning for sparse systems
- **Matrix Operations**
  - Dense matrix-vector multiplication (BLAS optimized)
  - Sparse matrix-vector multiplication (OpenMP parallelized)
  - Matrix transpose operations
  - Element-wise matrix operations (addition, subtraction)

  ![GMRES](https://github.com/Yin169/FASTSolver/blob/dev/doc/pic_2.png)

### Object Types
- **Dense Matrix Object** <mcsymbol name="DenseObj" filename="DenseObj.hpp" path="src/Obj/DenseObj.hpp" startline="62" type="class"></mcsymbol>
  - Row/column swapping operations
  - BLAS-level matrix-vector multiplication
  - Element-wise arithmetic operations
  - Matrix transposition with OpenMP parallelization
- **Sparse Matrix Object** <mcsymbol name="SparseMatrixCSC" filename="SparseObj.hpp" path="src/Obj/SparseObj.hpp" startline="76" type="class"></mcsymbol> (CSC format)
  - Compressed Sparse Column (CSC) storage
  - Scalar multiplication and matrix addition
  - Parallelized sparse matrix-vector product
  - Matrix Market file I/O support
- **Vector Object** <mcsymbol name="VectorObj" filename="VectorObj.hpp" path="src/Obj/VectorObj.hpp" startline="1" type="class"></mcsymbol>
  - Basic linear algebra operations
  - Element-wise arithmetic operations
  - BLAS-compatible memory layout

### PDE Solvers
- **Finite Difference Method (FDM)**
  - <mcsymbol name="NavierStoke" filename="NavierStoke.hpp" path="src/PDEs/FDM/NavierStoke.hpp" startline="33" type="class">Navier-Stokes solver</mcsymbol>
  - Time-stepping algorithm with pressure projection
  - 3D spatial indexing and boundary condition handling
  - Reynolds number parameterization

 ![laplacian](https://github.com/Yin169/FASTSolver/blob/dev/doc/laplacian.png)

### ODE Solvers
- **Runge-Kutta Methods** <mcsymbol name="RungeKutta" filename="RungeKutta.hpp" path="src/ODE/RungeKutta.hpp" startline="1" type="class"></mcsymbol>
  - Explicit time integration schemes
  - Adaptive time-stepping capabilities

### Utilities
- Matrix Market file reader <mcsymbol name="readMatrixMarket" filename="utils.hpp" path="src/utils.hpp" startline="31" type="function"></mcsymbol>
- Performance tracking and analysis tools
- VTK export for visualization

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

