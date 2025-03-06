# FASTSolver: Advanced Scientific Computing Framework
## Overview
FASTSolver is a state-of-the-art scientific computing framework that delivers exceptional performance for numerical computations, CFD simulations, matrix operations, and scientific data visualization.
## Core Capabilities
### Advanced Numerical Methods
- Optimized linear algebra computations
  - LU Decomposition
  - Cholesky Decomposition
  - ILU (Incomplete LU) Factorization
- High-performance iterative solvers
  - GMRES (Generalized Minimal Residual)
  - CG (Conjugate Gradient)
- Adaptive multi-grid algorithms
- Robust ODE integration
  - Runge-Kutta Methods
- Advanced Newton-Raphson Implementation
  - Efficient nonlinear system resolution
  - Flexible Jacobian matrix handling
  - Seamless GMRES integration
  - Sophisticated convergence monitoring

![GMRES](https://github.com/Yin169/FASTSolver/blob/dev/doc/pic_2.png)

### CFD Engine
- Advanced Lattice Boltzmann Implementation
  - Full D2Q9 and D3Q19 support
  - Comprehensive boundary condition library
  - Integrated IBM capabilities
- Interactive flow visualization
- Real-time performance analytics
### Development Tools
- Streamlined VTK export
- Dynamic flow visualization
- Comprehensive performance tracking
![JetFlow](https://github.com/Yin169/FASTSolver/blob/dev/doc/pic_1.png)
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
iimport fastsolver as fs
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

