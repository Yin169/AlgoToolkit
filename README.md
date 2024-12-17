# Numerical Solvers and Linear Algebra Tools

Welcome to **numerical_solver**, a high-performance numerical analysis and scientific computation library written in C++ with Python bindings via **Pybind11**. This project is designed to provide fast, accurate, and lightweight tools for solving linear systems, performing decompositions, and other numerical operations.

---

## Features

- **LU Decomposition**: Perform LU decomposition with partial pivoting for matrix factorization.
- **Power Iteration**: Estimate dominant eigenvalues using the power iteration method.
- **Rayleigh Quotient**: Efficiently compute the Rayleigh quotient for matrix-vector systems.
- **Krylov Subspace Methods**: Use Arnoldi iteration for orthonormalization in Krylov subspaces.
- **Conjugate Gradient Solver**: Solve large, sparse linear systems iteratively.
- **Gradient Descent and Static Iterative Methods**: Iterative solvers for optimization problems and linear systems.

---

## Requirements

- **C++11 or higher**
- **Python 3.6+**
- **Pybind11**
- **CMake** (optional for build automation)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/numerical_solver.git
cd numerical_solver
```

### 2. Build the Module

#### With `setup.py`:
```bash
pip install pybind11 setuptools
python setup.py build_ext --inplace
```

#### With CMake (Optional):
```bash
mkdir build && cd build
cmake ..
make
```

---

## Usage

Here is an example of how to use the `numerical_solver` module in Python:

```python
import numerical_solver as ns

# Example: LU decomposition
A = [[4, 1], [1, 3]]
P = [0, 1]
ns.pivot_lu(A, P)
print("LU Decomposition Result:", A, "Pivot:", P)

# Example: Solve using Conjugate Gradient
solver = ns.ConjugateGrad()
solver.call_update()
```

---

## Project Structure

```
/your-project-root
|-- LU.hpp                  # LU Decomposition implementation
|-- basic.hpp               # Basic operations (Power Iteration, QR factorization, etc.)
|-- ConjugateGradient.hpp   # Conjugate Gradient Solver
|-- KrylovSubspace.hpp      # Krylov Subspace Methods
|-- IterSolver.hpp          # Iterative Solvers (Gradient Descent, etc.)
|-- pybind_interface.cpp    # Pybind11 interface for Python bindings
|-- setup.py                # Python setup script
|-- README.md               # Project documentation
```

---

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

1. Fork the project
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

## Author

**Your Name**  
[Your Email](yincheang.ng@outlook.com) 
[GitHub Profile](https://github.com/Yin169)

---

## Acknowledgments

- Pybind11 for seamless C++ and Python integration
- Community contributions to numerical methods and solvers

