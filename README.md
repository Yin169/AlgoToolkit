# FASTSolver: A High-Performance Numerical Solution Framework

## Author(s)
NG YIN CHEASNG

## Description:

 FASTSolver is a versatile numerical solution framework designed to efficiently tackle a wide range of scientific and engineering problems. It leverages high-performance computing techniques to provide fast and accurate solutions, making it ideal for researchers and practitioners who require robust computational tools.

## Key Features:

**Efficient Solvers**: FASTSolver implements a variety of numerical solvers. These solvers are optimized for performance, ensuring fast computations for complex problems.
**Scalability**: The framework is designed to scale effectively on multi-core and multi-processor systems, allowing you to harness the power of parallel computing for even larger-scale problems.
**User-Friendly Interface**: FASTSolver provides a user-friendly interface that simplifies problem setup and solution extraction. This makes it accessible to users with varying levels of programming experience.
**Customizable**: The framework allows for customization through. This flexibility enables users to tailor FASTSolver to their specific needs.

## Requirements

- **C++17 or higher**
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
solver.solve(x)
```

---

## Project Structure

```
.
├── CMakeLists.txt
├── Doxyfile
├── LICENSE
├── README.md
├── conanfile.txt
├── main
│   ├── ConjugateGradient_test.cpp
│   ├── KrylovSubspace_test.cpp
│   ├── LU_test.cpp
│   ├── SparseMatrixCSCTest.cpp
│   ├── basic_test.cpp
│   ├── demo.cpp
│   ├── itersolver_test.cpp
│   ├── matrix_obj_test.cpp
│   └── test.cpp
├── python
│   └── pybind.cpp
├── script_test.sh
├── setup.py
└── src
    ├──  Optimization
    │   ├── Adjoint.hpp
    │   └── TrustRegion.hpp
    ├── LinearAlgebra
    │   ├── Factorized
    │   │   └── basic.hpp
    │   ├── Krylov
    │   │   ├── ConjugateGradient.hpp
    │   │   ├── GMRES.hpp
    │   │   └── KrylovSubspace.hpp
    │   ├── Preconditioner
    │   │   ├── LU.hpp
    │   │   └── MultiGrid.hpp
    │   └── Solver
    │       ├── IterSolver.hpp
    │       └── SolverBase.hpp
    ├── MixedPrecision
    └── Obj
        ├── MatrixObj.hpp
        ├── SparseObj.hpp
        └── VectorObj.hpp
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

## Acknowledgments

- Pybind11 for seamless C++ and Python integration
- Community contributions to numerical methods and solvers

