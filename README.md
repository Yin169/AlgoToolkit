# FASTSolver: A High-Performance Numerical Solution Framework

## Author(s)
`NG YIN CHEANG`

## Description:

 FASTSolver is a versatile numerical solution framework designed to efficiently tackle a wide range of scientific and engineering problems. It leverages high-performance computing techniques to provide fast and accurate solutions, making it ideal for researchers and practitioners who require robust computational tools.

## Key Features:

FASTSolver is a C++ numerical computing library with Python bindings, featuring:
- Linear algebra operations
- Iterative solvers (GMRES, Conjugate Gradient)
- Multi-grid methods
- Numerical integration
- ODE solvers

## Requirements

- **C++17 or higher**
- **Python 3.6+**
- **Pybind11**
- **CMake** (optional for build automation)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Yin169/FASTSolver.git
cd FASTSolver
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

Here is an example of how to use the `FASTSolver` module in Python:

```python
import FASTSolver as ns

# Example: LU decomposition
A = [[4, 1], [1, 3]]
P = [0, 1]
ns.pivot_lu(A, P)
print("LU Decomposition Result:", A, "Pivot:", P)

# Example: Solve using Conjugate Gradient
solver = ns.ConjugateGrad()
x = [0, 0]  # Initialize x with appropriate values
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
├── src
│   ├──  Optimization
│   │   ├── Adjoint.hpp
│   │   └── TrustRegion.hpp
│   ├── LinearAlgebra
│   │   ├── Factorized
│   │   │   └── basic.hpp
│   │   ├── Krylov
│   │   │   ├── ConjugateGradient.hpp
│   │   │   ├── GMRES.hpp
│   │   │   └── KrylovSubspace.hpp
│   │   ├── Preconditioner
│   │   │   ├── LU.hpp
│   │   │   └── MultiGrid.hpp
│   │   └── Solver
│   │       ├── IterSolver.hpp
│   │       └── SolverBase.hpp
│   ├── MixedPrecision
│   └── Obj
│       ├── MatrixObj.hpp
│       ├── SparseObj.hpp
│       └── VectorObj.hpp
├── application
│   ├── LatticeBoltz/
│   ├── LinearAlgebra/
│   └── PostProcess/
├── examples
│   ├── CylinderFlow.cpp
│   ├── JetFlow.cpp
│   └── JetFlow_3D.cpp
├── src
│   ├── Mesh/
│   ├── Optimization/
│   └── Solver/
└── tests/
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

---

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

