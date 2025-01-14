# FASTSolver: High-Performance Scientific Computing Framework

## Author
`NG YIN CHEANG`

## Overview
FASTSolver is a comprehensive scientific computing framework focused on:
- High-performance numerical solutions
- Computational Fluid Dynamics (CFD)
- Linear algebra operations
- Scientific visualization

## Key Features

### Numerical Methods
- Linear algebra operations
- Iterative solvers (GMRES, CG)
- Multi-grid methods
- ODE solvers

### CFD Capabilities
- Lattice Boltzmann Method (LBM)
  - D2Q9 and D3Q19 models
  - Various boundary conditions
  - Immersed Boundary Method (IBM)
- Flow visualization
- Performance monitoring

### Spectral Element Method (SEM)

The Spectral Element Method (SEM) is a high-order numerical technique for solving partial differential equations (PDEs). It combines the flexibility of finite element methods with the accuracy of spectral methods. The implementation in FASTSolver supports:

- High-order polynomial approximations
- Multi-dimensional problems (1D, 2D, and 3D)
- Customizable PDE operators and boundary conditions
- Integration with iterative solvers like GMRES

### Tools
- VTK output generation
- Real-time flow visualization
- Performance metrics logging

![JetFlow](https://github.com/Yin169/FASTSolver/blob/dev/doc/pic_1.png)

## Requirements
- C++17 or higher
- CMake 3.15+
- Python 3.6+ (optional)
- VTK 9.0+ (for visualization)

## Installation

```bash
git clone https://github.com/Yin169/FASTSolver.git
cd FASTSolver
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
FASTSolver
├── Doxyfile
├── LICENSE
├── README.md
├── application
│   ├── LatticeBoltz
│   │   └── LBMSolver.hpp
│   ├── Mesh
│   │   └── MeshObj.hpp
│   └── PostProcess
│       └── Visual.hpp
├── code
│   ├── test.ipynb
│   └── test.py
├── conanfile.txt
├── data
│   ├── Chem97ZtZ
│   │   └── Chem97ZtZ.mtx
│   └── Chem97ZtZ.tar
├── examples
│   ├── CylinderFlow.cpp
│   ├── JetFlow_2D.cpp
│   └── JetFlow_3D.cpp
├── python
│   └── pybind.cpp
├── script_test.sh
├── src
│   ├── Intergal
│   │   └── GaussianQuad.hpp
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
│   │       └── IterSolver.hpp
│   ├── ODE
│   │   └── RungeKutta.hpp
│   ├── Obj
│   │   ├── DenseObj.hpp
│   │   ├── MatrixObj.hpp
│   │   ├── SparseObj.hpp
│   │   └── VectorObj.hpp
│   ├── PDEs
│   │   └── SpectralElementMethod.hpp
│   └── utils.hpp
└── test
    ├── ConjugateGradient_test.cpp
    ├── GMRES_test.cpp
    ├── GaussianQuad_test.cpp
    ├── KrylovSubspace_test.cpp
    ├── LBMSolver_test.cpp
    ├── LU_test.cpp
    ├── MeshObj_test.cpp
    ├── MultiGrid_test.cpp
    ├── RungeKutta_test.cpp
    ├── SparseMatrixCSCTest.cpp
    ├── SpectralElementMethod_test.cpp
    ├── Visual_test.cpp
    ├── basic_test.cpp
    ├── debuglogger.cpp
    ├── demo.cpp
    ├── itersolver_test.cpp
    ├── matrix_obj_test.cpp
    ├── test.cpp
    └── testfile.cpp
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

