# Linear Algebra Toolkit: LAToolkit

Welcome to the LAToolkit, a C++ library designed to provide a robust and efficient set of tools for performing linear algebra operations. This repository contains two core header files: `MatrixObj.hpp` and `basic.hpp`, which together form the foundation of our linear algebra capabilities.

## Overview

### MatrixObj.hpp
This file defines the `MatrixObj` and `VectorObj` classes, which are templates that allow for the creation and manipulation of matrices and vectors with arbitrary data types. The `MatrixObj` class includes methods for matrix addition, subtraction, scalar multiplication, and matrix multiplication, as well as transpose operations. The `VectorObj` class, which inherits from `MatrixObj`, extends these capabilities to one-dimensional arrays, providing additional methods for vector normalization and dot product calculations.

### Basic.hpp
The `basic.hpp` file contains a namespace `basic` that encompasses a variety of linear algebra algorithms and utility functions. These include:

- **Power Iteration**: An iterative method for finding the dominant eigenvalue and its corresponding eigenvector of a matrix.
- **Rayleigh Quotient**: A method for estimating an eigenvalue of a matrix given an eigenvector.
- **Subtractive Projection**: A technique for orthogonalizing vectors.
- **Gram-Schmidt Process**: An algorithm for orthonormalizing a set of vectors.
- **Householder Reflections**: A method for constructing orthogonal matrices, used in the QR decomposition.
- **QR Decomposition**: A factorization of a matrix into an orthogonal matrix Q and an upper triangular matrix R.

## Features

- **Template-Based**: Works with any data type that supports the necessary operations.
- **Efficiency**: Optimized for performance with in-place operations where possible.
- **Extensibility**: Easily add new algorithms and functions to the library.
- **Portability**: Written in standard C++, compatible with any compliant compiler.

## Getting Started

To get started with LAToolkit, simply clone this repository and include the `MatrixObj.hpp` and `basic.hpp` files in your project. You can then instantiate `MatrixObj` and `VectorObj` with your desired data type and utilize the functions provided in the `basic` namespace.

### Usage Example

```cpp
#include "MatrixObj.hpp"
#include "basic.hpp"

using namespace basic;

int main() {
    MatrixObj<double> A(2, 2); // Create a 2x2 matrix
    VectorObj<double> b(2);    // Create a 2-dimensional vector

    // Initialize A and b with values...

    powerIter(A, b, 1000); // Perform power iteration

    return 0;
}
```

## Contributing

We welcome contributions to LAToolkit! If you have an algorithm you'd like to add or find a bug, please submit a pull request or open an issue.

## License

LAToolkit is released under the [MIT License](LICENSE). Feel free to use, modify, and distribute this software as long as you comply with the license.
