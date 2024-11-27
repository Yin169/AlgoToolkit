# MatrixObj and Basic Linear Algebra Operations Library

This repository contains a C++ template library for basic linear algebra operations, focusing on matrix and vector manipulation. The library is designed to be efficient, easy to use, and templated for flexibility with different data types.

## Overview

The library consists of two main components:

1. **MatrixObj.hpp**: A template class for matrix operations.
2. **basic.hpp**: A namespace containing basic linear algebra algorithms and operations.

## MatrixObj.hpp

This file defines the `MatrixObj` and `VectorObj` classes, which are templated to work with any data type (`TObj`). The `MatrixObj` class provides a robust interface for matrix operations such as addition, subtraction, scalar multiplication, matrix multiplication, and transposition. The `VectorObj` class is derived from `MatrixObj` and is specialized for vector operations, including L2 norm calculation and normalization.

### Key Features of MatrixObj

- **Dynamic Memory Management**: Allocates and deallocates memory automatically.
- **Copy and Move Semantics**: Implements copy and move constructors and assignment operators for efficient memory management.
- **Element Access**: Provides `operator[]` for accessing matrix elements.
- **Matrix Operations**: Supports matrix addition, subtraction, scalar multiplication, and matrix multiplication.
- **Transpose**: Easily transpose a matrix.
- **Slice**: Extract a slice of the matrix.

### Key Features of VectorObj

- **L2 Norm and Normalization**: Computes the L2 norm and normalizes the vector.
- **Element Access**: Provides `operator[]` for accessing vector elements.
- **Dot Product**: Implements the dot product operation between two vectors.

## basic.hpp

This file introduces the `basic` namespace, which contains several template functions for performing fundamental linear algebra operations:

- **powerIter**: Implements the power iteration method for finding the dominant eigenvalue and eigenvector of a matrix.
- **rayleighQuotient**: Calculates the Rayleigh quotient, which is used in eigenvalue problems.
- **subtProj**: Performs a vector projection by subtracting the projection of one vector onto another.
- **gramSmith**: Applies the Gram-Schmidt process to a set of vectors to produce an orthogonal set.

## Usage

To use this library, simply include the `MatrixObj.hpp` and `basic.hpp` files in your project and start utilizing the provided classes and functions. Since the library is templated, you can use it with various data types, such as `int`, `float`, `double`, or even custom types.

## Building and Testing

The library does not require any external dependencies other than the C++ Standard Library. To build and test the library, you can use any standard C++ compiler that supports C++11 or later. We recommend using a build system like CMake for ease of compilation and testing.

## Contributing

Contributions to the library are welcome! If you find any bugs, have suggestions for improvements, or want to add new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

 
