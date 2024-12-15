#include <gtest/gtest.h>
#include "LU.hpp"
#include "MatrixObj.hpp"

// Test case for checking if the PivotLU function throws an exception for non-square matrices.
TEST(LUDecomposition, NonSquareMatrix) {
    MatrixObj<double> A(3, 4); // Non-square matrix
    std::vector<double> P;
    EXPECT_THROW(LU::PivotLU(A, P), std::invalid_argument);
}

// Test case for checking if the PivotLU function correctly performs LU decomposition on a square matrix.
TEST(LUDecomposition, SquareMatrix) {
    int n = 3;

    // Initialize matrix A with known values.
    double data[] = {4, 3, 2, 3, 4, 1, 2, 1, 4};
    MatrixObj<double>  A(data, n, n);
    std::vector<double> P;

    // Perform LU decomposition.
    LU::PivotLU(A, P);

    // Construct L and U matrices from the decomposed A.
    MatrixObj<double> L(n, n), U(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) {
                U[i + j * n] = A[i + j * n]; // Upper triangular part of U.
            } else {
                L[i + j * n] = A[i + j * n]; // Lower triangular part of L.
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        L[i + i * n] = 1.0; // Diagonal elements of L are 1.
    }

    // Verify if A is approximately equal to L * U
    MatrixObj<double> A_reconstructed = L * U;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(A[i + j * n], A_reconstructed[i + j * n], 1e-8);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}