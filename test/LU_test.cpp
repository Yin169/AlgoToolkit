#include <gtest/gtest.h>
#include "basic.hpp"
#include "DenseObj.hpp"

// Helper function to initialize a square matrix
DenseObj<double> initializeSquareMatrix(int n) {
    // Example data for a 3x3 matrix
    std::vector<double> data = {
        4, 3, 2,
        3, 4, 1,
        2, 1, 4
    };
    return DenseObj<double>(data, n, n);
}

// Test case: Check if PivotLU throws an exception for non-square matrices.
// TEST(LUDecomposition, NonSquareMatrix) {
//     DenseObj<double> A(3, 4); // Non-square matrix
//     std::vector<int> P;

//     EXPECT_THROW(LU::PivotLU<double, DenseObj<double>>(A, P), std::invalid_argument);
// }

// Test case: Check if PivotLU correctly performs LU decomposition on a square matrix.
TEST(LUDecomposition, SquareMatrix) {
    const int n = 3;

    // Initialize matrix A with known values.
    DenseObj<double> A = initializeSquareMatrix(n);
    std::vector<int> P;

    // Copy the original matrix for comparison
    DenseObj<double> originalA = A;

    // Perform LU decomposition
    basic::PivotLU<double, DenseObj<double>>(A, P);

    // Initialize L and U matrices
    DenseObj<double> L(n, n); // Start with all zeros
    DenseObj<double> U(n, n);

    // Extract L and U from the decomposed matrix A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > j) {
                L(i, j) = A(i, j); // Lower triangular part
            } else {
                U(i, j) = A(i, j); // Upper triangular part
            }
        }
        L(i, i) = 1.0; // Set diagonal elements of L to 1
    }

    // Apply permutation vector P to reorder rows of the original matrix
    DenseObj<double> PA = originalA;
    for (int i = 0; i < n; ++i) {
        PA.swapRows(i, P[i]);
    }

    // Compute L * U
    DenseObj<double> L_times_U = L * U;

    // Verify if PA is approximately equal to L * U
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(PA(i, j), L_times_U(i, j), 1e-8)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }

    // Additional check: Verify L is lower triangular and U is upper triangular
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            EXPECT_NEAR(U(i, j), 0.0, 1e-8) << "U should be upper triangular";
        }
        for (int j = i + 1; j < n; ++j) {
            EXPECT_NEAR(L(i, j), 0.0, 1e-8) << "L should be lower triangular";
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
