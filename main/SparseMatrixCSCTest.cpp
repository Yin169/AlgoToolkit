#include <gtest/gtest.h>
#include "VectorObj.hpp"
#include "SparseObj.hpp"

// Test fixture for SparseMatrixCSC
class SparseMatrixCSCTest : public ::testing::Test {
protected:
    SparseMatrixCSC<int> matA;
    SparseMatrixCSC<int> matB;

    SparseMatrixCSCTest() : matA(3, 3), matB(3, 3) {}

    void SetUp() override {
        // Initialize matA
        matA.addValue(0, 0, 1);
        matA.addValue(1, 1, 2);
        matA.addValue(2, 2, 3);
        matA.finalize();

        // Initialize matB
        matB.addValue(0, 0, 4);
        matB.addValue(1, 1, 5);
        matB.addValue(2, 2, 6);
        matB.finalize();
    }
};

// Test: Accessing matrix elements
TEST_F(SparseMatrixCSCTest, ElementAccess) {
    EXPECT_EQ(matA(0, 0), 1);
    EXPECT_EQ(matA(1, 1), 2);
    EXPECT_EQ(matA(2, 2), 3);
    EXPECT_EQ(matA(0, 1), 0); // Non-existent element should return 0
}

// Test: Addition of two matrices
TEST_F(SparseMatrixCSCTest, MatrixAddition) {
    auto matC = matA + matB;
    EXPECT_EQ(matC(0, 0), 5); // 1 + 4
    EXPECT_EQ(matC(1, 1), 7); // 2 + 5
    EXPECT_EQ(matC(2, 2), 9); // 3 + 6
    EXPECT_EQ(matC(0, 1), 0); // Non-existent elements remain 0
}

// Test: Multiplication of two matrices
TEST_F(SparseMatrixCSCTest, MatrixMultiplication) {
    SparseMatrixCSC<int> matC(3, 3);
    matC.addValue(0, 1, 2);
    matC.addValue(1, 2, 3);
    matC.finalize();

    auto result = matA * matC;

    EXPECT_EQ(result(0, 1), 2); // 1 * 2
    EXPECT_EQ(result(1, 2), 6); // 2 * 3
    EXPECT_EQ(result(2, 0), 0); // No interaction
}

// Test: Get column as VectorObj
TEST_F(SparseMatrixCSCTest, GetColumn) {
    auto col = matA.getColumn(1); // Column 1 of matA
    EXPECT_EQ(col[0], 0); // Row 0
    EXPECT_EQ(col[1], 2); // Row 1
    EXPECT_EQ(col[2], 0); // Row 2
}

// Test: Out-of-bounds element access
TEST_F(SparseMatrixCSCTest, OutOfBoundsAccess) {
    EXPECT_THROW(matA(3, 0), std::out_of_range);
    EXPECT_THROW(matA(0, 3), std::out_of_range);
}

// Test: Adding incompatible matrices
TEST_F(SparseMatrixCSCTest, IncompatibleMatrixAddition) {
    SparseMatrixCSC<int> matC(2, 2); // Incompatible dimensions
    EXPECT_THROW(matA + matC, std::invalid_argument);
}

// Test: Multiplying incompatible matrices
TEST_F(SparseMatrixCSCTest, IncompatibleMatrixMultiplication) {
    SparseMatrixCSC<int> matC(4, 4); // Incompatible dimensions
    EXPECT_THROW(matA * matC, std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
