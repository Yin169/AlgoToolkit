#include <gtest/gtest.h>
#include "VectorObj.hpp"
#include "SparseObj.hpp"

// Helper function to initialize matrices
template <typename T>
void initializeSparseMatrix(SparseMatrixCSC<T>& mat) {
    mat.addValue(0, 0, 1);
    mat.addValue(1, 1, 2);
    mat.addValue(2, 2, 3);
    mat.finalize();
}

// Test fixture for SparseMatrixCSC
class SparseMatrixCSCTest : public ::testing::Test {
protected:
    SparseMatrixCSC<int> matA;
    SparseMatrixCSC<int> matB;

    SparseMatrixCSCTest() : matA(3, 3), matB(3, 3) {}

    void SetUp() override {
        // Initialize matA and matB using the helper function
        initializeSparseMatrix(matA);
        matB.addValue(0, 0, 4);
        matB.addValue(1, 1, 5);
        matB.addValue(2, 2, 6);
        matB.finalize();
    }
};

TEST(SparseMatrixTest, Construction) {
    SparseMatrixCSC<double> mat(3, 3);
    mat.addValue(0, 2, 3.0);
    mat.addValue(1, 0, 4.0);
    mat.addValue(2, 1, 5.0);
    mat.addValue(2, 2, 6.0);
    mat.finalize();

    EXPECT_EQ(mat(0, 2), 3.0);
    EXPECT_EQ(mat(1, 0), 4.0);
    EXPECT_EQ(mat(2, 1), 5.0);
    EXPECT_EQ(mat(2, 2), 6.0);
    EXPECT_EQ(mat(0, 0), 0.0); // Zero element
}

TEST(SparseMatrixTest, MatrixVectorMultiplication) {
    SparseMatrixCSC<double> mat(3, 3);
    mat.addValue(0, 2, 3.0);
    mat.addValue(1, 0, 4.0);
    mat.addValue(2, 1, 5.0);
    mat.addValue(2, 2, 6.0);
    mat.finalize();

    VectorObj<double> vec(3);
    vec[0] = 1.0;
    vec[1] = 2.0;
    vec[2] = 3.0;

    VectorObj<double> result = mat * vec;
    EXPECT_EQ(result[0], 9.0);  // 0*1 + 0*2 + 3*3
    EXPECT_EQ(result[1], 4.0);  // 4*1 + 0*2 + 0*3
    EXPECT_EQ(result[2], 28.0); // 0*1 + 5*2 + 6*3
}

TEST(SparseMatrixTest, MatrixMatrixMultiplication) {
    SparseMatrixCSC<double> matA(2, 3);
    matA.addValue(0, 0, 1.0);
    matA.addValue(0, 2, 2.0);
    matA.addValue(1, 1, 3.0);
    matA.finalize();

    SparseMatrixCSC<double> matB(3, 2);
    matB.addValue(0, 1, 4.0);
    matB.addValue(1, 0, 5.0);
    matB.addValue(2, 1, 6.0);
    matB.finalize();

    SparseMatrixCSC<double> result = matA * matB;
    EXPECT_EQ(result(0, 0), 0.0);  // 1*0 + 0*5 + 2*0
    EXPECT_EQ(result(0, 1), 16.0); // 1*4 + 0*0 + 2*6
    EXPECT_EQ(result(1, 0), 15.0); // 0*0 + 3*5 + 0*0
    EXPECT_EQ(result(1, 1), 0.0);  // 0*4 + 3*0 + 0*6
}
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

TEST_F(SparseMatrixCSCTest, VectorMultiplication) {
    VectorObj<int>  vec3(3, 0.0);
    vec3[0] = 1.0;
    vec3[1] = 2.0;
    vec3[2] = 3.0;
    auto result = matA * vec3;
    EXPECT_EQ(result[0], 1); // 1 * 1   
    EXPECT_EQ(result[1], 4); // 2 * 2
    EXPECT_EQ(result[2], 9); // 3 * 3
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
