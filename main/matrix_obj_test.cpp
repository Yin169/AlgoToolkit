#include <gtest/gtest.h>
#include "../src/Obj/MatrixObj.hpp"

// Test fixture for MatrixObj
class MatrixObjTest : public ::testing::Test {
protected:
    MatrixObj<int> matrix1;
    MatrixObj<int> matrix2;
    MatrixObj<int> matrix3;

    void SetUp() override {
        // Initialize matrices for testing
        matrix1 = MatrixObj<int>(2, 3); // 2x3 matrix
        matrix2 = MatrixObj<int>(2, 3); // another 2x3 matrix
        matrix3 = MatrixObj<int>(3, 2); // 3x2 matrix, for dimension mismatch test

        // Populate matrix1 with some values
        for (int i = 0; i < 2 * 3; ++i) {
            matrix1[i] = i;
        }
        // Populate matrix2 with some values for testing addition and subtraction
        for (int i = 0; i < 2 * 3; ++i) {
            matrix2[i] = i + 3; // Start from 3 and go up to 7
        }
    }

    void TearDown() override {
        // Clean up if necessary
    }
};

// Test for constructor and default values
TEST_F(MatrixObjTest, DefaultConstructor) {
    MatrixObj<int> testMatrix(2, 2);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(testMatrix[i], 0);
    }
}

// Test for getting dimensions
TEST_F(MatrixObjTest, GetDimensions) {
    EXPECT_EQ(matrix1.get_row(), 2);
    EXPECT_EQ(matrix1.get_col(), 3);
}

// Test for setting dimensions (should throw an exception)
TEST_F(MatrixObjTest, SetDimensions) {
    EXPECT_THROW(matrix1.setDim(3, 3), std::invalid_argument);
}

// Test for addition operator
TEST_F(MatrixObjTest, Addition) {
    MatrixObj<int> result = matrix1 + matrix2;
    for (int i = 0; i < 2 * 3; ++i) {
        EXPECT_EQ(result[i], matrix1[i] + matrix2[i]);
    }
}

// Test for subtraction operator
TEST_F(MatrixObjTest, Subtraction) {
    MatrixObj<int> result = matrix1 - matrix2;
    for (int i = 0; i < 2 * 3; ++i) {
        EXPECT_EQ(result[i], matrix1[i] - matrix2[i]);
    }
}

// Test for multiplication by a scalar
TEST_F(MatrixObjTest, ScalarMultiplication) {
    matrix1 *= 2;
    for (int i = 0; i < 2 * 3; ++i) {
        EXPECT_EQ(matrix1[i], 2 * i);
    }
}

// Test for matrix multiplication
TEST_F(MatrixObjTest, MatrixMultiplication) {
    MatrixObj<int> matrix4(3, 2); // Change to 3x2 for multiplication with matrix1
    for (int i = 0; i < 3 * 2; ++i) {
        matrix4[i] = i + 1; // Populate matrix4 with some values
    }
    MatrixObj<int> result = matrix1 * matrix4;
    EXPECT_EQ(result.get_row(), 2);
    EXPECT_EQ(result.get_col(), 2);
    // Add more specific tests for the values in the result matrix
    // For example, result[0] should be matrix1[0]*matrix4[0] + matrix1[1]*matrix4[2] + ...
}

// Test for transpose
TEST_F(MatrixObjTest, Transpose) {
    MatrixObj<int> transposed = matrix1.Transpose();
    EXPECT_EQ(transposed.get_row(), 3);
    EXPECT_EQ(transposed.get_col(), 2);
    // Add more specific tests for the values in the transposed matrix
}

// Test for exception when multiplying matrices of incompatible dimensions
TEST_F(MatrixObjTest, IncompatibleDimensionsMultiplication) {
    EXPECT_THROW(matrix1 * matrix2, std::invalid_argument);
}

// Test for accessing elements
TEST_F(MatrixObjTest, ElementAccess) {
    EXPECT_EQ(matrix1[0], 0);
    EXPECT_EQ(matrix1[5], 5);
}

// Test for getting a column
TEST_F(MatrixObjTest, GetColumn) {
    VectorObj<int> col = matrix1.get_Col(1);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(col[i], matrix1[( (1 * 2) + i)]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}