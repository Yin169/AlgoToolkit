#include <gtest/gtest.h>
#include "DenseObj.hpp"
#include "VectorObj.hpp"

// Test fixture for DenseObj
class DenseObjTest : public ::testing::Test {
protected:
    DenseObj<double> matrix1;
    DenseObj<double> matrix2;
    DenseObj<double> matrix3;

    void SetUp() override {
        matrix1 = DenseObj<double>(2, 3); // 2x3 matrix
        matrix2 = DenseObj<double>(2, 3); // another 2x3 matrix
        matrix3 = DenseObj<double>(3, 2); // 3x2 matrix for dimension mismatch tests

        // Initialize values in matrix1 and matrix2
        for (int i = 0; i < matrix1.getRows(); ++i) {
            for (int j = 0; j < matrix1.getCols(); ++j){
                matrix1(i, j) = i * matrix1.getCols() + j;        // 0, 1, 2, ...
                matrix2(i, j) = i * matrix2.getCols() + j + 3;    // 3, 4, 5, ...
            }
        }
    }
};

// Test for default construction and initialization
TEST_F(DenseObjTest, DefaultConstructor) {
    DenseObj<double> testMatrix(2, 2);
    ASSERT_EQ(testMatrix.getRows(), 2);
    ASSERT_EQ(testMatrix.getCols(), 2);
    for (int i = 0; i < testMatrix.size(); ++i) {
        EXPECT_EQ(testMatrix[i], 0.0);
    }
}

// Test for getting matrix dimensions
TEST_F(DenseObjTest, GetDimensions) {
    ASSERT_EQ(matrix1.getRows(), 2);
    ASSERT_EQ(matrix1.getCols(), 3);
}

// Test for addition of matrices
TEST_F(DenseObjTest, Addition) {
    DenseObj<double> result = matrix1 + matrix2;
    ASSERT_EQ(result.getRows(), 2);
    ASSERT_EQ(result.getCols(), 3);
    for (int i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], matrix1[i] + matrix2[i]);
    }
}

// Test for subtraction of matrices
TEST_F(DenseObjTest, Subtraction) {
    DenseObj<double> result = matrix1 - matrix2;
    ASSERT_EQ(result.getRows(), 2);
    ASSERT_EQ(result.getCols(), 3);
    for (int i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], matrix1[i] - matrix2[i]);
    }
}

// Test for scalar multiplication
TEST_F(DenseObjTest, ScalarMultiplication) {
    matrix1 *= 2.0;
    for (int i = 0; i < matrix1.getRows(); ++i) {
        for (int j = 0; j < matrix1.getCols(); ++j){
            EXPECT_EQ(matrix1(i, j), 2.0 * ( i * matrix1.getCols() + j));
        }
    }
}

// Test for matrix multiplication
TEST_F(DenseObjTest, MatrixMultiplication) {
    DenseObj<double> matrix4(3, 2);

    for (int i = 0; i < matrix4.getRows(); ++i) {
        for (int j = 0; j < matrix4.getCols(); ++j){
            matrix1(j, i) = (j * matrix1.getCols()) + i;
            matrix4(i, j) = (i * matrix4.getCols()) + j + 1; // 1, 2, 3, ...
        }
    }

    DenseObj<double> result = matrix1 * matrix4;

    ASSERT_EQ(result.getRows(), 2);
    ASSERT_EQ(result.getCols(), 2);

    // Verify specific values in the result matrix
    EXPECT_DOUBLE_EQ(result(0, 0), 13.0);  // 0*1 + 1*3 + 2*5
    EXPECT_DOUBLE_EQ(result(0, 1), 16.0);  // 0*2 + 1*4 + 2*6
    EXPECT_DOUBLE_EQ(result(1, 0), 40.0);  // 3*1 + 4*3 + 5*5
    EXPECT_DOUBLE_EQ(result(1, 1), 52.0);  // 3*2 + 4*4 + 5*6
}

// Test for matrix transpose
TEST_F(DenseObjTest, Transpose) {
    DenseObj<double> transposed = matrix1.Transpose();
    ASSERT_EQ(transposed.getRows(), matrix1.getCols());
    ASSERT_EQ(transposed.getCols(), matrix1.getRows());

    for (int i = 0; i < transposed.getRows(); ++i) {
        for (int j = 0; j < transposed.getCols(); ++j) {
            EXPECT_EQ(transposed(i, j), matrix1(j, i));
        }
    }
}

// Test for incompatible dimensions in multiplication
TEST_F(DenseObjTest, IncompatibleDimensionsMultiplication) {
    EXPECT_THROW(matrix1 * matrix2, std::invalid_argument);
}

// Test for accessing elements
TEST_F(DenseObjTest, ElementAccess) {
    ASSERT_EQ(matrix1[0], 0.0);
    ASSERT_EQ(matrix1[5], 5.0);

    EXPECT_THROW(matrix1[-1], std::out_of_range);
    EXPECT_THROW(matrix1[matrix1.size()], std::out_of_range);
}

// Test for getting a column as a vector
TEST_F(DenseObjTest, GetColumn) {
    VectorObj<double> col = matrix1.getColumn(1);
    ASSERT_EQ(col.size(), matrix1.getRows());
    for (int i = 0; i < col.size(); ++i) {
        EXPECT_EQ(col[i], matrix1(i, 1));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
