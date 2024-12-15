#include <gtest/gtest.h>
#include "KrylovSubspace.hpp"
#include "MatrixObj.hpp"
#include "VectorObj.hpp"

// Define a test fixture class for the KrylovSubspace tests
class KrylovSubspaceTest : public ::testing::Test {
protected:
    // Objects and setup code that is common to all tests can be added here
    MatrixObj<double> A;
    std::vector<VectorObj<double>> Q;
    MatrixObj<double> H;

    void SetUp() override {
        // Code here will be called immediately after the constructor and
        // right before each test
    }

    void TearDown() override {
        // Code here will be called after each test
    }
};

// Test case for the Arnoldi method
TEST_F(KrylovSubspaceTest, ArnoldiBasic) {
    double A_data[] = {
        1, 0, 0,
        1, 1, 0,
        0, 0, 1
    };
    double r_data[] = {
        1, 1, 1
    };

    VectorObj<double> r(r_data, 3);
    A = MatrixObj<double>(A_data, 3, 3);
    Q.resize(3);
    H = MatrixObj<double>(3, 3);

    Krylov::Arnoldi(A, r, Q, H, 3, 1e-10);

    // Test the first element of H
    double expected_H_0 = 2.0;
    double actual_H_0 = H[0];
    EXPECT_NEAR(expected_H_0, actual_H_0, 1e-10);

    // Test the first column of Q
    double norm = std::sqrt(r_data[0] * r_data[0] + r_data[1] * r_data[1] + r_data[2] * r_data[2]);
    EXPECT_NEAR(Q[0][0], r_data[0] / norm, 1e-10);
    EXPECT_NEAR(Q[0][1], r_data[1] / norm, 1e-10);
    EXPECT_NEAR(Q[0][2], r_data[2] / norm, 1e-10);
}

// Add more tests as needed to cover different scenarios and edge cases

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}