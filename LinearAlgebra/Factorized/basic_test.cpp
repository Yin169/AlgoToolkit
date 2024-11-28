#include "basic.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>

namespace {

class BasicTest : public ::testing::Test {
protected:
    MatrixObj<double> A;
    VectorObj<double> b;
    double setA[3*3] = {1,0,0,0,1,0,0,0,1};
    double setB[3] = {1, 0, 0};

    void SetUp() override {
        A = MatrixObj<double>(3, 3);
        for (int i = 0; i < 3*3; ++i) {
            A.data()[i] = setA[i];
        }
        b = VectorObj<double>(setB, 3);
    }
};

TEST_F(BasicTest, RayleighQuotientTest) {
    double quotient = basic::rayleighQuotient(A, b);
    ASSERT_NEAR(quotient, 1, 1e-6) << "Rayleigh quotient test failed. Actual: " << quotient;
}

TEST_F(BasicTest, PowerIterTest) {
    VectorObj<double> b_init = b;
    basic::powerIter(A, b, 10);
    for (int i = 0; i < b.get_row(); ++i) { 
        ASSERT_NEAR(b_init[i], b[i], 1e-6) << "Power iteration test failed at index " << i << ". Actual: " << b[i];
    }
}

TEST_F(BasicTest, GramSmithTest) {
    std::cout << "Starting Gram-Schmidt test..." << std::endl;
    int m = A.get_col();
    std::vector<VectorObj<double>> orthSet(m);
    basic::gramSmith(A, orthSet);

    for (int i = 0; i < m; ++i) {
        std::cout << "Checking orthogonality for vector " << i << "..." << std::endl;
        for (int j = i + 1; j < m; ++j) {
            auto dot_product = orthSet[i] * orthSet[j];
            ASSERT_NEAR(dot_product, 0.0, 1e-6) << "Gram-Schmidt test failed at (" << i << ", " << j << "). Actual: " << dot_product;
        }
    }
    std::cout << "Gram-Schmidt test completed successfully." << std::endl;
}

TEST_F(BasicTest, GenUnitvecTest) {
    VectorObj<double> e;
    basic::genUnitvec(1, 3, e);
    ASSERT_NEAR(e[0], 0, 1e-6);
    ASSERT_NEAR(e[1], 1, 1e-6);
    ASSERT_NEAR(e[2], 0, 1e-6);
}

TEST_F(BasicTest, GenUnitMatxTest) {
    MatrixObj<double> I;
    basic::genUnitMatx(3, 3, I);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j) {
                ASSERT_NEAR(I[i * 3 + j], 1, 1e-6);
            } else {
                ASSERT_NEAR(I[i * 3 + j], 0, 1e-6);
            }
        }
    }
}

TEST_F(BasicTest, SignTest) {
    ASSERT_GT(basic::sign(5.0), 0);
    ASSERT_LT(basic::sign(-3.0), 0);
}

TEST_F(BasicTest, HouseHTest) {
    MatrixObj<double> H;
    basic::houseH(A, H, 0);
    // Add assertions to check if H is a valid Householder matrix
    // This might involve checking orthogonality and other properties
}

} // namespace