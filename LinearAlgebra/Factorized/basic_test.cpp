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

TEST_F(BasicTest, PowerIterTest) {
    VectorObj<double> b_init = b;
    basic::powerIter(A, b, 10);
    for (int i = 0; i < b.get_row(); ++i) { 
        ASSERT_NEAR(b_init[i], b[i], 1e-6) << "Power iteration test failed at index " << i << ". Actual: " << b[i];
    }
}

} // namespace
