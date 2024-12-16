// GTest for Arnoldi Iteration Implementation
#include <vector>
#include "KrylovSubspace.hpp" // Include the Arnoldi header file
#include "basic.hpp"
#include <gtest/gtest.h>

// Test Fixture for Arnoldi Tests
class ArnoldiTest : public ::testing::Test {
protected:
    void SetUp() override {
        tol = 1e-8;
        n = 4;
        A = MatrixObj<double>(n, n);
        Q = std::vector<VectorObj<double>>(n + 1, VectorObj<double>(n));
        H = MatrixObj<double>(n + 1, n);

        // Example initialization of a simple matrix A (identity matrix)
        for (size_t i = 0; i < n; ++i) {
            A(i, i) = 1.0;
        }

        r = VectorObj<double>(n);
        // Example initialization of the first vector in Q
        for (size_t i = 0; i < n; ++i) {
            r[i] = 1.0 / std::sqrt(static_cast<double>(n));
        }
        r.normalized();
        Q[0] = r;
    }

    double tol;
    size_t n;
    MatrixObj<double> A;
    std::vector<VectorObj<double>> Q;
    MatrixObj<double> H;
    VectorObj<double> r;
};

// Test for Arnoldi orthogonality
TEST_F(ArnoldiTest, OrthogonalityCheck) {
    Krylov::Arnoldi(A, Q, H, tol);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dot = Q[i] * Q[j];
            EXPECT_NEAR(dot, 0.0, tol) << "Vectors Q[" << i << "] and Q[" << j << "] are not orthogonal.";
        }
    }
}
// Test for Hessenberg matrix correctness
TEST_F(ArnoldiTest, HessenbergStructure) {
    Krylov::Arnoldi(A, Q, H, tol);

    for (size_t i = 0; i < H.get_row(); ++i) {
        for (size_t j = 0; j < H.get_col(); ++j) {
            if (i > j + 1) {
                EXPECT_NEAR(H(i, j), 0.0, tol) << "Hessenberg matrix entry H[" << i << "][" << j << "] is non-zero.";
            }
        }
    }
}

// Test for breakdown tolerance
TEST_F(ArnoldiTest, BreakdownTolerance) {
    // Modify A to induce breakdown (e.g., zero matrix)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = 0.0;
        }
    }

    EXPECT_NO_THROW(Krylov::Arnoldi(A, Q, H, tol)) << "Arnoldi iteration should handle breakdown gracefully.";
}

// Test for specific example with known results
TEST_F(ArnoldiTest, KnownExample) {
    // Define a simple diagonal matrix
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = static_cast<double>(i+1);
    }

    Krylov::Arnoldi(A, Q, H, tol);

    MatrixObj<double> HA(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            HA(i, j) = H(i, j);
        }
    }

    Q.pop_back();
    MatrixObj matQ(Q);
    MatrixObj<double> checkLeft = A * matQ ;
    MatrixObj<double> checkRight = matQ * HA;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            EXPECT_NEAR(checkLeft(i, j), checkRight(i, j), tol) <<  i << " " << j << " does not match the expected value.";
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
