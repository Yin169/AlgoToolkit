#include <vector>
#include "DenseObj.hpp"
#include "KrylovSubspace.hpp" // Include the Arnoldi header file
#include "basic.hpp"
#include <gtest/gtest.h>

// Test Fixture for Arnoldi Tests
class ArnoldiTest : public ::testing::Test {
protected:
    void SetUp() override {
        tol = 1e-8;
        n = 4;
        A = DenseObj<double>(n, n);
        Q = std::vector<VectorObj<double>>(n + 1, VectorObj<double>(n));
        H = DenseObj<double>(n + 1, n);

        // Initialize matrix A as an identity matrix
        initializeIdentityMatrix(A);

        // Initialize first vector in Q
        r = VectorObj<double>(n);
        initializeVector(r, 1.0 / std::sqrt(static_cast<double>(n)));
        r.normalize();
        Q[0] = r;
    }

    // Helper functions for initialization
    void initializeIdentityMatrix(DenseObj<double>& matrix) {
        for (size_t i = 0; i < n; ++i) {
            matrix(i, i) = 1.0;
        }
    }

    void initializeVector(VectorObj<double>& vec, double value) {
        for (size_t i = 0; i < n; ++i) {
            vec[i] = value;
        }
    }

    double tol;
    size_t n;
    DenseObj<double> A;
    std::vector<VectorObj<double>> Q;
    DenseObj<double> H;
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

    for (size_t i = 0; i < H.getRows(); ++i) {
        for (size_t j = 0; j < H.getCols(); ++j) {
            if (i > j + 1) {
                EXPECT_NEAR(H(i, j), 0.0, tol) << "Hessenberg matrix entry H[" << i << "][" << j << "] is non-zero.";
            }
        }
    }
}

// Helper function to initialize zero matrix
void initializeZeroMatrix(DenseObj<double>& matrix) {
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            matrix(i, j) = 0.0;
        }
    }
}

// Helper functions for matrix initialization and copying
void initializeDiagonalMatrix(DenseObj<double>& matrix) {
    for (int i = 0; i < matrix.getRows(); ++i) {
        matrix(i, i) = static_cast<double>(i + 1);
    }
}

// Test for breakdown tolerance
TEST_F(ArnoldiTest, BreakdownTolerance) {
    // Modify A to induce breakdown (e.g., zero matrix)
    A = DenseObj<double>(n, n); // Reset A
    initializeZeroMatrix(A);

    EXPECT_NO_THROW(Krylov::Arnoldi(A, Q, H, tol)) << "Arnoldi iteration should handle breakdown gracefully.";
}

// Test for specific example with known results
TEST_F(ArnoldiTest, KnownExample) {
    // Define a simple diagonal matrix
    initializeDiagonalMatrix(A);

    Krylov::Arnoldi(A, Q, H, tol);

    DenseObj<double> HA(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            HA(i, j) = H(i, j);
        }
    }

    Q.pop_back();
    DenseObj matQ(Q, Q[0].size(), Q.size());
    DenseObj<double> checkLeft = A * matQ;
    DenseObj<double> checkRight = matQ * HA;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << i << " " << j << " " << std::endl;
            EXPECT_NEAR(checkLeft(i, j), checkRight(i, j), tol) << "Mismatch at position (" << i << ", " << j << ")";
        }
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
