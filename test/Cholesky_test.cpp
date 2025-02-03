#include <gtest/gtest.h>
#include "../src/LinearAlgebra/Factorized/basic.hpp"
#include "../src/Obj/DenseObj.hpp"
#include <cmath>

class CholeskyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize a 3x3 symmetric positive definite matrix
        spd_matrix = DenseObj<double>(3, 3);
        double data[9] = {
            4.0, 1.0, 1.0,
            1.0, 5.0, 2.0,
            1.0, 2.0, 6.0
        };
        std::copy(std::begin(data), std::end(data), spd_matrix.data());

        // Expected Cholesky factor L
        expected_L = DenseObj<double>(3, 3);
        double l_data[9] = {
            2.0, 0.0, 0.0,
            0.5, 2.179449, 0.0,
            0.5, 0.803964, 2.236068
        };
        std::copy(std::begin(l_data), std::end(l_data), expected_L.data());
    }

    DenseObj<double> spd_matrix;
    DenseObj<double> expected_L;
};

TEST_F(CholeskyTest, CorrectDecomposition) {
    DenseObj<double> L;
    basic::Cholesky(spd_matrix, L);

    // Check dimensions
    EXPECT_EQ(L.getRows(), 3);
    EXPECT_EQ(L.getCols(), 3);

	DenseObj<double> recronstured_A = L * L.Transpose();

    // Check L values against expected values
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(recronstured_A(i, j), spd_matrix(i, j), 1e-6)
                << "Mismatch at position (" << i << ", " << j << ")";
        }
    }
}

TEST_F(CholeskyTest, ReconstructionTest) {
    SparseMatrixCSC<double> L;
	SparseMatrixCSC<double> sparse_spd_matrix(spd_matrix);
	
	for (size_t i=0; i<sparse_spd_matrix.getRows(); i++){
		for (size_t j=0; j<sparse_spd_matrix.getCols(); j++){
			std::cout << sparse_spd_matrix(i, j) << " ";
		}
		std::cout << std::endl;
	}
    basic::Cholesky<double, SparseMatrixCSC<double>>(sparse_spd_matrix, L);

    // Compute L * L^T
    SparseMatrixCSC<double> Lt = L.Transpose();
    SparseMatrixCSC<double> reconstructed = L * Lt;

    // Compare with original matrix
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(reconstructed(i, j), spd_matrix(i, j), 1e-6)
                << "Reconstruction failed at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(CholeskyTest, NonSquareMatrix) {
    DenseObj<double> non_square(2, 3);
    DenseObj<double> L;
    EXPECT_THROW(basic::Cholesky(non_square, L), std::invalid_argument);
}

TEST_F(CholeskyTest, NonPositiveDefiniteMatrix) {
    // Create a non-positive definite matrix
    DenseObj<double> non_pd(2, 2);
    double data[4] = {
        1.0, 2.0,
        2.0, 1.0  // This makes the matrix non-positive definite
    };
    std::copy(std::begin(data), std::end(data), non_pd.data());

    DenseObj<double> L;
    EXPECT_THROW(basic::Cholesky(non_pd, L), std::runtime_error);
}

TEST_F(CholeskyTest, LargeMatrixTest) {
    const int n = 100;
    DenseObj<double> large_matrix(n, n);
    
    // Create a large SPD matrix: A = B * B^T + nI
    // Fill diagonal with n and upper triangle with 1.0
    for (int i = 0; i < n; ++i) {
        large_matrix.addValue(i, i, n);  // Diagonal
        for (int j = i + 1; j < n; ++j) {
            large_matrix.addValue(i, j, 1.0);  // Upper triangle
            large_matrix.addValue(j, i, 1.0);  // Lower triangle (symmetry)
        }
    }
    large_matrix.finalize();

    // Perform Cholesky decomposition
    DenseObj<double> L;
    EXPECT_NO_THROW(basic::Cholesky(large_matrix, L));

    // Verify reconstruction
    DenseObj<double> reconstructed = L * L.Transpose();

    // Check if reconstruction matches original matrix
    double tolerance = 1e-6;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(reconstructed(i, j), large_matrix(i, j), tolerance)
                << "Large matrix reconstruction failed at (" << i << ", " << j << ")";
        }
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}