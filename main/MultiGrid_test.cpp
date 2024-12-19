#include <gtest/gtest.h>
#include "MultiGrid.hpp"
#include "SparseObj.hpp" // For SparseMatrixCSC
#include "IterSolver.hpp" // For iterative solvers like Jacobi
#include <vector>

// Test fixture for MultiGrid
class MultiGridTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Define a simple diagonal matrix for testing
        matrix = SparseMatrixCSC<double>(4, 4);
        matrix.addValue(0, 0, 4.0);
        matrix.addValue(1, 1, 3.0);
        matrix.addValue(2, 2, 2.0);
        matrix.addValue(3, 3, 1.0);
        matrix.finalize();

        rhs = VectorObj<double>(4);
        rhs[0] = 8.0;
        rhs[1] = 9.0;
        rhs[2] = 4.0;
        rhs[3] = 1.0;

        expected_solution = VectorObj<double>(4);
        expected_solution[0] = 2.0; // Corrected indices
        expected_solution[1] = 3.0;
        expected_solution[2] = 2.0;
        expected_solution[3] = 1.0;
    }

    SparseMatrixCSC<double> matrix;
    VectorObj<double> rhs;
    VectorObj<double> expected_solution;
};

// Test MultiGrid solver for convergence
TEST_F(MultiGridTest, SolvesSimpleDiagonalMatrix) {
    AlgebraicMultiGrid<double, VectorObj<double>> mg_solver;

    VectorObj<double> solution(rhs.size(), 0.0);
    mg_solver.amgVCycle(matrix, rhs, solution, 5, 100, 1.5);

    for (size_t i = 0; i < solution.size(); ++i) {
        EXPECT_NEAR(solution[i], expected_solution[i], 1e-5)
            << "Mismatch at index " << i;
    }
}

// Test for zero RHS (should return a zero vector)
TEST_F(MultiGridTest, ZeroRHS) {
    VectorObj<double> zero_rhs(rhs.size(), 0.0);
    VectorObj<double> solution(rhs.size(), 0.0);

    AlgebraicMultiGrid<double, VectorObj<double>> mg_solver;

    mg_solver.amgVCycle(matrix, zero_rhs, solution, 5, 100, 1.5);

    for (size_t i = 0; i < solution.size(); ++i) {
        EXPECT_NEAR(solution[i], 0.0, 1e-12) << "Mismatch at index " << i;
    }
}

// Test for handling non-square matrices (should throw an exception)
TEST_F(MultiGridTest, NonSquareMatrix) {
    SparseMatrixCSC<double> non_square_matrix(2, 3);
    non_square_matrix.addValue(0, 0, 1.0);
    non_square_matrix.addValue(1, 2, 1.0);
    non_square_matrix.finalize();

    VectorObj<double> rhs_non_square(2);
    rhs_non_square[0] = 1.0;
    rhs_non_square[1] = 2.0;

    VectorObj<double> solution(rhs_non_square.size(), 0.0);

    AlgebraicMultiGrid<double, VectorObj<double>> mg_solver;

    EXPECT_THROW(mg_solver.amgVCycle(non_square_matrix, rhs_non_square, solution, 5, 100, 1.5), std::runtime_error);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}