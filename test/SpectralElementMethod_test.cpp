#include <gtest/gtest.h>
#include "../src/PDEs/SpectralElementMethod.hpp"
#include <cmath>

TEST(SpectralElementMethodTest, PoissonEquation1D) {
    // Problem: -u'' = 1 on [0,1] with u(0) = u(1) = 0
    // Analytical solution: u(x) = x(1-x)/2
    
    auto pde_op = [](const std::vector<double>& x) { return 1.0; };
    auto bc = [](const std::vector<double>& x) { return 0.0; };
    auto source = [](const std::vector<double>& x) { return 1.0; };
    auto exact = [](double x) { return x * (1 - x) / 2; };

    SpectralElementMethod<double> sem(
        4,                  // elements
        4,                  // polynomial order
        1,                  // dimension
        {0.0, 1.0},         // domain bounds
        pde_op, bc, source
    );

    VectorObj<double> solution(20); // (4 elements * 4 nodes + 1)
    sem.solve(solution);

    // Validate solution at nodes
    for (size_t i = 0; i < solution.size(); ++i) {
        double x = i / (solution.size() - 1.0);
        EXPECT_NEAR(solution[i], exact(x), 1e-4);
    }
}

TEST(SpectralElementMethodTest, LaplaceEquation2D) {
    // Problem: Δu = 0 on [0,1]×[0,1]
    // Boundary condition: u = sin(πx)sinh(πy)

    auto pde_op = [](const std::vector<double>& x) { return 1.0; };
    auto bc = [](const std::vector<double>& x) { 
        return std::sin(M_PI * x[0]) * std::sinh(M_PI * x[1]); 
    };
    auto source = [](const std::vector<double>& x) { return 0.0; };
    auto exact = [](double x, double y) { 
        return std::sin(M_PI * x) * std::sinh(M_PI * y); 
    };

    SpectralElementMethod<double> sem(
        3,                          // elements per dimension
        3,                          // polynomial order
        2,                          // dimension
        {0.0, 1.0, 0.0, 1.0},       // domain bounds
        pde_op, bc, source
    );

    size_t n = 3 * 3 + 1; // nodes per dimension
    VectorObj<double> solution(n * n);
    sem.solve(solution);

    // Validate solution at interior points
    for (size_t i = 1; i < n - 1; ++i) {
        for (size_t j = 1; j < n - 1; ++j) {
            double x = i / (n - 1.0);
            double y = j / (n - 1.0);
            EXPECT_NEAR(solution[i * n + j], exact(x, y), 1e-3);
        }
    }
}

TEST(SpectralElementMethodTest, ConvergenceTest) {
    // Verify p-convergence for 1D Poisson Equation
    auto pde_op = [](const std::vector<double>& x) { return 1.0; };
    auto bc = [](const std::vector<double>& x) { return 0.0; };
    auto source = [](const std::vector<double>& x) { return 1.0; };
    auto exact = [](double x) { return x * (1 - x) / 2; };

    std::vector<double> errors;
    for (size_t p = 2; p <= 8; ++p) {
        SpectralElementMethod<double> sem(2, p, 1, {0.0, 1.0}, pde_op, bc, source);
        VectorObj<double> solution(2 * p + 1);
        sem.solve(solution);

        // Compute L2 error
        double error = 0.0;
        for (size_t i = 0; i < solution.size(); ++i) {
            double x = i / (solution.size() - 1.0);
            error += std::pow(solution[i] - exact(x), 2);
        }
        error = std::sqrt(error / solution.size());
        errors.push_back(error);

        if (p > 2) {
            double rate = std::log(errors[p - 3] / errors[p - 2]) / std::log(p / (p - 1.0));
            EXPECT_GT(rate, p - 1); // Expect spectral convergence
        }
    }
}

TEST(SpectralElementMethodTest, BoundaryConditionTest) {
    // Verify enforcement of Dirichlet boundary conditions
    auto pde_op = [](const std::vector<double>& x) { return 1.0; };
    auto bc = [](const std::vector<double>& x) { return 1.0; };
    auto source = [](const std::vector<double>& x) { return 0.0; };

    SpectralElementMethod<double> sem(
        2,                  // elements
        3,                  // polynomial order
        1,                  // dimension
        {0.0, 1.0},         // domain bounds
        pde_op, bc, source
    );

    VectorObj<double> solution(7);
    sem.solve(solution);

    // Validate boundary values
    EXPECT_NEAR(solution[0], 1.0, 1e-10);
    EXPECT_NEAR(solution[6], 1.0, 1e-10);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
