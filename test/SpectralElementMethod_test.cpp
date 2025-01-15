#include <gtest/gtest.h>
#include "SpectralElementMethod.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

// Helper functions for tests
namespace {
    template<typename T>
    T l2_error(const std::vector<T>& computed, const std::vector<T>& exact) {
        T error = 0.0;
        for (size_t i = 0; i < computed.size(); ++i) {
            error += (computed[i] - exact[i]) * (computed[i] - exact[i]);
        }
        return std::sqrt(error / computed.size());
    }
}

TEST(SpectralElementMethodTest, PoissonEquation1D) {
    // Problem: u'' = 1 on [0,1] with u(0) = u(1) = 0
    // Analytical solution: u(x) = x(x - 1)/2
    
    auto pde_op = [](const std::vector<double>& x) { return 1.0; };
    auto bc = [](const std::vector<double>& x) { return 0.0; };
    auto source = [](const std::vector<double>& x) { return 1.0; };
    auto exact = [](double x) { return (x * (x - 1.0)) / 2.0; };

    const size_t num_elements = 12;
    const size_t poly_order = 12;
    const size_t num_nodes = num_elements * (poly_order + 1);

    SpectralElementMethod<double> sem(
        num_elements,
        poly_order,
        1,                  // dimension
        {0.0, 1.0},        // domain bounds
        "",                // mesh file
        pde_op, bc, source
    );

    VectorObj<double> solution(num_nodes);
    sem.solve(solution);

    std::vector<double> global_solution;
    for (int i=0; i<solution.size(); i++){
        if ( i==0 ||global_solution.back()!= solution[i]){
            global_solution.push_back(solution[i]);
        }
    }

    // Validate solution at nodes
    std::vector<double> exact_solution(num_nodes);
    for (size_t i = 0; i < global_solution.size(); ++i) {
        double x = static_cast<double>(i) / (num_nodes - 1);
        exact_solution[i] = exact(x);
        EXPECT_NEAR(global_solution[i], exact_solution[i], 1e-4) 
            << "Failed at node " << i << " (x = " << x << ")";
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

    const size_t num_elements = 8;
    const size_t poly_order = 2;
    const size_t nodes_per_dim = num_elements * (poly_order + 1);
    const size_t total_nodes = nodes_per_dim * nodes_per_dim;

    SpectralElementMethod<double> sem(
        num_elements,
        poly_order,
        2,                          // dimension
        {0.0, 1.0, 0.0, 1.0},      // domain bounds
        "",                         // mesh file
        pde_op, bc, source
    );

    VectorObj<double> solution(total_nodes);
    sem.solve(solution);

    // Validate solution at interior points
    std::vector<double> exact_solution(total_nodes);
    for (size_t i = 0; i < nodes_per_dim; ++i) {
        for (size_t j = 0; j < nodes_per_dim; ++j) {
            double x = static_cast<double>(i) / (nodes_per_dim - 1);
            double y = static_cast<double>(j) / (nodes_per_dim - 1);
            size_t idx = i * nodes_per_dim + j;
            exact_solution[idx] = exact(x, y);
            EXPECT_NEAR(solution[idx], exact_solution[idx], 1e-3)
                << "Failed at node (" << i << "," << j << ")";
        }
    }
}

TEST(SpectralElementMethodTest, ConvergenceTest) {
    auto pde_op = [](const std::vector<double>& x) { return 1.0; };
    auto bc = [](const std::vector<double>& x) { return 0.0; };
    auto source = [](const std::vector<double>& x) { return 1.0; };
    auto exact = [](double x) { return x * (x - 1) / 2; };

    std::vector<double> errors;
    for (size_t p = 1; p <= 4; ++p) {
        const size_t num_elements = 12;
        const size_t num_nodes = num_elements * (p + 1);

        SpectralElementMethod<double> sem(
            num_elements,
            p,
            1,
            {0.0, 1.0},
            "",
            pde_op, bc, source
        );

        VectorObj<double> solution(num_nodes);
        sem.solve(solution);

        for (size_t i = 0; i < num_nodes; ++i) {
            double x = static_cast<double>(i) / (num_nodes - 1);
            EXPECT_NEAR(solution[i], exact(x), 1e-3) << " i: " << i << std::endl;
        }

    }
}

TEST(SpectralElementMethodTest, BoundaryConditionTest) {
    auto pde_op = [](const std::vector<double>& x) { return 1.0; };
    auto bc = [](const std::vector<double>& x) { 
        return std::sin(M_PI * x[0]); // Non-zero boundary condition
    };
    auto source = [](const std::vector<double>& x) { return 1.0; };

    const size_t num_elements = 12;
    const size_t poly_order = 2;
    const size_t num_nodes = num_elements * (poly_order + 1);

    SpectralElementMethod<double> sem(
        num_elements,
        poly_order,
        1,
        {0.0, 1.0},
        "",
        pde_op, bc, source
    );

    VectorObj<double> solution(num_nodes);
    sem.solve(solution);

    // Check boundary values
    EXPECT_NEAR(solution[0], bc({0.0}), 1e-4);
    EXPECT_NEAR(solution[num_nodes-1], bc({1.0}), 1e-4);
}


// TEST(SpectralElementMethodTest, MeshFileTest) {
//     auto pde_op = [](const std::vector<double>& x) { return 1.0; };
//     auto bc = [](const std::vector<double>& x) { return 0.0; };
//     auto source = [](const std::vector<double>& x) { return 1.0; };

//     // Test non-existent mesh file
//     EXPECT_THROW({
//         SpectralElementMethod<double> sem(
//             12,
//             2,
//             1,
//             {0.0, 1.0},
//             "nonexistent.su2",
//             pde_op, bc, source
//         );
//     }, std::runtime_error);
// }

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}