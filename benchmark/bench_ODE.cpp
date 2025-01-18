#include <benchmark/benchmark.h>
#include "RungeKutta.hpp"
#include "GaussianQuad.hpp"
#include "DenseObj.hpp"
#include "VectorObj.hpp"
#include <functional>

// Benchmark for RungeKutta::solve
static void BM_RungeKuttaSolve(benchmark::State& state) {
    int n = state.range(0); // Problem size
    VectorObj<double> y(n);
    for (int i = 0; i < n; ++i) {
        y[i] = i + 1; // Initialize state vector
    }

    // Define the system function (e.g., a simple linear system)
    auto f = [](const VectorObj<double>& y) {
        VectorObj<double> dy(y.size());
        for (int i = 0; i < y.size(); ++i) {
            dy[i] = -y[i]; // Example: dy/dt = -y
        }
        return dy;
    };

    RungeKutta<double, VectorObj<double>, VectorObj<double>> rk;

    for (auto _ : state) {
        rk.solve(y, f, 0.01, 100); // Step size = 0.01, 100 steps
        benchmark::DoNotOptimize(y); // Prevent compiler optimizations
    }
}
BENCHMARK(BM_RungeKuttaSolve)->Range(8, 1 << 10); // Test sizes from 8 to 1024

// Benchmark for GaussianQuadrature::integrate
static void BM_GaussianQuadratureIntegrate(benchmark::State& state) {
    int n = state.range(0); // Number of quadrature points

    // Define the function to integrate (e.g., f(x) = x^2)
    auto f = [](double x) {
        return x * x;
    };

    Quadrature::GaussianQuadrature<double> gq(n);

    for (auto _ : state) {
        double result = gq.integrate(f, 0.0, 1.0); // Integrate f(x) = x^2 from 0 to 1
        benchmark::DoNotOptimize(result); // Prevent compiler optimizations
    }
}
BENCHMARK(BM_GaussianQuadratureIntegrate)->Range(8, 1 << 10); // Test sizes from 8 to 1024

// Main function for Google Benchmark
BENCHMARK_MAIN();