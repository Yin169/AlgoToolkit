#include <benchmark/benchmark.h>
#include "basic.hpp"
#include "LU.hpp"
#include "DenseObj.hpp"
#include "VectorObj.hpp"

// Benchmark for powerIter
static void BM_PowerIteration(benchmark::State& state) {
    int n = state.range(0);
    DenseObj<double> A(n, n);
    VectorObj<double> b(n);

    // Initialize matrix A and vector b with some values
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = i + j + 1; // Example initialization
        }
        b[i] = i + 1; // Example initialization
    }

    for (auto _ : state) {
        basic::powerIter<double, DenseObj<double>>(A, b, 10); // Perform 10 iterations
        benchmark::DoNotOptimize(b); // Prevent compiler optimizations
    }
}
BENCHMARK(BM_PowerIteration)->Range(8, 1 << 10); // Test sizes from 8 to 1024

// Benchmark for rayleighQuotient
static void BM_RayleighQuotient(benchmark::State& state) {
    int n = state.range(0);
    DenseObj<double> A(n, n);
    VectorObj<double> b(n);

    // Initialize matrix A and vector b with some values
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = i + j + 1; // Example initialization
        }
        b[i] = i + 1; // Example initialization
    }

    for (auto _ : state) {
        double result = basic::rayleighQuotient<double, DenseObj<double>>(A, b);
        benchmark::DoNotOptimize(result); // Prevent compiler optimizations
    }
}
BENCHMARK(BM_RayleighQuotient)->Range(8, 1 << 10); // Test sizes from 8 to 1024

// Benchmark for qrFactorization
static void BM_QRFactorization(benchmark::State& state) {
    int n = state.range(0);
    DenseObj<double> A(n, n);
    DenseObj<double> Q(n, n);
    DenseObj<double> R(n, n);

    // Initialize matrix A with some values
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = i + j + 1; // Example initialization
        }
    }

    for (auto _ : state) {
        basic::qrFactorization<double, DenseObj<double>>(A, Q, R);
        benchmark::DoNotOptimize(Q); // Prevent compiler optimizations
        benchmark::DoNotOptimize(R); // Prevent compiler optimizations
    }
}
BENCHMARK(BM_QRFactorization)->Range(8, 1 << 10); // Test sizes from 8 to 1024

// Benchmark for PivotLU
static void BM_PivotLU(benchmark::State& state) {
    int n = state.range(0);
    DenseObj<double> A(n, n);
    std::vector<int> P;

    // Initialize matrix A with some values
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = i%12 + j%34 + 1; // Example initialization
        }
    }

    for (auto _ : state) {
        LU::PivotLU<double, DenseObj<double>>(A, P);
        benchmark::DoNotOptimize(A); // Prevent compiler optimizations
        benchmark::DoNotOptimize(P); // Prevent compiler optimizations
    }
}
BENCHMARK(BM_PivotLU)->Range(8, 1 << 10); // Test sizes from 8 to 1024

// Main function for Google Benchmark
BENCHMARK_MAIN();