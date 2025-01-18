#include <benchmark/benchmark.h>
#include "LU.hpp"
#include "MultiGrid.hpp"
#include "KrylovSubspace.hpp"
#include "GMRES.hpp"
#include "ConjugateGradient.hpp"
#include "DenseObj.hpp"
#include "VectorObj.hpp"
#include "SparseObj.hpp"

// Benchmark for AMG V-cycle
static void BM_AMGVCycle(benchmark::State& state) {
    int n = state.range(0);
    SparseMatrixCSC<double> A(n, n);
    VectorObj<double> b(n);
    VectorObj<double> x(n, 0.0);

    // Initialize matrix A and vector b with some values
    for (int i = 0; i < n; ++i) {
        A.addValue(i, i, i + 1); // Diagonal matrix
        b[i] = i + 1; // Example initialization
    }
    A.finalize();

    AlgebraicMultiGrid<double, VectorObj<double>> amg;

    for (auto _ : state) {
        amg.amgVCycle(A, b, x, 3, 5, 0.25); // Perform AMG V-cycle
        benchmark::DoNotOptimize(x); // Prevent compiler optimizations
    }
}
BENCHMARK(BM_AMGVCycle)->Range(8, 1 << 10); // Test sizes from 8 to 1024

// Benchmark for Arnoldi iteration
static void BM_Arnoldi(benchmark::State& state) {
    int n = state.range(0);
    DenseObj<double> A(n, n);
    std::vector<VectorObj<double>> Q(n + 1, VectorObj<double>(n));
    DenseObj<double> H(n + 1, n);

    // Initialize matrix A with some values
    for (int i = 0; i < n; ++i) {
        A(i, i) = i + 1; // Diagonal matrix
    }

    // Initialize Q[0] with some values
    for (int i = 0; i < n; ++i) {
        Q[0][i] = i + 1; // Example initialization
    }

    for (auto _ : state) {
        Krylov::Arnoldi(A, Q, H, 1e-6); // Perform Arnoldi iteration
        benchmark::DoNotOptimize(Q); // Prevent compiler optimizations
        benchmark::DoNotOptimize(H); // Prevent compiler optimizations
    }
}
BENCHMARK(BM_Arnoldi)->Range(8, 1 << 10); // Test sizes from 8 to 1024

// Benchmark for Conjugate Gradient solve
static void BM_ConjugateGradient(benchmark::State& state) {
    int n = state.range(0);
    SparseMatrixCSC<double> A(n, n);
    VectorObj<double> b(n);
    VectorObj<double> x(n, 0.0);

    // Initialize matrix A and vector b with some values
    for (int i = 0; i < n; ++i) {
        A.addValue(i, i, i + 1); // Diagonal matrix
        b[i] = i + 1; // Example initialization
    }
    A.finalize();

    ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>> cg(A, b, 100, 1e-6);

    for (auto _ : state) {
        cg.solve(x); // Perform Conjugate Gradient solve
        benchmark::DoNotOptimize(x); // Prevent compiler optimizations
    }
}
BENCHMARK(BM_ConjugateGradient)->Range(8, 1 << 10); // Test sizes from 8 to 1024

// Benchmark for GMRES solve
static void BM_GMRES(benchmark::State& state) {
    int n = state.range(0);
    DenseObj<double> A(n, n);
    VectorObj<double> b(n);
    VectorObj<double> x(n, 0.0);

    // Initialize matrix A and vector b with some values
    for (int i = 0; i < n; ++i) {
        A(i, i) = i + 1; // Diagonal matrix
        b[i] = i + 1; // Example initialization
    }
	DenseObj<double> L(n,n), U(n,n);
	LU::ILU<double, DenseObj<double>>(A, L, U);
    GMRES<double, DenseObj<double>, VectorObj<double>> gmres;

    for (auto _ : state) {
        gmres.solve(L*U*A, L*U*b, x, 100, n, 1e-6); // Perform GMRES solve
        benchmark::DoNotOptimize(x); // Prevent compiler optimizations
    }
}
BENCHMARK(BM_GMRES)->Range(8, 1 << 10); // Test sizes from 8 to 1024

// Main function for Google Benchmark
BENCHMARK_MAIN();