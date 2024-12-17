#include <benchmark/benchmark.h>
#include "basic.hpp"
#include "MatrixObj.hpp"
#include "VectorObj.hpp"

// Helper function to generate a random MatrixObj
template <typename TNum>
MatrixObj<TNum> GenerateRandomMatrixObj(int rows, int cols) {
    MatrixObj<TNum> mat(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<TNum>(rand()) / RAND_MAX;
    }
    return mat;
}

// Helper function to generate a random VectorObj
template <typename TNum>
VectorObj<TNum> GenerateRandomVectorObj(int size) {
    VectorObj<TNum> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = static_cast<TNum>(rand()) / RAND_MAX;
    }
    return vec;
}

// Benchmark for powerIter
static void BM_PowerIter(benchmark::State& state) {
    MatrixObj<double> A = GenerateRandomMatrixObj<double>(100, 100);
    VectorObj<double> b = GenerateRandomVectorObj<double>(100);
    int max_iter_num = 100;

    for (auto _ : state) {
        basic::powerIter(A, b, max_iter_num);
    }
}
BENCHMARK(BM_PowerIter);

// Benchmark for rayleighQuotient
static void BM_RayleighQuotient(benchmark::State& state) {
    MatrixObj<double> A = GenerateRandomMatrixObj<double>(100, 100);
    VectorObj<double> b = GenerateRandomVectorObj<double>(100);

    for (auto _ : state) {
        benchmark::DoNotOptimize(basic::rayleighQuotient(A, b));
    }
}
BENCHMARK(BM_RayleighQuotient);

// Benchmark for gramSmith
static void BM_GramSmith(benchmark::State& state) {
    MatrixObj<double> A = GenerateRandomMatrixObj<double>(100, 100);
    std::vector<VectorObj<double>> orthSet(100);

    for (auto _ : state) {
        basic::gramSchmidt(A, orthSet);
    }
}
BENCHMARK(BM_GramSmith);

// Benchmark for subtProj
static void BM_SubtProj(benchmark::State& state) {
    VectorObj<double> u = GenerateRandomVectorObj<double>(100);
    VectorObj<double> v = GenerateRandomVectorObj<double>(100);

    for (auto _ : state) {
        benchmark::DoNotOptimize(basic::subtProj(u, v));
    }
}
BENCHMARK(BM_SubtProj);

// Benchmark for genUnitvec
static void BM_GenUnitvec(benchmark::State& state) {
    VectorObj<double> e(100);

    for (auto _ : state) {
        basic::genUnitVec(50, 100, e); // Assuming index 50
    }
}
BENCHMARK(BM_GenUnitvec);

// Benchmark for genUnitMatx
static void BM_GenUnitMatx(benchmark::State& state) {
    MatrixObj<double> I;

    for (auto _ : state) {
        basic::genUnitMatx(100, 100, I);
    }
}
BENCHMARK(BM_GenUnitMatx);

// Benchmark for houseH
static void BM_HouseH(benchmark::State& state) {
    MatrixObj<double> A = GenerateRandomMatrixObj<double>(100, 100);
    MatrixObj<double> H;

    for (auto _ : state) {
        basic::houseH(A, H, 50); // Assuming index 50
    }
}
BENCHMARK(BM_HouseH);

BENCHMARK_MAIN();
