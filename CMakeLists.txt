cmake_minimum_required(VERSION 3.10)
project(FASTSolver)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Factorized
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Preconditioner
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Solver
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Krylov
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/FastMultipole
    ${PROJECT_SOURCE_DIR}/src/LinearAlgebra/Nonlinear
    ${PROJECT_SOURCE_DIR}/src/ODE
    ${PROJECT_SOURCE_DIR}/src/PDEs
    ${PROJECT_SOURCE_DIR}/src/Intergal
    ${PROJECT_SOURCE_DIR}/src/Obj
    ${PROJECT_SOURCE_DIR}/src/Mesh
    ${PROJECT_SOURCE_DIR}/application
    ${PROJECT_SOURCE_DIR}/application/Mesh
    ${PROJECT_SOURCE_DIR}/application/PostProcess
    ${PROJECT_SOURCE_DIR}/application/LatticeBoltz
    ${PROJECT_SOURCE_DIR}/benchmark
)

# Find required packages
find_package(OpenBLAS REQUIRED)
find_package(benchmark REQUIRED)  # Google Benchmark
find_package(GTest REQUIRED)

# Define common test dependencies
set(TEST_DEPENDENCIES
    GTest::gtest 
    GTest::gmock 
    OpenBLAS::OpenBLAS
)

# Test executables
set(TEST_SOURCES
    matrix_obj_test
    basic_test
    debuglogger
    itersolver_test
    LU_test
    Cholesky_test
    KrylovSubspace_test
    MultiGrid_test
    ConjugateGradient_test
    SparseMatrixCSCTest
    ILU_test
    GMRES_test
    RungeKutta_test
    GaussianQuad_test
    NewtonMethod_test
    MeshObj_test
    Visual_test
    SVD_test
    FMM_test
)

# Add test executables
foreach(test_source ${TEST_SOURCES})
    add_executable(${test_source} test/${test_source}.cpp)
    target_link_libraries(${test_source} PRIVATE ${TEST_DEPENDENCIES})
    add_test(NAME ${test_source} COMMAND ${test_source})
endforeach()

# Enable testing
enable_testing()

# Benchmark executable
set(BENCHMARK_SOURCES
    bench_Obj
    bench_LU
    bench_Solver
    bench_ODE
    MainTest
    # CylinderFlow2D
    # JetFlow_2D
)

# Add benchmark executables
foreach(benchmark_source ${BENCHMARK_SOURCES})
    add_executable(${benchmark_source} benchmark/${benchmark_source}.cpp)
    target_link_libraries(${benchmark_source} PRIVATE benchmark::benchmark OpenBLAS::OpenBLAS)
endforeach()
